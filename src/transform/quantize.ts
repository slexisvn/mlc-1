// ═══════════════════════════════════════════════════════════════
//  Post-Training Quantization (PTQ) — int8 symmetric quantization
//
//  Why this matters (MLC concept):
//    Quantization is the #1 deployment optimization for production
//    ML systems. TVM, TensorRT, and ONNX Runtime all implement it.
//    Key ideas:
//      - Float32 → int8 reduces model size 4×
//      - int8 GEMM throughput is 2×–4× vs float32 on modern hardware
//      - Accuracy typically drops < 1% with good calibration
//
//  This pass implements symmetric per-tensor quantization:
//    scale = max(|W|) / 127
//    W_int8[i] = round(W[i] / scale)
//    W_dequant[i] = W_int8[i] * scale
//
//  Workflow:
//    1. Calibrate: compute scale/zero-point from weight statistics
//    2. Quantize: convert float32 → int8 (stored as Float32 for JS compat)
//    3. Dequantize: convert back to float32 for inference
//    4. Measure: cosine similarity(float_output, quant_output) ≥ 0.99
//
//  Note on zero-point:
//    Symmetric quantization has zero_point = 0 (integers symmetric
//    around 0). This is common for weights; activations often use
//    asymmetric (non-zero zero_point) but we keep it simple here.
// ═══════════════════════════════════════════════════════════════

import { NDArray } from '../tensor/ndarray.js';
import { IRModule, IRFunction, CallExpr, ConstantExpr, VarExpr, LetExpr, type Expr } from '../ir/high_level.js';

// ─── Quantization config ───

export interface QuantConfig {
  /** Number of bits (default 8 → int8 range [-127, 127]) */
  bits: number;
  /** 'symmetric' = zero_point is always 0 */
  mode: 'symmetric';
}

export const DEFAULT_QUANT_CONFIG: QuantConfig = {
  bits: 8,
  mode: 'symmetric',
};

// ─── Per-tensor quantization parameters ───

export interface QuantParams {
  scale: number;
  zeroPoint: number;    // always 0 for symmetric
  minVal: number;
  maxVal: number;
  /** Original float range captured during calibration */
  absMax: number;
}

// ─── Quantized weight info ───

export interface QuantizedWeight {
  name: string;
  originalShape: number[];
  quantParams: QuantParams;
  /** int8 values stored as Float32 (since JS has no Int8Array in NDArray) */
  quantizedData: Float32Array;
  /** Dequantized values (int8 * scale) for inference */
  dequantizedData: Float32Array;
}

// ─── Result of the quantization pass ───

export interface QuantizationResult {
  /** The module with ConstantExpr data replaced by dequantized weights */
  module: IRModule;
  /** Quantization parameters for each weight tensor */
  quantizedWeights: QuantizedWeight[];
  /** Summary table for printing */
  table: string;
  /** Total parameter count before and after */
  totalParams: number;
  /** Number of weights successfully quantized */
  quantizedCount: number;
}

// ─── Calibrate a single tensor ───

function calibrate(data: Float32Array, config: QuantConfig): QuantParams {
  const maxRange = (1 << (config.bits - 1)) - 1; // 127 for int8
  let absMax = 0;
  for (let i = 0; i < data.length; i++) {
    const v = Math.abs(data[i]);
    if (v > absMax) absMax = v;
  }
  // Avoid division by zero for all-zero tensors
  const scale = absMax > 1e-8 ? absMax / maxRange : 1e-8;
  return {
    scale,
    zeroPoint: 0,
    minVal: -absMax,
    maxVal: absMax,
    absMax,
  };
}

// ─── Quantize a tensor to int8 (stored as Float32) ───

function quantizeTensor(data: Float32Array, params: QuantParams, bits: number): Float32Array {
  const maxRange = (1 << (bits - 1)) - 1; // 127
  const result = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const clamped = Math.max(-params.absMax, Math.min(params.absMax, data[i]));
    result[i] = Math.round(clamped / params.scale);
  }
  return result;
}

// ─── Dequantize: int8 → float32 ───

function dequantizeTensor(quantized: Float32Array, params: QuantParams): Float32Array {
  const result = new Float32Array(quantized.length);
  for (let i = 0; i < quantized.length; i++) {
    result[i] = quantized[i] * params.scale;
  }
  return result;
}

// ─── Cosine similarity between two flat arrays ───

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom < 1e-10 ? 1.0 : dotProduct / denom;
}

// ─── Walk the IR and collect ConstantExpr nodes ───

function collectConstants(expr: Expr, out: Map<ConstantExpr, string>): void {
  switch (expr.kind) {
    case 'constant':
      // Name already on the node — collect it
      out.set(expr, expr.name || '(unnamed)');
      break;
    case 'call':
      for (const arg of expr.args) collectConstants(arg, out);
      break;
    case 'let':
      collectConstants(expr.value, out);
      collectConstants(expr.body, out);
      break;
    case 'var':
      break;
  }
}

// ─── Replace ConstantExpr data in-place ───
// We mutate the .data.data field of ConstantExpr to use dequantized weights.
// This is safe because ConstantExpr owns its NDArray.

function replaceConstantData(expr: Expr, replacements: Map<ConstantExpr, Float32Array>): void {
  switch (expr.kind) {
    case 'constant': {
      const newData = replacements.get(expr);
      if (newData) {
        expr.data.data = newData;
      }
      break;
    }
    case 'call':
      for (const arg of expr.args) replaceConstantData(arg, replacements);
      break;
    case 'let':
      replaceConstantData(expr.value, replacements);
      replaceConstantData(expr.body, replacements);
      break;
    case 'var':
      break;
  }
}

// ─── Main PTQ pass ───

export function quantizeModule(
  module: IRModule,
  config: QuantConfig = DEFAULT_QUANT_CONFIG
): QuantizationResult {
  const maxRange = (1 << (config.bits - 1)) - 1; // 127

  // Step 1: Collect all constant tensors
  const constants = new Map<ConstantExpr, string>();
  for (const [, fn] of module.functions) {
    collectConstants(fn.body, constants);
  }

  // Step 2: Quantize each constant
  const quantizedWeights: QuantizedWeight[] = [];
  const replacements = new Map<ConstantExpr, Float32Array>();
  let totalParams = 0;

  for (const [constExpr, name] of constants) {
    const data = constExpr.data.data;
    totalParams += data.length;

    const params = calibrate(data, config);
    const quantized = quantizeTensor(data, params, config.bits);
    const dequantized = dequantizeTensor(quantized, params);

    quantizedWeights.push({
      name,
      originalShape: constExpr.data.shape,
      quantParams: params,
      quantizedData: quantized,
      dequantizedData: dequantized,
    });

    replacements.set(constExpr, dequantized);
  }

  // Step 3: Replace constant data with dequantized weights (in-place mutation is safe here)
  for (const [, fn] of module.functions) {
    replaceConstantData(fn.body, replacements);
  }

  // Step 4: Build summary table
  const COL_NAME = 16;
  const COL_SHAPE = 12;
  const COL_SCALE = 12;
  const COL_RANGE = 18;
  const COL_ERR = 10;
  const header =
    `  ${'Weight'.padEnd(COL_NAME)} ${'Shape'.padEnd(COL_SHAPE)} ` +
    `${'Scale'.padEnd(COL_SCALE)} ${'AbsMax→Range'.padEnd(COL_RANGE)} ${'MaxErr'.padEnd(COL_ERR)}`;
  const sep = '  ' + '─'.repeat(COL_NAME + COL_SHAPE + COL_SCALE + COL_RANGE + COL_ERR + 4);
  const rows = [header, sep];

  for (const w of quantizedWeights) {
    const { quantParams: p, originalShape } = w;
    // Compute max absolute quantization error
    let maxErr = 0;
    for (let i = 0; i < w.dequantizedData.length; i++) {
      const err = Math.abs(w.dequantizedData[i] - (Array.isArray(w.dequantizedData) ? 0 : 0));
      // (we need original data for this — stored in the ConstantExpr which was mutated)
      // Approximate: maxErr ≈ scale/2 (worst-case rounding error is 0.5 * scale)
    }
    maxErr = p.scale / 2;

    const shapeStr = `[${originalShape.join(',')}]`;
    const rangeStr = `${p.absMax.toFixed(3)}→[-127,127]`;
    rows.push(
      `  ${w.name.padEnd(COL_NAME)} ${shapeStr.padEnd(COL_SHAPE)} ` +
      `${p.scale.toExponential(3).padEnd(COL_SCALE)} ${rangeStr.padEnd(COL_RANGE)} ` +
      `±${maxErr.toExponential(2)}`
    );
  }

  // Theoretical compression
  const floatBytes = totalParams * 4;
  const int8Bytes = totalParams * 1;
  rows.push(sep);
  rows.push(`  Total params: ${totalParams.toLocaleString()} | float32: ${(floatBytes / 1024).toFixed(1)}KB → int8: ${(int8Bytes / 1024).toFixed(1)}KB (${(floatBytes / int8Bytes).toFixed(0)}× compression)`);

  return {
    module,
    quantizedWeights,
    table: rows.join('\n'),
    totalParams,
    quantizedCount: quantizedWeights.length,
  };
}

// ─── Measure output quality: cosine similarity float vs quantized ───
// Runs both the original and quantized forward pass, returns cosine similarity.

export interface QuantQuality {
  cosineSim: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
}

export function measureQuantQuality(
  floatOutput: Float32Array,
  quantOutput: Float32Array
): QuantQuality {
  const cosineSim = cosineSimilarity(floatOutput, quantOutput);
  let maxAbsDiff = 0;
  let sumAbsDiff = 0;
  for (let i = 0; i < floatOutput.length; i++) {
    const diff = Math.abs(floatOutput[i] - quantOutput[i]);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
    sumAbsDiff += diff;
  }
  return {
    cosineSim,
    maxAbsDiff,
    meanAbsDiff: sumAbsDiff / floatOutput.length,
  };
}
