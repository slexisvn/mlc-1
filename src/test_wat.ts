// ═══════════════════════════════════════════════════════════════
//  test_wat.ts — End-to-end WAT execution test (multiclass classifier)
//
//  Model: Linear(32,64)→ReLU→Linear(64,8),  batch=4
//  Matches main.ts createClassifier('multiclass') exactly.
//  → 2 PrimFuncs: fused_dense_bias_relu + fused_dense_bias
//  → ~0.13 MFLOPs per forward pass
//
//  Pipeline:
//    1. Build deep model, MLC pipeline (trace → lower → arithmeticSimplify)
//    2. Generate WAT via codegenWAT (storageRewrite intentionally skipped!)
//    3. Assemble WAT → WASM binary via wabt
//    4. Instantiate in Node.js, marshal buffers, run N kernels generically
//    5. Verify WASM output matches JS reference (max delta < 1e-4)
//    6. Benchmark: JS vs WASM N-kernel forward pass
//
//  Key constraint: storageRewrite must NOT run before codegenWAT.
//  WAT codegen detects scalar accumulators via func.allocations.
//  After storageRewrite those allocs are removed and replaced by
//  ScalarDeclNode/ScalarStoreNode — which WAT codegen doesn't handle.
// ═══════════════════════════════════════════════════════════════

import WabtModule from 'wabt';
import { NDArray } from './tensor/ndarray.js';
import { Linear, ReLU, Sequential } from './model/nn.js';
import { Tracer } from './trace/tracer.js';
import { buildIR } from './ir/high_level.js';
import { constantFold } from './transform/constant_fold.js';
import { fuseOps } from './transform/op_fusion.js';
import { deadCodeElimination } from './transform/dead_code_elimination.js';
import { cseModule } from './transform/cse.js';
import { lowerModule } from './lower/lowering.js';
import { arithmeticSimplify } from './transform/arithmetic_simplify.js';
import { codegenWAT } from './codegen/wat_codegen.js';
import { compile } from './codegen/js_codegen.js';

const BATCH  = 4;
const IN     = 32;
const HID    = 64;
const OUT    = 8;
const WARMUP = 300;
const REPS   = 1000;

// ─── helpers ──────────────────────────────────────────────────

function sep(label: string) {
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`  ${label}`);
  console.log('─'.repeat(60));
}

function computeOffsets(funcs: ReturnType<typeof lowerModule>): Array<Map<string, number>> {
  // Rebuild per-function byte offsets matching the sequential layout
  // used by codegenWAT. The WATModule.bufferOffsets Map has name
  // collisions when two functions share param names (A, W, B, Out) —
  // this helper avoids that by tracking offsets per-function.
  let cursor = 0;
  return funcs.map(pf => {
    const m = new Map<string, number>();
    for (const param of pf.params) {
      m.set(param.name, cursor);
      cursor += param.shape.reduce((a, b) => a * b, 1) * 4;
    }
    return m;
  });
}

// ─── main ─────────────────────────────────────────────────────

async function main() {
  console.log('═'.repeat(60));
  console.log('  WAT Execution Test — Multiclass Classifier');
  console.log(`  Model: Linear(${IN},${HID})→ReLU→Linear(${HID},${OUT}),  batch=${BATCH}`);
  console.log(`  (matches main.ts createClassifier('multiclass'))`);
  console.log('═'.repeat(60));

  // ── 1. Build model ───────────────────────────────────────────
  sep('1. Build model');
  const model = new Sequential([
    new Linear(IN,  HID),
    new ReLU(),
    new Linear(HID, OUT),
  ]);

  // model.parameters() = [W0[64,32], B0[1,64], W1[8,64], B1[1,8]]
  const allParams = model.parameters();
  const nLayers = allParams.length / 2;
  // kernelWeights[i] / kernelBiases[i] → flat Float32Array for kernel i
  const kernelWeights = Array.from({ length: nLayers }, (_, i) => allParams[2 * i].data.data);
  const kernelBiases  = Array.from({ length: nLayers }, (_, i) => allParams[2 * i + 1].data.data);

  for (let i = 0; i < nLayers; i++) {
    const w = allParams[2 * i];
    const b = allParams[2 * i + 1];
    console.log(`  Layer ${i}: W[${w.data.shape.join('×')}], B[${b.data.shape.join('×')}]`);
  }

  // Fixed input (same array used for both WASM and JS reference)
  const inputNd  = NDArray.rand([BATCH, IN]);
  const inputArr = inputNd.data;  // Float32Array [BATCH*IN]

  // ── 2. MLC Pipeline ──────────────────────────────────────────
  sep('2. MLC Pipeline');
  console.log('  traceInference → buildIR → constantFold → fuseOps');
  console.log('  → deadCodeElimination → cseModule → lowerModule → arithmeticSimplify');

  const tracer = new Tracer();
  const graph   = tracer.traceInference(model, [BATCH, IN]);
  const irMod   = buildIR(graph);
  const folded  = constantFold(irMod);
  const fused   = fuseOps(folded);
  const { module: dceMod } = deadCodeElimination(fused);
  const { module: cseMod } = cseModule(dceMod);
  const primFuncs = lowerModule(cseMod);

  // arithmeticSimplify is safe before WAT codegen (doesn't touch allocations)
  // storageRewrite is intentionally SKIPPED — it removes alloc nodes that
  // WAT codegen needs to detect scalar accumulator buffers.
  const optFuncs = primFuncs.map(pf => arithmeticSimplify(pf));

  console.log(`\n  Lowered to ${optFuncs.length} PrimFunc(s):`);
  for (const pf of optFuncs) {
    const sig = pf.params.map(p => `${p.name}[${p.shape.join('×')}]`).join(', ');
    console.log(`    ${pf.name}(${sig})`);
  }

  if (optFuncs.length === 0) {
    throw new Error('No PrimFuncs produced. Check fusion pipeline.');
  }
  if (optFuncs.length !== nLayers) {
    throw new Error(`Expected ${nLayers} PrimFuncs (one per layer), got ${optFuncs.length}.`);
  }

  // Compute per-kernel FLOP counts for display
  const kernelFlops = optFuncs.map(pf => {
    const A = pf.params.find(p => p.name === 'A')!;
    const W = pf.params.find(p => p.name === 'W')!;
    // A: [BATCH, K],  W: [N, K]  →  2 × BATCH × K × N FLOPs (mul+add per element)
    const batchSize = A.shape[0];
    const K = A.shape[1];
    const N = W.shape[0];
    return 2 * batchSize * K * N;
  });
  const totalFlops = kernelFlops.reduce((a, b) => a + b, 0);
  console.log(`\n  Total FLOPs/forward: ${(totalFlops / 1e6).toFixed(2)} MFLOPs`);
  for (let i = 0; i < optFuncs.length; i++) {
    console.log(`    kernel ${i}  ${optFuncs[i].name.padEnd(26)}  ${(kernelFlops[i]/1e3).toFixed(0).padStart(6)} KFLOPs`);
  }

  // ── 3. WAT Codegen ───────────────────────────────────────────
  sep('3. WAT Codegen');
  const watMod = codegenWAT(optFuncs);
  const watPages = Math.ceil(watMod.totalBytes / 65536);
  console.log(`  Total bytes : ${watMod.totalBytes} (${watPages} WASM page)`);
  console.log(`  Exports     : [${watMod.exports.join(', ')}]`);

  // Rebuild buffer layout (no Map collisions)
  const funcOffsets = computeOffsets(optFuncs);
  console.log('\n  Buffer layout:');
  for (let fi = 0; fi < optFuncs.length; fi++) {
    console.log(`    [${optFuncs[fi].name}]`);
    for (const [name, byteOff] of funcOffsets[fi]) {
      const param = optFuncs[fi].params.find(p => p.name === name)!;
      const size  = param.shape.reduce((a, b) => a * b, 1) * 4;
      console.log(`      ${name.padEnd(6)}  byte ${String(byteOff).padStart(6)}–${String(byteOff + size - 1).padStart(6)}  (${size}B)`);
    }
  }

  // ── 4. Assemble WAT → WASM ───────────────────────────────────
  sep('4. Assemble WAT → WASM');
  const wabt    = await WabtModule();
  const wasmSrc = wabt.parseWat('kernels.wat', watMod.text);
  const { buffer: wasmBin } = wasmSrc.toBinary({});
  wasmSrc.destroy();
  console.log(`  Binary size : ${wasmBin.byteLength} bytes`);

  // ── 5. Instantiate WASM ──────────────────────────────────────
  sep('5. Instantiate WASM');
  const { instance } = await WebAssembly.instantiate(wasmBin);
  const wasmMem = instance.exports.memory as WebAssembly.Memory;
  const memF32  = new Float32Array(wasmMem.buffer);
  console.log(`  Memory      : ${wasmMem.buffer.byteLength / 1024}KB`);

  const wasmFns = optFuncs.map(pf => instance.exports[pf.name] as CallableFunction);
  for (let i = 0; i < optFuncs.length; i++) {
    console.log(`  Kernel ${i}    : ${optFuncs[i].name}`);
  }

  // ── 6. Marshal input + weights into WASM memory ──────────────
  sep('6. Marshal buffers → WASM linear memory');

  // Write input at kernel-0 A slot
  memF32.set(inputArr, funcOffsets[0].get('A')! / 4);
  // Write weights + biases for every kernel (slots are pre-allocated by codegenWAT)
  let writtenBytes = inputArr.length * 4;
  for (let i = 0; i < optFuncs.length; i++) {
    memF32.set(kernelWeights[i], funcOffsets[i].get('W')! / 4);
    memF32.set(kernelBiases[i],  funcOffsets[i].get('B')! / 4);
    writtenBytes += (kernelWeights[i].length + kernelBiases[i].length) * 4;
  }
  console.log(`  Written ${writtenBytes} bytes`);

  // ── 7. Execute WASM kernels (generic N-kernel pipeline) ───────
  sep('7. Execute WASM kernels');

  // Zero-copy activation pipe:
  //   kernel[0] reads from its own A slot (the input we wrote above)
  //   kernel[k>0] reads from kernel[k-1].Out offset (no buffer copy needed)
  let wasmPrevOutOff = funcOffsets[0].get('A')!;
  for (let i = 0; i < optFuncs.length; i++) {
    const off   = funcOffsets[i];
    const aOff  = (i === 0) ? off.get('A')! : wasmPrevOutOff;
    const outOff = off.get('Out')!;
    wasmFns[i](aOff, off.get('W')!, off.get('B')!, outOff);
    wasmPrevOutOff = outOff;
    console.log(`  Kernel ${i} done  (out @ byte ${outOff})`);
  }

  // Read WASM final output
  const finalOff   = wasmPrevOutOff;
  const outElems   = BATCH * OUT;
  const wasmOutput = new Float32Array(outElems);
  for (let i = 0; i < outElems; i++) wasmOutput[i] = memF32[finalOff / 4 + i];

  // ── 8. JS reference (generic N-kernel forward) ───────────────
  sep('8. JS reference forward pass');

  const jsKernels = optFuncs.map(pf => compile(pf));
  // Allocate intermediate activation buffers for every kernel output
  const jsActBufs = optFuncs.map(pf => {
    const outParam = pf.params.find(p => p.name === 'Out')!;
    return new Float32Array(outParam.shape.reduce((a, b) => a * b, 1));
  });

  let jsActivation: Float32Array = inputArr;
  for (let i = 0; i < optFuncs.length; i++) {
    jsKernels[i](jsActivation, kernelWeights[i], kernelBiases[i], jsActBufs[i]);
    jsActivation = jsActBufs[i];
  }
  const jsOutput = jsActBufs[jsActBufs.length - 1];
  console.log(`  Done (${optFuncs.length} kernels)`);

  // ── 9. Verify ────────────────────────────────────────────────
  sep('9. Verify WASM vs JS');

  let maxDelta = 0;
  let maxIdx   = 0;
  for (let i = 0; i < outElems; i++) {
    const d = Math.abs(wasmOutput[i] - jsOutput[i]);
    if (d > maxDelta) { maxDelta = d; maxIdx = i; }
  }

  const THRESH = 1e-4;
  const pass   = maxDelta < THRESH;

  // Print first few values side-by-side
  console.log(`  ${'idx'.padStart(4)}  ${'WASM'.padStart(12)}  ${'JS'.padStart(12)}  ${'|diff|'.padStart(12)}`);
  for (let i = 0; i < Math.min(outElems, 8); i++) {
    const d = Math.abs(wasmOutput[i] - jsOutput[i]);
    console.log(`  ${String(i).padStart(4)}  ${wasmOutput[i].toFixed(6).padStart(12)}  ${jsOutput[i].toFixed(6).padStart(12)}  ${d.toExponential(2).padStart(12)}`);
  }

  console.log(`\n  max|WASM − JS| = ${maxDelta.toExponential(3)}  (at index ${maxIdx})`);
  console.log(`  threshold      = ${THRESH.toExponential(0)}`);
  console.log(`  ${pass ? '✓ PASS' : '✗ FAIL'}`);

  // ── 10. Benchmark ────────────────────────────────────────────
  sep('10. Benchmark');

  // Helper: run one full JS forward pass
  function runJS() {
    let act: Float32Array = inputArr;
    for (let i = 0; i < optFuncs.length; i++) {
      jsKernels[i](act, kernelWeights[i], kernelBiases[i], jsActBufs[i]);
      act = jsActBufs[i];
    }
  }

  // Helper: run one full WASM forward pass
  function runWASM() {
    let prevOut = funcOffsets[0].get('A')!;
    for (let i = 0; i < optFuncs.length; i++) {
      const off    = funcOffsets[i];
      const aOff   = (i === 0) ? off.get('A')! : prevOut;
      const outOff = off.get('Out')!;
      wasmFns[i](aOff, off.get('W')!, off.get('B')!, outOff);
      prevOut = outOff;
    }
  }

  // JS warmup (V8 TurboFan needs ~300 calls for large unrolled functions)
  for (let i = 0; i < WARMUP; i++) runJS();
  const jsStart = performance.now();
  for (let i = 0; i < REPS; i++) runJS();
  const jsMs = (performance.now() - jsStart) / REPS;

  // WASM warmup
  for (let i = 0; i < WARMUP; i++) runWASM();
  const wasmStart = performance.now();
  for (let i = 0; i < REPS; i++) runWASM();
  const wasmMs = (performance.now() - wasmStart) / REPS;

  const speedup    = jsMs / wasmMs;
  const jsGflops   = (totalFlops / 1e9) / (jsMs   / 1e3);
  const wasmGflops = (totalFlops / 1e9) / (wasmMs / 1e3);

  console.log(`  ${REPS} reps, ${WARMUP} warmup  |  ${(totalFlops / 1e6).toFixed(1)} MFLOPs/call`);
  console.log(`  JS   ${optFuncs.length}-kernel forward : ${jsMs.toFixed(4)} ms/call  (${jsGflops.toFixed(2)} GFLOP/s)`);
  console.log(`  WASM ${optFuncs.length}-kernel forward : ${wasmMs.toFixed(4)} ms/call  (${wasmGflops.toFixed(2)} GFLOP/s)`);
  console.log(`  Speedup               : ${speedup.toFixed(2)}×  (${speedup >= 1 ? 'WASM faster' : 'JS faster'})`);

  // ── Final summary ────────────────────────────────────────────
  console.log('\n' + '═'.repeat(60));
  console.log(`  Result: ${pass ? '✓ PASS' : '✗ FAIL'}  |  max delta = ${maxDelta.toExponential(3)}`);
  console.log('═'.repeat(60));

  if (!pass) process.exit(1);
}

main().catch(err => {
  console.error('\n[FATAL]', err);
  process.exit(1);
});
