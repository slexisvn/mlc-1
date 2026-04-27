// ═══════════════════════════════════════════════════════════════
//  Shape Inference Pass — propagates tensor shapes through the
//  high-level IR so every CallExpr node carries its output shape.
//
//  Why this matters (MLC concept):
//    Compilers must know output shapes before they can allocate
//    buffers, lower to loop nests, or apply schedule transforms.
//    TVM's Relay uses a type inference pass for exactly this.
//    Here we implement a lightweight post-order traversal that
//    applies per-op shape rules to annotate the whole IR.
//
//  After this pass, every CallExpr has attrs.outputShape filled.
// ═══════════════════════════════════════════════════════════════

import {
  Expr, VarExpr, ConstantExpr, CallExpr, LetExpr,
  IRModule, IRFunction
} from '../ir/high_level.js';

// ─── Result type returned by the pass ───

export interface ShapeInferResult {
  /** Total number of CallExpr nodes visited */
  totalOps: number;
  /** Number of nodes whose shape was successfully inferred */
  inferred: number;
  /** Formatted table for printing */
  table: string;
}

// ─── Per-op shape inference rules ───
// Each rule receives the inferred shapes of all input args and
// returns the output shape, or null if unsupported.

function inferOutputShape(
  opName: string,
  argShapes: number[][],
  attrs: Record<string, any>
): number[] | null {
  switch (opName) {
    // ── Dense: [M, K] × [N, K]^T → [M, N] ──
    case 'nn.dense': {
      const [M] = argShapes[0] ?? [];
      const [N] = argShapes[1] ?? [];
      if (M == null || N == null) return null;
      return [M, N];
    }

    // ── Bias add: keeps input shape ──
    case 'nn.bias_add': {
      return argShapes[0] ?? null;
    }

    // ── Element-wise activations: shape-preserving ──
    case 'nn.relu':
    case 'nn.sigmoid':
    case 'nn.tanh':
    case 'nn.leaky_relu':
    case 'nn.exp':
    case 'nn.log':
    case 'nn.neg':
      return argShapes[0] ?? null;

    // ── Softmax: shape-preserving ──
    case 'nn.softmax':
      return argShapes[0] ?? null;

    // ── Element-wise binary: broadcast shape ──
    case 'add':
    case 'subtract':
    case 'multiply': {
      const a = argShapes[0];
      const b = argShapes[1];
      if (!a || !b) return null;
      // Simple broadcast: take the longer shape; matching dims use max()
      const rank = Math.max(a.length, b.length);
      const result: number[] = [];
      for (let i = 0; i < rank; i++) {
        const da = a[a.length - rank + i] ?? 1;
        const db = b[b.length - rank + i] ?? 1;
        result.push(Math.max(da, db));
      }
      return result;
    }

    // ── Reductions: collapse all dims to scalar [1] ──
    case 'reduce_sum':
    case 'reduce_max':
    case 'reduce_mean':
      return [1];

    // ── Loss ops: scalar output ──
    case 'cross_entropy':
    case 'bce_with_logits':
    case 'mse':
      return [1];

    // ── Fused ops: same shape as nn.dense (M × N) ──
    case 'fused.dense_bias_relu':
    case 'fused.dense_bias_sigmoid':
    case 'fused.dense_bias_tanh':
    case 'fused.dense_bias': {
      const [M] = argShapes[0] ?? [];
      const [N] = argShapes[1] ?? [];
      if (M == null || N == null) return null;
      return [M, N];
    }

    case 'fused.softmax_ce':
    case 'fused.sigmoid_bce':
      return [1];

    // ── Gradient ops: same shape as first input ──
    case 'nn.dense_grad_data':
    case 'nn.dense_grad_weight':
    case 'nn.relu_grad':
    case 'nn.sigmoid_grad':
    case 'nn.tanh_grad':
    case 'nn.leaky_relu_grad':
    case 'nn.softmax_grad':
    case 'nn.bias_add_grad':
    case 'nn.cross_entropy_grad':
    case 'nn.bce_grad':
      return argShapes[0] ?? null;

    default:
      // Fall back to attrs.outputShape if lowering already set it
      if (Array.isArray(attrs.outputShape)) return attrs.outputShape as number[];
      return null;
  }
}

// ─── Get the inferred shape of any Expr ───

function getShape(expr: Expr, shapeMap: Map<Expr, number[]>): number[] | null {
  if (expr.kind === 'var') return expr.type.shape;
  if (expr.kind === 'constant') return expr.data.shape;
  return shapeMap.get(expr) ?? null;
}

// ─── Main inference traversal (post-order) ───
// Fills shapeMap with output shapes for every CallExpr.
// Also sets attrs.outputShape on each CallExpr so later passes
// (lowering, codegen) can read the inferred shape.

function inferShapes(
  expr: Expr,
  shapeMap: Map<Expr, number[]>,
  rows: { op: string; inputShapes: string; outputShape: string }[]
): void {
  if (expr.kind === 'let') {
    inferShapes(expr.value, shapeMap, rows);
    inferShapes(expr.body, shapeMap, rows);
    return;
  }
  if (expr.kind !== 'call') return;

  // Post-order: infer args first
  for (const arg of expr.args) {
    inferShapes(arg, shapeMap, rows);
  }

  const argShapes = expr.args.map(a => getShape(a, shapeMap) ?? []);
  const outputShape = inferOutputShape(expr.op.name, argShapes, expr.attrs);

  if (outputShape) {
    shapeMap.set(expr, outputShape);
    // Annotate the node so downstream passes can use it without re-inference
    expr.attrs.outputShape = outputShape;
  }

  rows.push({
    op: expr.op.name,
    inputShapes: argShapes.map(s => `[${s.join(',')}]`).join(' × '),
    outputShape: outputShape ? `[${outputShape.join(',')}]` : '?',
  });
}

// ─── Public entry point ───

export function inferModuleShapes(module: IRModule): ShapeInferResult {
  const shapeMap = new Map<Expr, number[]>();
  const rows: { op: string; inputShapes: string; outputShape: string }[] = [];

  for (const [, fn] of module.functions) {
    inferShapes(fn.body, shapeMap, rows);
  }

  // Build formatted table
  const COL_OP = 28;
  const COL_IN = 30;
  const COL_OUT = 14;
  const header =
    `  ${'Op'.padEnd(COL_OP)} ${'Input shapes'.padEnd(COL_IN)} ${'→ Output'.padEnd(COL_OUT)}`;
  const sep = '  ' + '─'.repeat(COL_OP + COL_IN + COL_OUT + 2);
  const tableLines = [header, sep];
  for (const r of rows) {
    tableLines.push(
      `  ${r.op.padEnd(COL_OP)} ${r.inputShapes.padEnd(COL_IN)} → ${r.outputShape}`
    );
  }
  const table = tableLines.join('\n');

  const inferred = rows.filter(r => r.outputShape !== '?').length;
  return { totalOps: rows.length, inferred, table };
}
