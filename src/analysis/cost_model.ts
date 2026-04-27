// ═══════════════════════════════════════════════════════════════
//  D6: Analytical Cost Model
//
//  Why this matters (MLC concept):
//    A cost model predicts the performance of a kernel WITHOUT
//    actually running it. This is used by the auto-tuner to
//    rank schedule candidates without expensive benchmarking.
//
//    Real compilers (TVM, Halide, Triton) use learned or analytical
//    cost models to guide search. This is an analytical model that
//    reasons from first principles:
//
//    1. Count FLOPs — how many arithmetic operations?
//    2. Count bytes transferred — how much data moves?
//    3. Compute Arithmetic Intensity (AI) = FLOPs / bytes
//    4. Apply Roofline: predicted_perf = min(peak_compute, AI × peak_bandwidth)
//    5. Estimate utilization = predicted / peak_compute
//
//    The key insight is that AI determines the bottleneck:
//      AI < ridge_point  → memory-bound (bandwidth-limited)
//      AI > ridge_point  → compute-bound (FLOP-limited)
//
//    Schedule transforms affect AI:
//      Tiling: increases data reuse → higher AI
//      Vectorization: amortizes loop overhead → better compute util
//      Cache-read: reduces DRAM accesses → effectively higher AI
//
//  Loop analysis:
//    We walk the loop nest to count:
//      - iterations (extent product of all loops)
//      - FLOPs (count of BinOpExpr per iteration)
//      - memory reads/writes (BufferLoad/BufferStore per iteration)
// ═══════════════════════════════════════════════════════════════

import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, AllocNode,
  BufferLoadExpr, BinOpExpr, ConstExpr, VarRefExpr, MaxExpr, MinExpr,
  type Stmt, type ValueExpr
} from '../ir/low_level.js';
import type { HardwareProfile } from './op_profiler.js';
import { ScalarStoreNode, ScalarLoadExpr } from '../transform/storage_rewrite.js';
import { countKernelFlops } from './memory_planner.js';

// ─── Cost prediction result ───────────────────────────────────

export interface CostPrediction {
  funcName: string;
  /** Total floating-point operations */
  flops: number;
  /** Total bytes read + written (float32 → 4 bytes each) */
  bytes: number;
  /** Arithmetic intensity = flops / bytes */
  arithmeticIntensity: number;
  /** Predicted GFLOPS based on roofline */
  predictedGFLOPS: number;
  /** Whether this kernel is memory-bound or compute-bound */
  bottleneck: 'memory' | 'compute';
  /** Predicted utilization of peak compute */
  utilizationPct: number;
}

// ─── Static analysis helpers ──────────────────────────────────

/** Count total loop iterations from a nested ForNode (product of all extents) */
function countIterations(stmt: Stmt): number {
  if (stmt instanceof ForNode) {
    return stmt.extent * countIterations(stmt.body);
  }
  if (stmt instanceof SeqNode) {
    return stmt.stmts.reduce((sum, s) => sum + countIterations(s), 0);
  }
  if (stmt instanceof AllocNode) {
    return countIterations(stmt.body);
  }
  // BufferStoreNode — leaf, 1 iteration (the write itself)
  return 1;
}

/** Count FLOPs in a ValueExpr (each BinOp = 1 FLOP) */
function countFlopsInExpr(expr: ValueExpr): number {
  if (expr instanceof BinOpExpr) {
    return 1 + countFlopsInExpr(expr.left) + countFlopsInExpr(expr.right);
  }
  if (expr instanceof MaxExpr || expr instanceof MinExpr) {
    // max/min typically cost 1 comparison
    return 1 + countFlopsInExpr(expr.left) + countFlopsInExpr(expr.right);
  }
  // Load, Const, VarRef — no arithmetic
  return 0;
}

/** Count BufferLoad accesses in a ValueExpr.
 * ScalarLoadExpr (promoted register var) is NOT a memory access — register reads
 * are free from the roofline perspective, so they count as 0 loads. */
function countLoadsInExpr(expr: ValueExpr): number {
  if (expr instanceof BufferLoadExpr) {
    return 1;
  }
  if ((expr as any) instanceof ScalarLoadExpr) {
    return 0; // register read, not a DRAM/cache access
  }
  if (expr instanceof BinOpExpr) {
    return countLoadsInExpr(expr.left) + countLoadsInExpr(expr.right);
  }
  if (expr instanceof MaxExpr || expr instanceof MinExpr) {
    return countLoadsInExpr(expr.left) + countLoadsInExpr(expr.right);
  }
  return 0;
}

interface StmtCounts {
  flopsPerIter: number;
  loadsPerIter: number;
  storesPerIter: number;
}

/** Analyze flops and memory accesses inside the innermost loop body */
function analyzeStmtCounts(stmt: Stmt): StmtCounts {
  if (stmt instanceof ForNode) {
    return analyzeStmtCounts(stmt.body);
  }
  if (stmt instanceof SeqNode) {
    const totals: StmtCounts = { flopsPerIter: 0, loadsPerIter: 0, storesPerIter: 0 };
    for (const s of stmt.stmts) {
      const c = analyzeStmtCounts(s);
      totals.flopsPerIter += c.flopsPerIter;
      totals.loadsPerIter += c.loadsPerIter;
      totals.storesPerIter += c.storesPerIter;
    }
    return totals;
  }
  if (stmt instanceof AllocNode) {
    return analyzeStmtCounts(stmt.body);
  }
  if (stmt instanceof BufferStoreNode) {
    // 1 store + flops/loads inside the value expression
    return {
      flopsPerIter: countFlopsInExpr(stmt.value),
      loadsPerIter: countLoadsInExpr(stmt.value),
      storesPerIter: 1,
    };
  }
  if ((stmt as any) instanceof ScalarStoreNode) {
    // Scalar-promoted register assignment (e.g. acc = acc + A*W after storageRewrite).
    // Counts FLOPs and DRAM loads from the value expression.
    // storesPerIter = 0 because writing to a register is not a memory store.
    const s = stmt as unknown as ScalarStoreNode;
    return {
      flopsPerIter: countFlopsInExpr(s.value as ValueExpr),
      loadsPerIter: countLoadsInExpr(s.value as ValueExpr),
      storesPerIter: 0,
    };
  }
  return { flopsPerIter: 0, loadsPerIter: 0, storesPerIter: 0 };
}

/** Get total iteration count for the outermost loop levels only */
function getLoopIterations(stmt: Stmt): number {
  if (stmt instanceof ForNode) {
    return stmt.extent * getLoopIterations(stmt.body);
  }
  if (stmt instanceof SeqNode) {
    return stmt.stmts.reduce((s, st) => s + getLoopIterations(st), 0);
  }
  if (stmt instanceof AllocNode) {
    return getLoopIterations(stmt.body);
  }
  // leaf — just 1
  return 1;
}

// ─── Main cost model API ──────────────────────────────────────

/**
 * Analytically predict kernel performance using static IR analysis + Roofline.
 *
 * How it works:
 *   1. Walk the PrimFunc loop nest and count:
 *      - Total loop iterations (product of all For extents)
 *      - FLOPs per innermost iteration (BinOpExpr count in BufferStore.value)
 *      - Memory accesses per iteration (BufferLoad + BufferStore count)
 *   2. Total FLOPs = iterations × flopsPerIter
 *   3. Total bytes = (loads + stores) × sizeof(float32) = × 4
 *   4. AI = FLOPs / bytes
 *   5. Roofline: predicted = min(peak_compute, AI × peak_bandwidth)
 *   6. Utilization = predicted / peak_compute
 */
export function predictCost(func: PrimFunc, hw: HardwareProfile): CostPrediction {
  // Use the same recursive FLOP counter as Phase 9 memory_planner so both
  // sections report identical FLOPs.  The old approach (getLoopIterations ×
  // flopsPerIter) overcounted for mixed-depth SeqNodes (e.g. loop nests that
  // contain both an init statement and the accumulation loop as siblings).
  const totalFlops = countKernelFlops(func);

  // Bytes: keep the existing loads/stores estimator for the AI calculation
  const iterations = getLoopIterations(func.body);
  const innerCounts = analyzeStmtCounts(func.body);
  const totalAccesses = iterations * (innerCounts.loadsPerIter + innerCounts.storesPerIter);
  const totalBytes = totalAccesses * 4; // float32 = 4 bytes

  const ai = totalBytes > 0 ? totalFlops / totalBytes : 0;

  // Roofline model:
  //   memory ceiling = AI × bandwidth (GB/s → GFLOPS equivalent)
  //   compute ceiling = peak GFLOPS
  //   attainable GFLOPS = min(memory_ceiling, compute_ceiling)
  const memoryCeiling = ai * hw.peakBandwidthGBs;
  const predictedGFLOPS = Math.min(memoryCeiling, hw.peakGFLOPS);
  const bottleneck: 'memory' | 'compute' = ai < hw.ridgePoint ? 'memory' : 'compute';
  const utilizationPct = (predictedGFLOPS / hw.peakGFLOPS) * 100;

  return {
    funcName: func.name,
    flops: totalFlops,
    bytes: totalBytes,
    arithmeticIntensity: ai,
    predictedGFLOPS,
    bottleneck,
    utilizationPct,
  };
}

/**
 * Compare analytical prediction vs measured performance.
 * Returns a formatted report table.
 *
 * This validates the cost model — if predicted ≈ measured, the model
 * is useful for ranking schedule candidates (auto-tuner use case).
 * If predicted >> measured, the kernel has overhead the model doesn't see
 * (e.g. cache misses, loop overhead, JIT warm-up). This gap is expected
 * for small kernels in JavaScript but narrows for large matmul workloads.
 */
export function comparePredictedVsMeasured(
  predictions: CostPrediction[],
  measured: Map<string, number>, // funcName → measured ms/call
  hw: HardwareProfile
): string {
  const lines: string[] = [];
  const W = 26;

  lines.push(`  Analytical Cost Model vs Measured — ${hw.name}`);
  lines.push(`  ${'─'.repeat(90)}`);
  lines.push(
    `  ${'Kernel'.padEnd(W)} ${'FLOPs'.padStart(10)} ${'Bytes'.padStart(10)} ` +
    `${'AI(F/B)'.padStart(9)} ${'Pred.GFLOPS'.padStart(12)} ${'Meas.ms'.padStart(9)} ` +
    `${'Bottleneck'.padStart(10)} ${'Util%'.padStart(7)}`
  );
  lines.push(`  ${'─'.repeat(90)}`);

  for (const p of predictions) {
    const measMs = measured.get(p.funcName) ?? -1;
    const measGFLOPS = measMs > 0 ? (p.flops / 1e9) / (measMs / 1000) : -1;
    const acc = measGFLOPS > 0
      ? `${Math.min(measGFLOPS / p.predictedGFLOPS * 100, 999).toFixed(0)}%`
      : 'n/a';

    lines.push(
      `  ${p.funcName.padEnd(W)} ${p.flops.toExponential(2).padStart(10)} ` +
      `${p.bytes.toExponential(2).padStart(10)} ${p.arithmeticIntensity.toFixed(3).padStart(9)} ` +
      `${p.predictedGFLOPS.toFixed(4).padStart(12)} ${measMs > 0 ? measMs.toFixed(4) : 'n/a'.padStart(7)} ` +
      `${p.bottleneck.padStart(10)} ${p.utilizationPct.toFixed(1).padStart(5)}% | acc:${acc}`
    );
  }

  lines.push(`  ${'─'.repeat(90)}`);
  lines.push(`  Note: pred. assumes 100% DRAM traffic — actual AI is higher when data is cached.`);
  lines.push(`        acc = measured_GFLOPS / predicted_GFLOPS × 100%.`);
  lines.push(`        acc < 100%: JIT/cache-miss overhead not captured by model — normal for small kernels.`);
  lines.push(`        acc > 100%: buffers fit in L1/L2 → actual bandwidth > model's DRAM assumption.`);
  lines.push(`        acc ≈ 100% at large N (buffers exceed cache) confirms model accuracy.`);

  return lines.join('\n');
}
