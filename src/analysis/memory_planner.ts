// ═══════════════════════════════════════════════════════════════
//  Memory Planning & Analysis Pass (read-only)
//
//  Analyzes each PrimFunc to compute:
//  1. Total memory footprint (bytes per buffer)
//  2. FLOP count (number of arithmetic operations)
//  3. Memory traffic (bytes read/written, naive estimate)
//  4. Arithmetic Intensity (FLOP/byte)
//     → determines if kernel is compute-bound or memory-bound
//  5. Working set size with current tile config
//  6. Cache fitness estimate (does working set fit L1/L2?)
//
//  This is the foundation for "roofline model" analysis that
//  real MLC frameworks use to guide schedule search.
// ═══════════════════════════════════════════════════════════════

import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, AllocNode,
  BufferLoadExpr, BinOpExpr, ConstExpr, MaxExpr, MinExpr,
  CallExprTIR,
  type Stmt, type ValueExpr
} from '../ir/low_level.js';

// ─── Cache size estimates for common architectures ───
const CACHE_SIZES = {
  L1: 32 * 1024,       // 32 KB (typical per-core L1d)
  L2: 256 * 1024,      // 256 KB (typical per-core L2)
  L3: 8 * 1024 * 1024, // 8 MB (typical shared L3)
};

// ═══════════════════════════════════════
//  FLOP Counting
// ═══════════════════════════════════════

interface LoopContext {
  iterations: number;
}

function countFLOPsInValue(val: ValueExpr): number {
  if (val.kind === 'const' || val.kind === 'varref' || val.kind === 'load') return 0;
  if ((val as any).kind === 'scalar_load') return 0;

  if (val.kind === 'binop') {
    const v = val as BinOpExpr;
    return 1 + countFLOPsInValue(v.left) + countFLOPsInValue(v.right);
  }

  if (val.kind === 'max' || val.kind === 'min') {
    const v = val as (MaxExpr | MinExpr);
    return 1 + countFLOPsInValue(v.left) + countFLOPsInValue(v.right);
  }

  if (val.kind === 'call') {
    const v = val as CallExprTIR;
    // Transcendental functions count as ~10 FLOPs
    const callCost = v.funcName.startsWith('Math.') ? 10 : 1;
    return callCost + v.args.reduce((sum, a) => sum + countFLOPsInValue(a), 0);
  }

  return 0;
}

function countFLOPsInStmt(stmt: Stmt): number {
  if (stmt instanceof ForNode) {
    return stmt.extent * countFLOPsInStmt(stmt.body);
  }
  if (stmt instanceof SeqNode) {
    return stmt.stmts.reduce((sum, s) => sum + countFLOPsInStmt(s), 0);
  }
  if (stmt instanceof BufferStoreNode) {
    return countFLOPsInValue(stmt.value);
  }
  if (stmt instanceof AllocNode) {
    return countFLOPsInStmt(stmt.body);
  }
  // Handle ScalarStoreNode (from storage rewrite)
  if ((stmt as any).nodeType === 'scalar_store') {
    return countFLOPsInValue((stmt as any).value);
  }
  if ((stmt as any).nodeType === 'scalar_decl') {
    return countFLOPsInStmt((stmt as any).body);
  }
  return 0;
}

// ═══════════════════════════════════════
//  Memory Traffic Analysis
// ═══════════════════════════════════════

function countLoadsInValue(val: ValueExpr, loadCounts: Map<string, number>): void {
  if (val.kind === 'load') {
    const v = val as BufferLoadExpr;
    loadCounts.set(v.buffer.name, (loadCounts.get(v.buffer.name) || 0) + 1);
  } else if (val.kind === 'binop') {
    const v = val as BinOpExpr;
    countLoadsInValue(v.left, loadCounts);
    countLoadsInValue(v.right, loadCounts);
  } else if (val.kind === 'max' || val.kind === 'min') {
    const v = val as (MaxExpr | MinExpr);
    countLoadsInValue(v.left, loadCounts);
    countLoadsInValue(v.right, loadCounts);
  } else if (val.kind === 'call') {
    const v = val as CallExprTIR;
    v.args.forEach(a => countLoadsInValue(a, loadCounts));
  }
}

function analyzeMemoryTraffic(func: PrimFunc): { reads: Map<string, number>; writes: Map<string, number> } {
  const reads = new Map<string, number>();
  const writes = new Map<string, number>();

  function visitStmt(stmt: Stmt, multiplier: number): void {
    if (stmt instanceof ForNode) {
      visitStmt(stmt.body, multiplier * stmt.extent);
    } else if (stmt instanceof SeqNode) {
      stmt.stmts.forEach(s => visitStmt(s, multiplier));
    } else if (stmt instanceof BufferStoreNode) {
      // Count writes
      writes.set(stmt.buffer.name, (writes.get(stmt.buffer.name) || 0) + multiplier);
      // Count reads in value
      const loadCounts = new Map<string, number>();
      countLoadsInValue(stmt.value, loadCounts);
      for (const [name, count] of loadCounts) {
        reads.set(name, (reads.get(name) || 0) + count * multiplier);
      }
    } else if (stmt instanceof AllocNode) {
      visitStmt(stmt.body, multiplier);
    }
    // Handle scalar store
    if ((stmt as any).nodeType === 'scalar_store') {
      const loadCounts = new Map<string, number>();
      countLoadsInValue((stmt as any).value, loadCounts);
      for (const [name, count] of loadCounts) {
        reads.set(name, (reads.get(name) || 0) + count * multiplier);
      }
    }
    if ((stmt as any).nodeType === 'scalar_decl') {
      visitStmt((stmt as any).body, multiplier);
    }
  }

  visitStmt(func.body, 1);
  return { reads, writes };
}

// ═══════════════════════════════════════
//  Working Set Size (with tiling)
// ═══════════════════════════════════════

function estimateWorkingSet(func: PrimFunc, schedInfo: ScheduleInfo): number {
  // Working set = data that must be live in cache simultaneously at peak.
  //
  // IMPORTANT: getLoops() traverses ALL ForNodes including the packing stage
  // loops injected by cacheRead (_jl_cr, _kl_cr). Multiplying all spatial
  // extents together would therefore over-count by tileJ × K.  Instead, derive
  // the working set from what is actually resident:
  //
  //  • cache_read applied  → W_local (explicit local alloc) + A row-slice + Out tile
  //  • tiled (no W_local)  → W tile estimate + A row-slice + Out tile
  //  • no tiling           → product of *non-packing* spatial loop extents

  if (schedInfo.hasCacheRead && schedInfo.localBufBytes > 0) {
    // W_local[tileJ, K] is exactly the tile in cache
    const wParam = func.params.find(p => p.name === 'W');
    const K = wParam?.shape[1] ?? 1;
    const aSliceBytes = 1 * K * 4;          // one input row: A[i, 0..K)
    const outTileBytes = schedInfo.tileJ * 4; // partial output tile
    return schedInfo.localBufBytes + aSliceBytes + outTileBytes;
  }

  if (schedInfo.tiled) {
    const wParam = func.params.find(p => p.name === 'W');
    const K = wParam?.shape[1] ?? 1;
    const wTileBytes  = schedInfo.tileJ * K * 4;
    const aSliceBytes = K * 4;
    const outTileBytes = schedInfo.tileJ * 4;
    return wTileBytes + aSliceBytes + outTileBytes;
  }

  // No tiling: multiply only non-packing spatial loop extents.
  // Packing helper loops are named with a leading '_' (e.g. _jl_cr, _kl_cr).
  const loops = func.getLoops();
  if (loops.length === 0) return 0;
  let workingElements = 1;
  for (const l of loops) {
    if (l.loopVar.kind === 'spatial' && !l.loopVar.name.startsWith('_')) {
      workingElements *= l.forNode.extent;
    }
  }
  return workingElements * 4 * Math.max(1, func.params.length - 1);
}

// ═══════════════════════════════════════
//  Schedule Detection
//  Inspect loop names and allocations to infer which schedule
//  transforms have been applied, so AI can be computed correctly.
// ═══════════════════════════════════════

export interface ScheduleInfo {
  /** j has been split into j_outer × j_inner */
  tiled: boolean;
  /** tile size for j (the j_inner extent) */
  tileJ: number;
  /** a W_local or *_local buffer exists → cache_read applied */
  hasCacheRead: boolean;
  /** local buffer size in bytes */
  localBufBytes: number;
  /** W reuse factor due to tiling: the k-loop extent (each W row reused k times) */
  wReuseK: number;
  /** scalar promotion was applied (acc Float32Array[1] → let acc = 0) */
  hasScalarPromotion: boolean;
  /** j extent of the original function (used to suppress irrelevant hints) */
  jExtent: number;
  /** applied optimisation labels, used for recommendation text */
  applied: string[];
}

function detectScheduleInfo(func: PrimFunc): ScheduleInfo {
  const loops = func.getLoops();
  const loopNames = loops.map(l => l.loopVar.name);

  const tiled = loopNames.includes('j_outer') && loopNames.includes('j_inner');
  let tileJ = 1;
  if (tiled) {
    const jInnerLoop = loops.find(l => l.loopVar.name === 'j_inner');
    if (jInnerLoop) tileJ = jInnerLoop.forNode.extent;
  }

  // j extent: use j_outer*tileJ when tiled, otherwise direct j loop extent
  const jLoop = loops.find(l => l.loopVar.name === 'j');
  const jOuterLoop = loops.find(l => l.loopVar.name === 'j_outer');
  const jExtent = jLoop ? jLoop.forNode.extent
    : (jOuterLoop && tiled ? jOuterLoop.forNode.extent * tileJ : 1);

  // Detect cache_read: a local alloc whose name ends with '_local'
  const localAllocs = func.allocations.filter(a => a.name.endsWith('_local'));
  const hasCacheRead = localAllocs.length > 0;
  const localBufBytes = localAllocs.reduce(
    (sum, a) => sum + a.shape.reduce((s, d) => s * d, 1) * 4, 0
  );

  // k-loop reuse factor: how many times each loaded W element is reused.
  // For dense matmul: each W[j,k] is multiplied once per k iteration,
  // and the tile is loaded once for tileJ output elements → reuse = tileJ.
  // Without tiling each element is only loaded once per output row → reuse = 1.
  const wReuseK = tiled ? tileJ : 1;

  // Detect scalar promotion: storageRewrite removes acc from func.allocations
  // and rewrites BufferStore(acc)/BufferLoad(acc) → ScalarStoreNode/ScalarLoadExpr.
  // The signal is a ScalarStoreNode (or ScalarDeclNode) anywhere in the body.
  function hasScalarNode(stmt: Stmt): boolean {
    if ((stmt as any).nodeType === 'scalar_decl') return true;
    if ((stmt as any).nodeType === 'scalar_store') return true;
    if (stmt instanceof ForNode) return hasScalarNode(stmt.body);
    if (stmt instanceof SeqNode) return stmt.stmts.some(hasScalarNode);
    if (stmt instanceof AllocNode) return hasScalarNode(stmt.body);
    return false;
  }
  const hasScalarPromotion = hasScalarNode(func.body);

  const applied: string[] = [];
  if (tiled) applied.push(`tiling(j,${tileJ})`);
  if (hasCacheRead) applied.push('cache_read');
  if (hasScalarPromotion) applied.push('scalar_promotion');

  return { tiled, tileJ, hasCacheRead, localBufBytes, wReuseK, hasScalarPromotion, jExtent, applied };
}

// ─── Effective memory traffic accounting for cache reuse ───
// The naive traffic counter treats every array access as a DRAM round-trip.
// With tiling: within each j_outer tile, the W tile (tileJ × K floats) is
// loaded once and reused for tileJ output elements, so the effective W reads
// per tile are K (not tileJ × K). This raises AI by a factor of tileJ.
function effectiveTrafficBytes(
  rawReads: Map<string, number>,
  rawWrites: Map<string, number>,
  schedInfo: ScheduleInfo,
  func: PrimFunc,
): number {
  let totalEffectiveBytes = 0;

  for (const [name, count] of rawReads) {
    const param = func.params.find(p => p.name === name);
    const isLocalBuf = name.endsWith('_local');

    // Local buffers (acc, W_local) live in registers/L1 — not DRAM traffic
    if (isLocalBuf) continue;
    if (param?.scope === 'local') continue;

    // W buffer: with tiling, effective reads = totalW_elements (each loaded once per tile)
    // rather than tileJ × totalW_elements (once per output element)
    if (name === 'W' && schedInfo.tiled && schedInfo.wReuseK > 1) {
      const wParam = func.params.find(p => p.name === 'W');
      if (wParam) {
        // Effective: W is loaded exactly once (one DRAM pass), reused inside tile
        const wElements = wParam.shape.reduce((a, b) => a * b, 1);
        totalEffectiveBytes += wElements * 4;
        continue;
      }
    }

    totalEffectiveBytes += count * 4;
  }

  for (const [name, count] of rawWrites) {
    const isLocalBuf = name.endsWith('_local');
    if (isLocalBuf) continue;
    totalEffectiveBytes += count * 4;
  }

  return totalEffectiveBytes;
}

// ═══════════════════════════════════════
//  Main Analysis Entry Point
// ═══════════════════════════════════════

export interface BufferInfo {
  name: string;
  shape: number[];
  sizeBytes: number;
  scope: string;
  role: 'input' | 'output' | 'local';
}

export interface MemoryPlan {
  funcName: string;
  buffers: BufferInfo[];
  totalBytes: number;
  flopCount: number;
  memoryTraffic: { totalReads: number; totalWrites: number; totalBytes: number };
  effectiveTrafficBytes: number;
  arithmeticIntensity: number;  // FLOP / byte (raw, naive)
  effectiveAI: number;          // FLOP / byte (cache-aware, after tiling / cache_read)
  workingSetBytes: number;
  cacheFitness: {
    fitsL1: boolean;
    fitsL2: boolean;
    fitsL3: boolean;
  };
  boundedness: 'compute-bound' | 'memory-bound';
  scheduleInfo: ScheduleInfo;
}

export function analyzeMemory(func: PrimFunc): MemoryPlan {
  // ─── Buffer info ───
  const buffers: BufferInfo[] = [];
  const lastParam = func.params[func.params.length - 1];

  for (const p of func.params) {
    const sizeBytes = p.shape.reduce((a, b) => a * b, 1) * 4;
    buffers.push({
      name: p.name,
      shape: [...p.shape],
      sizeBytes,
      scope: p.scope,
      role: p === lastParam ? 'output' : 'input',
    });
  }
  for (const a of func.allocations) {
    const sizeBytes = a.shape.reduce((a2, b) => a2 * b, 1) * 4;
    buffers.push({
      name: a.name,
      shape: [...a.shape],
      sizeBytes,
      scope: a.scope,
      role: 'local',
    });
  }

  const totalBytes = buffers.reduce((sum, b) => sum + b.sizeBytes, 0);

  // ─── FLOP count ───
  const flopCount = countFLOPsInStmt(func.body);

  // ─── Memory traffic ───
  const { reads, writes } = analyzeMemoryTraffic(func);
  let totalReads = 0;
  for (const [, count] of reads) totalReads += count;
  let totalWrites = 0;
  for (const [, count] of writes) totalWrites += count;
  const totalTrafficBytes = (totalReads + totalWrites) * 4;

  // ─── Schedule detection & cache-aware effective traffic ───
  const scheduleInfo = detectScheduleInfo(func);
  const effTrafficBytes = effectiveTrafficBytes(reads, writes, scheduleInfo, func);

  // ─── Arithmetic intensity ───
  const arithmeticIntensity = totalTrafficBytes > 0
    ? flopCount / totalTrafficBytes
    : 0;
  const effectiveAI = effTrafficBytes > 0
    ? flopCount / effTrafficBytes
    : arithmeticIntensity;

  // ─── Working set ───
  const workingSetBytes = estimateWorkingSet(func, scheduleInfo);

  // ─── Cache fitness ───
  const cacheFitness = {
    fitsL1: workingSetBytes <= CACHE_SIZES.L1,
    fitsL2: workingSetBytes <= CACHE_SIZES.L2,
    fitsL3: workingSetBytes <= CACHE_SIZES.L3,
  };

  // ─── Boundedness — use effectiveAI for accuracy ───
  const ridgePoint = 0.5;
  const boundedness: 'compute-bound' | 'memory-bound' =
    effectiveAI >= ridgePoint ? 'compute-bound' : 'memory-bound';

  return {
    funcName: func.name,
    buffers,
    totalBytes,
    flopCount,
    memoryTraffic: { totalReads, totalWrites, totalBytes: totalTrafficBytes },
    effectiveTrafficBytes: effTrafficBytes,
    arithmeticIntensity,
    effectiveAI,
    workingSetBytes,
    cacheFitness,
    boundedness,
    scheduleInfo,
  };
}

// ═══════════════════════════════════════
//  Pretty-Print Memory Plan
// ═══════════════════════════════════════

export function printMemoryPlan(plan: MemoryPlan): string {
  const lines: string[] = [];
  lines.push(`Memory Analysis for ${plan.funcName}:`);
  lines.push(`  Total buffer size: ${formatBytes(plan.totalBytes)}`);

  for (const b of plan.buffers) {
    const marker = b.role === 'output' ? '(output)' : b.role === 'local' ? '(local)' : '(input)';
    lines.push(`    ${b.name}[${b.shape.join(',')}]: ${formatBytes(b.sizeBytes)} ${marker}`);
  }

  lines.push('');
  lines.push(`  FLOP count: ${plan.flopCount.toLocaleString()}`);
  lines.push(`  Memory traffic (naive):     ${formatBytes(plan.memoryTraffic.totalBytes)} (${plan.memoryTraffic.totalReads.toLocaleString()} reads, ${plan.memoryTraffic.totalWrites.toLocaleString()} writes)`);
  if (plan.scheduleInfo.applied.length > 0) {
    lines.push(`  Memory traffic (effective):  ${formatBytes(plan.effectiveTrafficBytes)}  [after ${plan.scheduleInfo.applied.join(', ')}]`);
    lines.push(`  Arithmetic intensity (naive):     ${plan.arithmeticIntensity.toFixed(2)} FLOP/byte`);
    lines.push(`  Arithmetic intensity (effective): ${plan.effectiveAI.toFixed(2)} FLOP/byte  ← cache-aware`);
    if (plan.scheduleInfo.hasScalarPromotion) {
      lines.push(`  Note: scalar_promotion eliminates Float32Array heap alloc → register variable.`);
      lines.push(`        Reduces per-call overhead; speedup exceeds what FLOP/byte AI predicts.`);
    }
  } else {
    lines.push(`  Arithmetic intensity: ${plan.arithmeticIntensity.toFixed(2)} FLOP/byte`);
  }
  lines.push(`  Working set: ${formatBytes(plan.workingSetBytes)}`);

  const cacheStr = plan.cacheFitness.fitsL1 ? 'L1 ✓' :
    plan.cacheFitness.fitsL2 ? 'L2 ✓ (L1 ✗)' :
    plan.cacheFitness.fitsL3 ? 'L3 ✓ (L1 ✗, L2 ✗)' : 'L3 ✗ (cache thrashing!)';
  lines.push(`  Cache fitness: ${cacheStr}`);

  const bar = plan.boundedness === 'compute-bound' ? '🔥 COMPUTE-BOUND' : '🧊 MEMORY-BOUND';
  lines.push(`  Bottleneck: ${bar}`);

  return lines.join('\n');
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
