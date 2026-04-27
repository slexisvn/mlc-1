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

function estimateWorkingSet(func: PrimFunc): number {
  // The working set is the amount of data accessed by the innermost
  // tile of loops. For tiled loops, this is tile_i * tile_j * sizeof(float).
  const loops = func.getLoops();
  if (loops.length === 0) return 0;

  // Find innermost spatial loop extents
  let workingElements = 1;
  for (const l of loops) {
    if (l.loopVar.kind === 'spatial') {
      workingElements *= l.forNode.extent;
    }
  }

  // Each element is 4 bytes (float32)
  // We need to account for all buffers accessed in inner loops
  return workingElements * 4 * func.params.length;
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
  arithmeticIntensity: number;  // FLOP / byte
  workingSetBytes: number;
  cacheFitness: {
    fitsL1: boolean;
    fitsL2: boolean;
    fitsL3: boolean;
  };
  boundedness: 'compute-bound' | 'memory-bound';
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

  // ─── Arithmetic intensity ───
  const arithmeticIntensity = totalTrafficBytes > 0
    ? flopCount / totalTrafficBytes
    : 0;

  // ─── Working set ───
  const workingSetBytes = estimateWorkingSet(func);

  // ─── Cache fitness ───
  const cacheFitness = {
    fitsL1: workingSetBytes <= CACHE_SIZES.L1,
    fitsL2: workingSetBytes <= CACHE_SIZES.L2,
    fitsL3: workingSetBytes <= CACHE_SIZES.L3,
  };

  // ─── Boundedness ───
  // Ridge point: where compute roof meets memory roof
  // For typical CPUs: ~10 GFLOPS peak, ~20 GB/s bandwidth
  // Ridge point = 10 / 20 = 0.5 FLOP/byte
  const ridgePoint = 0.5;
  const boundedness: 'compute-bound' | 'memory-bound' =
    arithmeticIntensity >= ridgePoint ? 'compute-bound' : 'memory-bound';

  return {
    funcName: func.name,
    buffers,
    totalBytes,
    flopCount,
    memoryTraffic: { totalReads, totalWrites, totalBytes: totalTrafficBytes },
    arithmeticIntensity,
    workingSetBytes,
    cacheFitness,
    boundedness,
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
  lines.push(`  Memory traffic: ${formatBytes(plan.memoryTraffic.totalBytes)} (${plan.memoryTraffic.totalReads.toLocaleString()} reads, ${plan.memoryTraffic.totalWrites.toLocaleString()} writes)`);
  lines.push(`  Arithmetic intensity: ${plan.arithmeticIntensity.toFixed(2)} FLOP/byte`);
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
