// ═══════════════════════════════════════════════════════════════
//  F9: Vectorization Pass
//
//  What this does (MLC concept):
//    Vectorization is a loop transformation that maps iterations
//    of a spatial loop to SIMD lanes, allowing N operations to
//    execute in parallel on vector hardware (SSE/AVX/NEON/WASM SIMD).
//
//    This pass does two things:
//    1. "Vectorize" transform — split the innermost spatial loop by
//       SIMD width W and annotate the inner loop as 'vectorize'
//    2. Codegen support — the JS codegen already has a 'vectorize'
//       annotation path; this pass makes it generate SIMD-width
//       unrolled blocks (W scalar ops per outer step)
//
//    In real compilers (TVM/Halide), 'vectorize' maps to CPU SIMD:
//      - x86: __m128 / __m256 intrinsics (SSE2/AVX)
//      - ARM: float32x4_t NEON intrinsics
//      - WASM: v128 SIMD instructions
//
//    In our JS target, we emit W-way unrolled scalar code.
//    This is semantically equivalent and shows the compiler's intent.
//    On V8 with TurboFan, this can trigger autovectorization.
//
//  Algorithm:
//    1. Walk the loop nest to find the innermost spatial ForNode
//    2. If extent >= SIMD_WIDTH (4), split: outer(extent/W) × inner(W)
//    3. Annotate inner loop as 'vectorize'
//    4. Rewrite body indices: loopVar → outer*W + inner
//
//  Example transformation for SIMD_WIDTH=4:
//    Before:
//      for j in range(64):                    // spatial
//        Out[i,j] = A[i,k] * W[j,k]
//
//    After:
//      for j_outer in range(16):             // non-vectorized outer
//        for j_inner in range(4) [vectorize]:  // 4 SIMD lanes
//          let j = j_outer*4 + j_inner
//          Out[i,j] = A[i,k] * W[j,k]
//
//    Codegen emits:
//      for (let j_outer = 0; j_outer < 16; j_outer++) {
//        // vectorize: 4 SIMD lanes
//        const j0 = j_outer * 4 + 0, j1 = j_outer * 4 + 1, ...
//        Out[i*64 + j0] = A[...] * W[j0*32 + k];
//        Out[i*64 + j1] = A[...] * W[j1*32 + k];
//        ...
//      }
// ═══════════════════════════════════════════════════════════════

import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, AllocNode,
  VarRefExpr, BinOpExpr, ConstExpr, BufferLoadExpr, MaxExpr, MinExpr,
  VarIndex, ConstIndex, BinOpIndex, LoopVar,
  type Stmt, type ValueExpr, type IndexExpr
} from '../ir/low_level.js';
import { ScalarStoreNode, ScalarDeclNode } from './storage_rewrite.js';

export const SIMD_WIDTH = 4;

export interface VectorizationResult {
  func: PrimFunc;
  /** Number of loops that were vectorized */
  vectorizedCount: number;
  /** Names of loop variables that were split */
  splitLoops: string[];
}

// ─── Main API ─────────────────────────────────────────────────

/**
 * Apply vectorization to a PrimFunc:
 * Find innermost non-vectorized spatial loops with extent ≥ SIMD_WIDTH
 * and split them into outer × SIMD_WIDTH with inner annotated 'vectorize'.
 */
export function vectorize(func: PrimFunc): VectorizationResult {
  const splitLoops: string[] = [];
  const newBody = vectorizeStmt(func.body, splitLoops);
  const newFunc = new PrimFunc(func.name, func.params, newBody, func.allocations);
  return { func: newFunc, vectorizedCount: splitLoops.length, splitLoops };
}

// ─── Loop nest traversal ──────────────────────────────────────

function vectorizeStmt(stmt: Stmt, splitLoops: string[]): Stmt {
  if (stmt instanceof ForNode) {
    // Check if this is an innermost spatial loop candidate
    if (isInnermostSpatialLoop(stmt) && stmt.extent >= SIMD_WIDTH && stmt.annotation === 'none') {
      splitLoops.push(stmt.loopVar.name);
      return splitForVectorize(stmt);
    }
    // Otherwise recurse into body
    return new ForNode(stmt.loopVar, stmt.min, stmt.extent, vectorizeStmt(stmt.body, splitLoops), stmt.annotation);
  }
  if (stmt instanceof SeqNode) {
    return new SeqNode(stmt.stmts.map(s => vectorizeStmt(s, splitLoops)));
  }
  if (stmt instanceof AllocNode) {
    return new AllocNode(stmt.buffer, vectorizeStmt(stmt.body, splitLoops));
  }
  if ((stmt as any) instanceof ScalarDeclNode) {
    const s = stmt as unknown as ScalarDeclNode;
    return new (ScalarDeclNode as any)(s.scalarName, s.initValue, vectorizeStmt(s.body as Stmt, splitLoops)) as unknown as Stmt;
  }
  return stmt;
}

/** True if this ForNode is spatial and has no nested SPATIAL ForNodes
 *  (reduction loops inside are ok — they will be left as-is) */
function isInnermostSpatialLoop(node: ForNode): boolean {
  if (node.loopVar.kind !== 'spatial') return false;
  return !containsSpatialForNode(node.body);
}

function containsForNode(stmt: Stmt): boolean {
  if (stmt instanceof ForNode) return true;
  if (stmt instanceof SeqNode) return stmt.stmts.some(containsForNode);
  if (stmt instanceof AllocNode) return containsForNode(stmt.body);
  if ((stmt as any) instanceof ScalarDeclNode) return containsForNode((stmt as any).body as Stmt);
  return false;
}

function containsSpatialForNode(stmt: Stmt): boolean {
  if (stmt instanceof ForNode) {
    if (stmt.loopVar.kind === 'spatial') return true;
    // Reduction ForNode — check its body too (might contain spatial loops)
    return containsSpatialForNode(stmt.body);
  }
  if (stmt instanceof SeqNode) return stmt.stmts.some(containsSpatialForNode);
  if (stmt instanceof AllocNode) return containsSpatialForNode(stmt.body);
  if ((stmt as any) instanceof ScalarDeclNode) return containsSpatialForNode((stmt as any).body as Stmt);
  return false;
}

// ─── Split loop for vectorization ────────────────────────────

/**
 * Split: for v in range(extent) →
 *   for v_outer in range(extent/W):
 *     for v_inner in range(W) [vectorize]:
 *       <body with v = v_outer*W + v_inner>
 */
function splitForVectorize(node: ForNode): Stmt {
  const W = SIMD_WIDTH;
  const outerExtent = Math.floor(node.extent / W);
  const tail = node.extent % W;

  const outerVar = new LoopVar(`${node.loopVar.name}_outer`, 'spatial');
  const innerVar = new LoopVar(`${node.loopVar.name}_inner`, 'spatial');

  // Substitute v → v_outer * W + v_inner in body
  const newBody = substituteLoopVar(node.body, node.loopVar.name, outerVar.name, W);

  const innerFor = new ForNode(innerVar, 0, W, newBody, 'vectorize');
  const outerFor = new ForNode(outerVar, 0, outerExtent, innerFor, 'none');

  // Handle tail elements (extent not divisible by W)
  if (tail > 0) {
    // For simplicity, emit tail as a regular loop starting at outerExtent*W
    const tailVar = new LoopVar(`${node.loopVar.name}_tail`, 'spatial');
    const tailBody = substituteLoopVarScalar(node.body, node.loopVar.name, tailVar.name);
    const tailFor = new ForNode(tailVar, outerExtent * W, tail, tailBody, 'none');
    return new SeqNode([outerFor, tailFor]);
  }

  return outerFor;
}

// ─── Index/Value substitution ─────────────────────────────────
// Replace loopVarName with (outerName * W + innerName) everywhere in the stmt tree

function substituteLoopVar(stmt: Stmt, varName: string, outerName: string, W: number): Stmt {
  const subIdx = (idx: IndexExpr): IndexExpr => substituteIndexVar(idx, varName, outerName, W);
  const subVal = (val: ValueExpr): ValueExpr => substituteValueVar(val, varName, outerName, W);

  if (stmt instanceof ForNode) {
    return new ForNode(stmt.loopVar, stmt.min, stmt.extent, substituteLoopVar(stmt.body, varName, outerName, W), stmt.annotation);
  }
  if (stmt instanceof SeqNode) {
    return new SeqNode(stmt.stmts.map(s => substituteLoopVar(s, varName, outerName, W)));
  }
  if (stmt instanceof AllocNode) {
    return new AllocNode(stmt.buffer, substituteLoopVar(stmt.body, varName, outerName, W));
  }
  if (stmt instanceof BufferStoreNode) {
    return new BufferStoreNode(stmt.buffer, stmt.indices.map(subIdx), subVal(stmt.value));
  }
  if ((stmt as any) instanceof ScalarStoreNode) {
    const s = stmt as unknown as ScalarStoreNode;
    return new (ScalarStoreNode as any)(s.scalarName, subVal(s.value)) as unknown as Stmt;
  }
  if ((stmt as any) instanceof ScalarDeclNode) {
    const s = stmt as unknown as ScalarDeclNode;
    return new (ScalarDeclNode as any)(s.scalarName, s.initValue, substituteLoopVar(s.body as Stmt, varName, outerName, W)) as unknown as Stmt;
  }
  return stmt;
}

/** Substitute varName → scalar varName (no split, just rename) */
function substituteLoopVarScalar(stmt: Stmt, varName: string, newName: string): Stmt {
  // Use W=1 and same var for inner, which effectively keeps the linear var
  // by building outerName*1 + innerName, but we can just rename
  const subIdx = (idx: IndexExpr): IndexExpr => renameIndexVar(idx, varName, newName);
  const subVal = (val: ValueExpr): ValueExpr => renameValueVar(val, varName, newName);

  if (stmt instanceof ForNode) {
    return new ForNode(stmt.loopVar, stmt.min, stmt.extent, substituteLoopVarScalar(stmt.body, varName, newName), stmt.annotation);
  }
  if (stmt instanceof SeqNode) {
    return new SeqNode(stmt.stmts.map(s => substituteLoopVarScalar(s, varName, newName)));
  }
  if (stmt instanceof AllocNode) {
    return new AllocNode(stmt.buffer, substituteLoopVarScalar(stmt.body, varName, newName));
  }
  if (stmt instanceof BufferStoreNode) {
    return new BufferStoreNode(stmt.buffer, stmt.indices.map(subIdx), subVal(stmt.value));
  }
  if ((stmt as any) instanceof ScalarStoreNode) {
    const s = stmt as unknown as ScalarStoreNode;
    return new (ScalarStoreNode as any)(s.scalarName, subVal(s.value)) as unknown as Stmt;
  }
  if ((stmt as any) instanceof ScalarDeclNode) {
    const s = stmt as unknown as ScalarDeclNode;
    return new (ScalarDeclNode as any)(s.scalarName, s.initValue, substituteLoopVarScalar(s.body as Stmt, varName, newName)) as unknown as Stmt;
  }
  return stmt;
}

// ─── Index substitution helpers ───────────────────────────────

/**
 * Replace VarIndex(varName) with VarIndex(outerName)*W + VarIndex(innerName)
 * where innerName is derived from the ForNode being split.
 *
 * We don't have access to innerName here, so we substitute with:
 *   VarIndex(outerName) * W + VarIndex(innerName)
 * by constructing a BinOpIndex.
 *
 * Since this is called on the body (which will be wrapped in an inner loop
 * with variable innerName = `varName_inner`), we use `${varName}_inner`.
 */
function substituteIndexVar(idx: IndexExpr, varName: string, outerName: string, W: number): IndexExpr {
  const innerName = `${varName}_inner`;
  if (idx instanceof VarIndex) {
    if (idx.loopVar.name === varName) {
      // v → v_outer * W + v_inner
      const outerPart = new BinOpIndex('*', new VarIndex(new LoopVar(outerName)), new ConstIndex(W));
      return new BinOpIndex('+', outerPart, new VarIndex(new LoopVar(innerName)));
    }
    return idx;
  }
  if (idx instanceof BinOpIndex) {
    return new BinOpIndex(idx.op, substituteIndexVar(idx.left, varName, outerName, W), substituteIndexVar(idx.right, varName, outerName, W));
  }
  return idx;
}

function renameIndexVar(idx: IndexExpr, varName: string, newName: string): IndexExpr {
  if (idx instanceof VarIndex) {
    return idx.loopVar.name === varName ? new VarIndex(new LoopVar(newName)) : idx;
  }
  if (idx instanceof BinOpIndex) {
    return new BinOpIndex(idx.op, renameIndexVar(idx.left, varName, newName), renameIndexVar(idx.right, varName, newName));
  }
  return idx;
}

function substituteValueVar(val: ValueExpr, varName: string, outerName: string, W: number): ValueExpr {
  const innerName = `${varName}_inner`;
  if (val instanceof VarRefExpr) {
    if (val.loopVar.name === varName) {
      // v → v_outer * W + v_inner (as a ValueExpr)
      const outerPart = new BinOpExpr('*', new VarRefExpr(new LoopVar(outerName)), new ConstExpr(W));
      return new BinOpExpr('+', outerPart, new VarRefExpr(new LoopVar(innerName)));
    }
    return val;
  }
  if (val instanceof BinOpExpr) {
    return new BinOpExpr(val.op, substituteValueVar(val.left, varName, outerName, W), substituteValueVar(val.right, varName, outerName, W));
  }
  if (val instanceof BufferLoadExpr) {
    return new BufferLoadExpr(val.buffer, val.indices.map(i => substituteIndexVar(i, varName, outerName, W)));
  }
  if (val instanceof MaxExpr) {
    return new MaxExpr(substituteValueVar(val.left, varName, outerName, W), substituteValueVar(val.right, varName, outerName, W));
  }
  if (val instanceof MinExpr) {
    return new MinExpr(substituteValueVar(val.left, varName, outerName, W), substituteValueVar(val.right, varName, outerName, W));
  }
  return val;
}

function renameValueVar(val: ValueExpr, varName: string, newName: string): ValueExpr {
  if (val instanceof VarRefExpr) {
    return val.loopVar.name === varName ? new VarRefExpr(new LoopVar(newName)) : val;
  }
  if (val instanceof BinOpExpr) {
    return new BinOpExpr(val.op, renameValueVar(val.left, varName, newName), renameValueVar(val.right, varName, newName));
  }
  if (val instanceof BufferLoadExpr) {
    return new BufferLoadExpr(val.buffer, val.indices.map(i => renameIndexVar(i, varName, newName)));
  }
  if (val instanceof MaxExpr) {
    return new MaxExpr(renameValueVar(val.left, varName, newName), renameValueVar(val.right, varName, newName));
  }
  if (val instanceof MinExpr) {
    return new MinExpr(renameValueVar(val.left, varName, newName), renameValueVar(val.right, varName, newName));
  }
  return val;
}
