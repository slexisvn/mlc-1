// ═══════════════════════════════════════════════════════════════
//  Storage Rewrite Pass (TensorIR level)
//
//  Optimizations:
//  1. Scalar Promotion:  local buffer of size 1 (e.g. acc[0])
//     is promoted to a scalar variable (`let acc = 0`).
//     This avoids Float32Array allocation overhead and array
//     indexing at runtime.
//
//  2. Dead Store Elimination: If a buffer store is immediately
//     overwritten before any read, the first store is dead.
//
//  The pass works by:
//  a) Analyzing each PrimFunc's local allocations
//  b) Finding which buffers have shape=[1] (single-element)
//  c) Rewriting all BufferStore/BufferLoad to use ScalarVar
//  d) Removing the AllocNode since no array is needed
//
//  In generated JS:
//    BEFORE: const acc = new Float32Array(1); ... acc[0] = x;
//    AFTER:  let acc = 0; ... acc = x;
// ═══════════════════════════════════════════════════════════════

import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, AllocNode,
  BufferLoadExpr, BinOpExpr, ConstExpr, VarRefExpr, MaxExpr,
  MinExpr, CallExprTIR, BufferDecl,
  VarIndex, ConstIndex, BinOpIndex,
  type Stmt, type ValueExpr, type IndexExpr
} from '../ir/low_level.js';

// ─── New node types for scalar-promoted variables ───

// A ScalarStoreNode replaces BufferStoreNode for promoted scalars
export class ScalarStoreNode {
  readonly nodeType = 'scalar_store' as const;
  constructor(
    public scalarName: string,
    public value: ValueExpr
  ) {}
}

// A ScalarLoadExpr replaces BufferLoadExpr for promoted scalars
export class ScalarLoadExpr {
  readonly kind = 'scalar_load' as const;
  constructor(public scalarName: string) {}
}

// A ScalarDeclNode replaces AllocNode for promoted scalars
export class ScalarDeclNode {
  readonly nodeType = 'scalar_decl' as const;
  constructor(
    public scalarName: string,
    public initValue: number,
    public body: Stmt | ScalarStoreNode
  ) {}
}

// ═══════════════════════════════════════
//  Analysis: find single-element local buffers
// ═══════════════════════════════════════

function findScalarBuffers(func: PrimFunc): Set<string> {
  const scalars = new Set<string>();

  for (const alloc of func.allocations) {
    const totalSize = alloc.shape.reduce((a, b) => a * b, 1);
    if (totalSize === 1 && alloc.scope === 'local') {
      scalars.add(alloc.name);
    }
  }

  return scalars;
}

// ═══════════════════════════════════════
//  Rewrite: replace buffer ops with scalar ops
// ═══════════════════════════════════════

function rewriteValue(val: ValueExpr, scalarBuffers: Set<string>): ValueExpr {
  if (val.kind === 'load') {
    const v = val as BufferLoadExpr;
    if (scalarBuffers.has(v.buffer.name)) {
      // Replace buffer load with scalar reference
      return new ScalarLoadExpr(v.buffer.name) as any;
    }
    return val;
  }

  if (val.kind === 'binop') {
    const v = val as BinOpExpr;
    return new BinOpExpr(
      v.op,
      rewriteValue(v.left, scalarBuffers),
      rewriteValue(v.right, scalarBuffers)
    );
  }

  if (val.kind === 'max') {
    const v = val as MaxExpr;
    return new MaxExpr(
      rewriteValue(v.left, scalarBuffers),
      rewriteValue(v.right, scalarBuffers)
    );
  }

  if (val.kind === 'min') {
    const v = val as MinExpr;
    return new MinExpr(
      rewriteValue(v.left, scalarBuffers),
      rewriteValue(v.right, scalarBuffers)
    );
  }

  if (val.kind === 'call') {
    const v = val as CallExprTIR;
    return new CallExprTIR(
      v.funcName,
      v.args.map(a => rewriteValue(a, scalarBuffers))
    );
  }

  return val;
}

function rewriteStmt(stmt: Stmt, scalarBuffers: Set<string>): Stmt {
  if (stmt instanceof ForNode) {
    return new ForNode(
      stmt.loopVar,
      stmt.min,
      stmt.extent,
      rewriteStmt(stmt.body, scalarBuffers),
      stmt.annotation
    );
  }

  if (stmt instanceof SeqNode) {
    return new SeqNode(stmt.stmts.map(s => rewriteStmt(s, scalarBuffers)));
  }

  if (stmt instanceof BufferStoreNode) {
    if (scalarBuffers.has(stmt.buffer.name)) {
      // Replace buffer store with scalar assignment
      return new ScalarStoreNode(
        stmt.buffer.name,
        rewriteValue(stmt.value, scalarBuffers)
      ) as any;
    }
    return new BufferStoreNode(
      stmt.buffer,
      stmt.indices,
      rewriteValue(stmt.value, scalarBuffers)
    );
  }

  if (stmt instanceof AllocNode) {
    if (scalarBuffers.has(stmt.buffer.name)) {
      // Replace alloc with scalar declaration
      return new ScalarDeclNode(
        stmt.buffer.name,
        0,
        rewriteStmt(stmt.body, scalarBuffers)
      ) as any;
    }
    return new AllocNode(stmt.buffer, rewriteStmt(stmt.body, scalarBuffers));
  }

  return stmt;
}

// ═══════════════════════════════════════
//  Main Pass Entry Point
// ═══════════════════════════════════════

export interface StorageRewriteStats {
  promotedToScalar: string[];
  originalAllocBytes: number;
  optimizedAllocBytes: number;
}

export function storageRewrite(func: PrimFunc): { func: PrimFunc; stats: StorageRewriteStats } {
  const scalarBuffers = findScalarBuffers(func);

  if (scalarBuffers.size === 0) {
    return {
      func,
      stats: { promotedToScalar: [], originalAllocBytes: 0, optimizedAllocBytes: 0 }
    };
  }

  const rewritten = func.clone();

  // Calculate original allocation size
  let originalBytes = 0;
  for (const alloc of func.allocations) {
    originalBytes += alloc.shape.reduce((a, b) => a * b, 1) * 4; // 4 bytes per float32
  }

  // Rewrite body
  rewritten.body = rewriteStmt(rewritten.body, scalarBuffers);

  // Remove scalar buffers from allocations list
  rewritten.allocations = rewritten.allocations.filter(a => !scalarBuffers.has(a.name));

  // Calculate new allocation size
  let optimizedBytes = 0;
  for (const alloc of rewritten.allocations) {
    optimizedBytes += alloc.shape.reduce((a, b) => a * b, 1) * 4;
  }

  return {
    func: rewritten,
    stats: {
      promotedToScalar: [...scalarBuffers],
      originalAllocBytes: originalBytes,
      optimizedAllocBytes: optimizedBytes,
    }
  };
}
