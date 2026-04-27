// ═══════════════════════════════════════════════════════════════
//  Layout Transform Pass (TensorIR level)
//
//  Transforms weight buffer W[N, K] into a packed layout
//  W_packed[N/blockN, K, blockN] to improve cache locality.
//
//  Why? In the inner-most k-loop, accessing W[j, k] with varying j
//  causes stride-K jumps in memory (cache-unfriendly). By packing:
//
//    W_packed[j / blockN, k, j % blockN]
//
//  the j-inner dimension is now contiguous, so a cache line
//  (typically 64 bytes = 16 floats) serves `blockN` elements of
//  different j values — drastically improving cache hit rate.
//
//  The pass:
//    1. Finds the W parameter buffer [N, K]
//    2. Creates local W_packed [N/blockN, K, blockN]
//    3. Prepends a packing loop nest inside the PrimFunc
//    4. Rewrites all W[j, k] loads to W_packed[j/blockN, k, j%blockN]
// ═══════════════════════════════════════════════════════════════

import {
  PrimFunc, BufferDecl, ForNode, SeqNode, BufferStoreNode, AllocNode,
  BufferLoadExpr, BinOpExpr, ConstExpr, MaxExpr, MinExpr, CallExprTIR,
  LoopVar, VarIndex, ConstIndex, BinOpIndex,
  type Stmt, type ValueExpr, type IndexExpr
} from '../ir/low_level.js';

// ─── Stats ───

export interface LayoutTransformStats {
  applied: boolean;
  originalShape: string;
  packedShape: string;
  packingOps: number;
  rewrittenLoads: number;
  blockN: number;
}

// ─── Index helper constructors ───

function varIdx(lv: LoopVar): IndexExpr { return new VarIndex(lv); }
function constIdx(v: number): IndexExpr { return new ConstIndex(v); }
function addIdx(a: IndexExpr, b: IndexExpr): IndexExpr { return new BinOpIndex('+', a, b); }
function mulIdx(a: IndexExpr, b: IndexExpr): IndexExpr { return new BinOpIndex('*', a, b); }
function divIdx(a: IndexExpr, b: IndexExpr): IndexExpr { return new BinOpIndex('/', a, b); }
function modIdx(a: IndexExpr, b: IndexExpr): IndexExpr { return new BinOpIndex('%', a, b); }

// ─── Rewrite counter (shared across recursion) ───

let _rewriteCount = 0;

// ─── Rewrite W loads to use W_packed ───

function rewriteValueWToPacked(
  val: ValueExpr,
  wBufName: string,
  wPacked: BufferDecl,
  bn: number
): ValueExpr {
  if (val.kind === 'load') {
    const v = val as BufferLoadExpr;
    if (v.buffer.name === wBufName && v.indices.length === 2) {
      const jIdx = v.indices[0];
      const kIdx = v.indices[1];
      // W[j, k]  →  W_packed[j/blockN, k, j%blockN]
      const joIdx = divIdx(jIdx, constIdx(bn));
      const jiIdx = modIdx(jIdx, constIdx(bn));
      _rewriteCount++;
      return new BufferLoadExpr(wPacked, [joIdx, kIdx, jiIdx]);
    }
    return val;
  }
  if (val.kind === 'binop') {
    const v = val as BinOpExpr;
    return new BinOpExpr(v.op,
      rewriteValueWToPacked(v.left, wBufName, wPacked, bn),
      rewriteValueWToPacked(v.right, wBufName, wPacked, bn)
    );
  }
  if (val.kind === 'max') {
    const v = val as MaxExpr;
    return new MaxExpr(
      rewriteValueWToPacked(v.left, wBufName, wPacked, bn),
      rewriteValueWToPacked(v.right, wBufName, wPacked, bn)
    );
  }
  if (val.kind === 'min') {
    const v = val as MinExpr;
    return new MinExpr(
      rewriteValueWToPacked(v.left, wBufName, wPacked, bn),
      rewriteValueWToPacked(v.right, wBufName, wPacked, bn)
    );
  }
  if (val.kind === 'call') {
    const v = val as CallExprTIR;
    return new CallExprTIR(v.funcName,
      v.args.map(a => rewriteValueWToPacked(a, wBufName, wPacked, bn))
    );
  }
  return val;
}

function rewriteStmtWToPacked(
  stmt: Stmt,
  wBufName: string,
  wPacked: BufferDecl,
  bn: number
): Stmt {
  if (stmt instanceof ForNode) {
    return new ForNode(stmt.loopVar, stmt.min, stmt.extent,
      rewriteStmtWToPacked(stmt.body, wBufName, wPacked, bn),
      stmt.annotation
    );
  }
  if (stmt instanceof SeqNode) {
    return new SeqNode(stmt.stmts.map(s =>
      rewriteStmtWToPacked(s, wBufName, wPacked, bn)
    ));
  }
  if (stmt instanceof BufferStoreNode) {
    return new BufferStoreNode(stmt.buffer, stmt.indices,
      rewriteValueWToPacked(stmt.value, wBufName, wPacked, bn)
    );
  }
  if (stmt instanceof AllocNode) {
    return new AllocNode(stmt.buffer,
      rewriteStmtWToPacked(stmt.body, wBufName, wPacked, bn)
    );
  }
  return stmt;
}

// ═══════════════════════════════════════
//  Main Pass Entry Point
// ═══════════════════════════════════════

export function layoutTransform(pf: PrimFunc, blockN = 16): {
  transformed: PrimFunc;
  stats: LayoutTransformStats;
} {
  const notApplied: LayoutTransformStats = {
    applied: false, originalShape: '', packedShape: '',
    packingOps: 0, rewrittenLoads: 0, blockN,
  };

  // Find W buffer: typically [N, K], named 'W'
  const wBuf = pf.params.find(p => p.name === 'W' && p.shape.length === 2);
  if (!wBuf) return { transformed: pf, stats: notApplied };

  const [N, K] = wBuf.shape;
  if (N < blockN || N % blockN !== 0) {
    return { transformed: pf, stats: notApplied };
  }

  const numBlocks = N / blockN;
  const wPacked = new BufferDecl('W_packed', [numBlocks, K, blockN], 'local');

  // ─── Build packing stage ───
  // for jo in [0, numBlocks):
  //   for kl in [0, K):
  //     for ji in [0, blockN):
  //       W_packed[jo, kl, ji] = W[jo * blockN + ji, kl]
  const joVar = new LoopVar('_jo_pack', 'spatial');
  const klVar = new LoopVar('_kl_pack', 'spatial');
  const jiVar = new LoopVar('_ji_pack', 'spatial');

  const packStmt: Stmt = new ForNode(joVar, 0, numBlocks,
    new ForNode(klVar, 0, K,
      new ForNode(jiVar, 0, blockN,
        new BufferStoreNode(wPacked,
          [varIdx(joVar), varIdx(klVar), varIdx(jiVar)],
          new BufferLoadExpr(wBuf, [
            addIdx(mulIdx(varIdx(joVar), constIdx(blockN)), varIdx(jiVar)),
            varIdx(klVar)
          ])
        )
      )
    )
  );

  const packingOps = numBlocks * K * blockN;

  // ─── Rewrite body ───
  _rewriteCount = 0;
  const newBody = rewriteStmtWToPacked(pf.body, wBuf.name, wPacked, blockN);
  const rewrittenLoads = _rewriteCount;

  // ─── Assemble new PrimFunc ───
  const newPf = new PrimFunc(
    `${pf.name}_packed`,
    pf.params,                                          // W still in params (input)
    new SeqNode([packStmt, newBody]),
    [...pf.allocations, wPacked]                        // W_packed as local alloc
  );

  return {
    transformed: newPf,
    stats: {
      applied: true,
      originalShape: `[${N}, ${K}]`,
      packedShape: `[${numBlocks}, ${K}, ${blockN}]`,
      packingOps,
      rewrittenLoads,
      blockN,
    },
  };
}
