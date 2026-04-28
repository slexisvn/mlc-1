// ─── Low-Level (TIR) fixture builders ─────────────────────────────

import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, AllocNode,
  BufferLoadExpr, BinOpExpr, ConstExpr, MaxExpr,
  LoopVar, BufferDecl,
  VarIndex, ConstIndex,
  type Stmt,
} from '../../src/ir/low_level.js';

// ─── Simple elemwise loop: for i: for j: Out[i,j] = max(A[i,j], 0) ───
//
//  Innermost loop j is spatial with extent=N.
//  Suitable for vectorize tests (N should be >= SIMD_WIDTH=4).
export function mkElemwiseFunc(M: number, N: number): PrimFunc {
  const iVar = new LoopVar('i', 'spatial');
  const jVar = new LoopVar('j', 'spatial');
  const aBuf = new BufferDecl('A', [M, N], 'global');
  const outBuf = new BufferDecl('Out', [M, N], 'global');

  const store = new BufferStoreNode(
    outBuf,
    [new VarIndex(iVar), new VarIndex(jVar)],
    new MaxExpr(
      new BufferLoadExpr(aBuf, [new VarIndex(iVar), new VarIndex(jVar)]),
      new ConstExpr(0)
    )
  );

  return new PrimFunc(
    'elemwise_relu',
    [aBuf, outBuf],
    new ForNode(iVar, 0, M,
      new ForNode(jVar, 0, N, store, 'none'),
      'none'
    ),
    []
  );
}

// ─── Dense loop nest with scalar accumulator ───
//
//  for i in [0,M):
//    for j in [0,N):
//      acc[0] = 0
//      for k in [0,K) [reduction]:
//        acc[0] = acc[0] + A[i,k] * W[j,k]
//      Out[i,j] = acc[0] + B[0,j]
//
//  `acc` is a local alloc of shape=[1] → storageRewrite will promote it.
//  W has shape [N,K] → layoutTransform can pack if N%blockN==0.
export function mkDenseWithAccFunc(M: number, N: number, K: number): PrimFunc {
  const iVar = new LoopVar('i', 'spatial');
  const jVar = new LoopVar('j', 'spatial');
  const kVar = new LoopVar('k', 'reduction');

  const aBuf  = new BufferDecl('A',   [M, K], 'global');
  const wBuf  = new BufferDecl('W',   [N, K], 'global');
  const bBuf  = new BufferDecl('B',   [1, N], 'global');
  const outBuf = new BufferDecl('Out', [M, N], 'global');
  const accBuf = new BufferDecl('acc', [1],   'local');

  // acc[0] = 0
  const initAcc = new BufferStoreNode(accBuf, [new ConstIndex(0)], new ConstExpr(0));

  // acc[0] = acc[0] + A[i,k] * W[j,k]
  const kBody = new BufferStoreNode(
    accBuf,
    [new ConstIndex(0)],
    new BinOpExpr('+',
      new BufferLoadExpr(accBuf, [new ConstIndex(0)]),
      new BinOpExpr('*',
        new BufferLoadExpr(aBuf,  [new VarIndex(iVar), new VarIndex(kVar)]),
        new BufferLoadExpr(wBuf,  [new VarIndex(jVar), new VarIndex(kVar)])
      )
    )
  );

  // Out[i,j] = acc[0] + B[0,j]
  const writeOut = new BufferStoreNode(
    outBuf,
    [new VarIndex(iVar), new VarIndex(jVar)],
    new BinOpExpr('+',
      new BufferLoadExpr(accBuf, [new ConstIndex(0)]),
      new BufferLoadExpr(bBuf,   [new ConstIndex(0), new VarIndex(jVar)])
    )
  );

  const jBody = new SeqNode([
    initAcc,
    new ForNode(kVar, 0, K, kBody, 'none'),
    writeOut,
  ]);

  const body = new AllocNode(
    accBuf,
    new ForNode(iVar, 0, M,
      new ForNode(jVar, 0, N, jBody, 'none'),
      'none'
    )
  );

  return new PrimFunc(
    'dense',
    [aBuf, wBuf, bBuf, outBuf],
    body,
    [accBuf]
  );
}

// ─── Producer PrimFunc for compute_inline tests ───
//
//  bias_add: for i: for j: BiasOut[i,j] = A[i,j] + B[0,j]
export function mkBiasAddProducer(M: number, N: number): PrimFunc {
  const iVar = new LoopVar('i', 'spatial');
  const jVar = new LoopVar('j', 'spatial');
  const aBuf  = new BufferDecl('A',       [M, N], 'global');
  const bBuf  = new BufferDecl('B',       [1, N], 'global');
  const outBuf = new BufferDecl('BiasOut', [M, N], 'global');

  return new PrimFunc(
    'bias_add',
    [aBuf, bBuf, outBuf],
    new ForNode(iVar, 0, M,
      new ForNode(jVar, 0, N,
        new BufferStoreNode(
          outBuf,
          [new VarIndex(iVar), new VarIndex(jVar)],
          new BinOpExpr('+',
            new BufferLoadExpr(aBuf, [new VarIndex(iVar), new VarIndex(jVar)]),
            new BufferLoadExpr(bBuf, [new ConstIndex(0), new VarIndex(jVar)])
          )
        ),
        'none'
      ),
      'none'
    ),
    []
  );
}

// ─── Consumer PrimFunc for compute_inline tests ───
//
//  relu: for i: for j: Out[i,j] = max(BiasOut[i,j], 0)
//  Reads from 'BiasOut' which is the producer's output buffer.
export function mkReluConsumer(M: number, N: number): PrimFunc {
  const iVar = new LoopVar('i', 'spatial');
  const jVar = new LoopVar('j', 'spatial');
  const inBuf  = new BufferDecl('BiasOut', [M, N], 'global');
  const outBuf = new BufferDecl('Out',     [M, N], 'global');

  return new PrimFunc(
    'relu',
    [inBuf, outBuf],
    new ForNode(iVar, 0, M,
      new ForNode(jVar, 0, N,
        new BufferStoreNode(
          outBuf,
          [new VarIndex(iVar), new VarIndex(jVar)],
          new MaxExpr(
            new BufferLoadExpr(inBuf, [new VarIndex(iVar), new VarIndex(jVar)]),
            new ConstExpr(0)
          )
        ),
        'none'
      ),
      'none'
    ),
    []
  );
}
