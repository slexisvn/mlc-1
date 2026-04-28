// ─── Unit tests: storage_rewrite pass ────────────────────────────
import { expect, test } from 'vitest';
import { mkDenseWithAccFunc } from '../helpers/tir_builders.js';
import { storageRewrite } from '../../src/transform/storage_rewrite.js';
import {
  PrimFunc, AllocNode, BufferDecl,
  ForNode, SeqNode, BufferStoreNode,
  LoopVar, ConstExpr, ConstIndex,
  type Stmt,
} from '../../src/ir/low_level.js';

// ─── Helper: PrimFunc with a small non-scalar local alloc ─────────

function mkLargeAllocFunc(): PrimFunc {
  // acc[4] is NOT a scalar → should NOT be promoted
  const accBuf = new BufferDecl('acc', [4], 'local');
  const outBuf = new BufferDecl('Out', [2, 4], 'global');
  const iVar   = new LoopVar('i', 'spatial');
  const jVar   = new LoopVar('j', 'spatial');

  const store = new BufferStoreNode(
    outBuf,
    [new ConstIndex(0), new ConstIndex(0)],
    new ConstExpr(0)
  );

  const body = new AllocNode(
    accBuf,
    new ForNode(iVar, 0, 2, new ForNode(jVar, 0, 4, store, 'none'), 'none')
  );

  return new PrimFunc('non_scalar', [outBuf], body, [accBuf]);
}

// ─── Test 1: scalar alloc [1] is promoted ────────────────────────
test('scalar alloc [1] → promoted, allocations become empty', () => {
  const func = mkDenseWithAccFunc(2, 4, 2);
  expect(func.allocations.length).toBe(1);

  const { func: rewritten, stats } = storageRewrite(func);
  expect(rewritten.allocations.length).toBe(0);
  expect(stats.promotedToScalar.length).toBeGreaterThan(0);
});

// ─── Test 2: promoted scalar name is reported in stats ───────────
test("stats.promotedToScalar contains 'acc'", () => {
  const func = mkDenseWithAccFunc(2, 4, 2);
  const { stats } = storageRewrite(func);
  expect(stats.promotedToScalar.includes('acc')).toBe(true);
});

// ─── Test 3: originalAllocBytes > optimizedAllocBytes ────────────
test('originalAllocBytes > optimizedAllocBytes after scalar promotion', () => {
  const func = mkDenseWithAccFunc(2, 4, 2);
  const { stats } = storageRewrite(func);
  expect(stats.originalAllocBytes).toBeGreaterThan(0);
  expect(stats.optimizedAllocBytes).toBe(0);
});

// ─── Test 4: large alloc [4] is NOT promoted ─────────────────────
test('alloc [4] (non-scalar) is NOT promoted', () => {
  const func = mkLargeAllocFunc();
  const { func: rewritten, stats } = storageRewrite(func);
  expect(stats.promotedToScalar.length).toBe(0);
  expect(rewritten.allocations.length).toBe(1);
});

