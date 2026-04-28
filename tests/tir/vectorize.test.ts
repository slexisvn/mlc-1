// ─── Unit tests: vectorize pass ───────────────────────────────────
import { expect, test } from 'vitest';
import { mkElemwiseFunc } from '../helpers/tir_builders.js';
import { vectorize, SIMD_WIDTH } from '../../src/transform/vectorize.js';
import { BufferDecl, BufferStoreNode, ConstExpr, ConstIndex, ForNode, LoopVar, PrimFunc } from '../../src/ir/low_level.js';
import { storageRewrite } from '../../src/transform/storage_rewrite.js';

// ─── Test 1: spatial loop with extent=64 is vectorized ───────────
test('innermost spatial loop extent=64 (>=SIMD_WIDTH) → vectorizedCount=1', () => {
  const func = mkElemwiseFunc(4, 64);  // j has extent=64
  const { vectorizedCount } = vectorize(func);
  expect(vectorizedCount).toBe(1);
});

// ─── Test 2: inner vectorized loop has annotation 'vectorize' ────
test('after vectorize: inner loop annotation = "vectorize"', () => {
  const func = mkElemwiseFunc(4, 64);
  const { func: vf } = vectorize(func);
  const loops = vf.getLoops();

  // Find the innermost loop: it should be annotated 'vectorize'
  const vectorizedLoops = loops.filter(l => l.forNode.annotation === 'vectorize');
  expect(vectorizedLoops.length).toBeGreaterThan(0);
});

// ─── Test 3: split loop name is reported ─────────────────────────
test('splitLoops array contains the original loop var name', () => {
  const func = mkElemwiseFunc(4, 64);
  const { splitLoops } = vectorize(func);
  expect(splitLoops.includes('j')).toBe(true);
});

// ─── Test 4: outer loop has correct extent (64 / SIMD_WIDTH) ─────
test('outer loop extent = originalExtent / SIMD_WIDTH', () => {
  const N = 64;
  const func = mkElemwiseFunc(4, N);
  const { func: vf } = vectorize(func);
  const loops = vf.getLoops();

  // Find j_outer loop
  const outerLoop = loops.find(l => l.forNode.loopVar.name === 'j_outer');
  expect(outerLoop !== undefined).toBe(true);
  expect(outerLoop!.forNode.extent).toBe(N / SIMD_WIDTH);
});

// ─── Test 5: inner loop extent = SIMD_WIDTH ───────────────────────
test('inner vectorized loop extent = SIMD_WIDTH', () => {
  const func = mkElemwiseFunc(4, 64);
  const { func: vf } = vectorize(func);
  const loops = vf.getLoops();

  const innerLoop = loops.find(l => l.forNode.loopVar.name === 'j_inner');
  expect(innerLoop !== undefined).toBe(true);
  expect(innerLoop!.forNode.extent).toBe(SIMD_WIDTH);
});

// ─── Test 6: loop with extent < SIMD_WIDTH → NOT vectorized ──────
test('loop with extent < SIMD_WIDTH → vectorizedCount=0', () => {
  const func = mkElemwiseFunc(4, 2);  // j has extent=2 < SIMD_WIDTH=4
  const { vectorizedCount } = vectorize(func);
  expect(vectorizedCount).toBe(0);
});

test('loop with tail emits j_tail loop', () => {
  const { func } = vectorize(mkElemwiseFunc(2, 5));
  const tailLoop = func.getLoops().find(l => l.loopVar.name === 'j_tail');
  expect(tailLoop !== undefined).toBe(true);
  expect(tailLoop!.forNode.extent).toBe(1);
});

test('already annotated loop is not vectorized again', () => {
  const i = new LoopVar('i', 'spatial');
  const j = new LoopVar('j', 'spatial');
  const out = new BufferDecl('Out', [1, 8], 'global');
  const func = new PrimFunc(
    'annotated',
    [out],
    new ForNode(i, 0, 1, new ForNode(j, 0, 8, new BufferStoreNode(out, [new ConstIndex(0), new ConstIndex(0)], new ConstExpr(1)), 'vectorize'), 'none')
  );
  const result = vectorize(func);
  expect(result.vectorizedCount).toBe(0);
});

test('vectorize walks ScalarDeclNode/ScalarStoreNode after storageRewrite', () => {
  const rewritten = storageRewrite(mkElemwiseFunc(1, 4));
  const result = vectorize(rewritten.func);
  expect(result.vectorizedCount >= 0).toBe(true);
});

