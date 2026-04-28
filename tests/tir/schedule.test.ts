// ─── Unit tests: Schedule transformations ─────────────────────────
import { expect, test } from 'vitest';
import { mkDenseWithAccFunc, mkElemwiseFunc } from '../helpers/tir_builders.js';
import { Schedule, applyDefaultSchedule } from '../../src/transform/schedule.js';

// ─── Helper: get a clean dense PrimFunc with known loop extents ───
// M=4, N=32, K=16 → loops i(4), j(32), k(16-reduction)
function denseFunc() { return mkDenseWithAccFunc(4, 32, 16); }

// ─── Test 1: split creates two nested loops ───────────────────────
test('split(j, 8) creates j_outer and j_inner loops', () => {
  const sch = new Schedule(denseFunc());
  const j = sch.getLoop('j');
  const [jOuter, jInner] = sch.split(j, 8);
  expect(jOuter.name).toBe('j_outer');
  expect(jInner.name).toBe('j_inner');
});

// ─── Test 2: split loop extents are correct ───────────────────────
test('split(j=32, factor=8) → outer=4, inner=8', () => {
  const sch = new Schedule(denseFunc());
  const j = sch.getLoop('j');
  sch.split(j, 8);
  const built = sch.build();
  const loops = built.getLoops();
  const outer = loops.find(l => l.forNode.loopVar.name === 'j_outer');
  const inner = loops.find(l => l.forNode.loopVar.name === 'j_inner');
  expect(outer !== undefined).toBe(true);
  expect(inner !== undefined).toBe(true);
  expect(outer!.forNode.extent).toBe(4);
  expect(inner!.forNode.extent).toBe(8);
});

// ─── Test 3: parallel annotation ─────────────────────────────────
test('parallel(j_outer) sets annotation to "parallel"', () => {
  const sch = new Schedule(denseFunc());
  const j = sch.getLoop('j');
  const [jOuter] = sch.split(j, 8);
  sch.parallel(jOuter);

  const built = sch.build();
  const loops = built.getLoops();
  const outer = loops.find(l => l.forNode.loopVar.name === 'j_outer');
  expect(outer !== undefined).toBe(true);
  expect(outer!.forNode.annotation).toBe('parallel');
});

// ─── Test 4: unroll annotation ────────────────────────────────────
test('unroll(j_inner) sets annotation to "unroll"', () => {
  const sch = new Schedule(denseFunc());
  const j = sch.getLoop('j');
  const [, jInner] = sch.split(j, 4);
  sch.unroll(jInner);

  const built = sch.build();
  const loops = built.getLoops();
  const inner = loops.find(l => l.forNode.loopVar.name === 'j_inner');
  expect(inner !== undefined).toBe(true);
  expect(inner!.forNode.annotation).toBe('unroll');
});

// ─── Test 5: vectorize annotation ────────────────────────────────
test('vectorize(j_inner) sets annotation to "vectorize"', () => {
  const sch = new Schedule(denseFunc());
  const j = sch.getLoop('j');
  const [, jInner] = sch.split(j, 4);
  sch.vectorize(jInner);

  const built = sch.build();
  const loops = built.getLoops();
  const inner = loops.find(l => l.forNode.loopVar.name === 'j_inner');
  expect(inner !== undefined).toBe(true);
  expect(inner!.forNode.annotation).toBe('vectorize');
});

// ─── Test 6: tile creates 4 loops ────────────────────────────────
test('tile(i, j, 2, 8) creates i_outer, j_outer, i_inner, j_inner', () => {
  const sch = new Schedule(denseFunc());
  const i = sch.getLoop('i');
  const j = sch.getLoop('j');
  const [iOut, jOut, iIn, jIn] = sch.tile(i, j, 2, 8);
  expect(iOut.name).toBe('i_outer');
  expect(jOut.name).toBe('j_outer');
  expect(iIn.name).toBe('i_inner');
  expect(jIn.name).toBe('j_inner');
});

// ─── Test 7: fuse two loops into one ─────────────────────────────
test('fuse(i_outer, i_inner) creates fused loop with extent outer*inner', () => {
  const func = mkElemwiseFunc(4, 8);
  const sch  = new Schedule(func);
  const i = sch.getLoop('i');
  const [iOuter, iInner] = sch.split(i, 2);   // i_outer(2), i_inner(2)
  const fused = sch.fuse(iOuter, iInner);

  expect(fused.name).toBe('i_outer_i_inner');
  const built = sch.build();
  const loops  = built.getLoops();
  const fusedLoop = loops.find(l => l.forNode.loopVar.name === 'i_outer_i_inner');
  expect(fusedLoop !== undefined).toBe(true);
  expect(fusedLoop!.forNode.extent).toBe(4);
});

// ─── Test 8: cacheRead creates W_local in allocations ─────────────
test('cacheRead("W", j_outer, 8) adds W_local to allocations', () => {
  const sch = new Schedule(denseFunc());
  const j = sch.getLoop('j');
  const [jOuter] = sch.split(j, 8);  // j_outer(4), j_inner(8)
  const wLocal = sch.cacheRead('W', jOuter, 8);

  expect(wLocal.name).toBe('W_local');
  const built = sch.build();
  const hasWLocal = built.allocations.some(a => a.name === 'W_local');
  expect(hasWLocal).toBe(true);
});

test('reorder changes outermost loop order', () => {
  const sch = new Schedule(denseFunc());
  const i = sch.getLoop('i');
  const j = sch.getLoop('j');
  sch.reorder([j, i]);

  const built = sch.build();
  expect(built.getLoops()[0].loopVar.name).toBe('j');
});

test('rfactor creates acc_rf allocation and parallel outer reduction loop', () => {
  const sch = new Schedule(denseFunc());
  const k = sch.getLoop('k');
  const [kOuter, kInner] = sch.rfactor(k, 4);
  const built = sch.build();

  expect(kOuter.name).toBe('k_outer');
  expect(kInner.name).toBe('k_inner');
  expect(built.allocations.some(a => a.name === 'acc_rf')).toBe(true);
  const outerLoop = built.getLoops().find(l => l.loopVar.name === 'k_outer');
  expect(outerLoop?.forNode.annotation).toBe('parallel');
});

test('computeAt is currently a no-op stub', () => {
  const sch = new Schedule(denseFunc());
  const before = sch.build();
  sch.computeAt('W', sch.getLoop('j'));
  const after = sch.build();
  expect(after.name).toBe(before.name);
});

test('applyDefaultSchedule returns naive func when loops are too small', () => {
  const func = mkDenseWithAccFunc(2, 8, 8);
  const scheduled = applyDefaultSchedule(func);
  expect(scheduled.getLoops()[0].loopVar.name).toBe('i');
});

test('applyDefaultSchedule tiles and annotates larger loops', () => {
  const scheduled = applyDefaultSchedule(mkDenseWithAccFunc(2, 32, 32));
  const hasJOuter = scheduled.getLoops().some(l => l.loopVar.name === 'j_outer');
  const parallelLoop = scheduled.getLoops().find(l => l.loopVar.name === 'j_outer');
  expect(hasJOuter).toBe(true);
  expect(parallelLoop?.forNode.annotation).toBe('parallel');
});

