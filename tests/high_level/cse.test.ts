// ─── Unit tests: cse (common subexpression elimination) pass ─────
import { expect, test } from 'vitest';
import { assertHLIRValid } from '../helpers/verify.js';
import { mkVar, mkConst, mkCall, mkModule } from '../helpers/hlir_builders.js';
import { cseModule } from '../../src/transform/cse.js';

// ─── Test 1: simple chain creates LetExpr bindings ───────────────
test('simple 2-call chain creates bindings (bindings=2)', () => {
  const x = mkVar('x', [4, 8]);
  const W = mkConst(new Array(8 * 8).fill(0.1), [8, 8], 'W');
  const dense = mkCall('nn.dense',  [x, W], [4, 8]);
  const relu  = mkCall('nn.relu',   [dense], [4, 8]);
  const mod   = mkModule('f', [x], relu, [4, 8]);

  const { stats } = cseModule(mod);
  expect(stats.bindings).toBe(2);
  expect(stats.checked).toBe(2);
});

// ─── Test 2: same object used twice → replaced via object identity ─
test('same CallExpr object used twice → replaced=1', () => {
  const x = mkVar('x', [1, 4]);
  const W = mkConst(new Array(4 * 4).fill(0.1), [4, 4], 'W');
  // dense is referenced by both relu1 and relu2 (same JS object)
  const dense = mkCall('nn.dense', [x, W], [1, 4]);
  const relu1 = mkCall('nn.relu',  [dense], [1, 4]);
  const relu2 = mkCall('nn.relu',  [dense], [1, 4]);  // same `dense` object
  const add   = mkCall('add',      [relu1, relu2], [1, 4]);
  const mod   = mkModule('f', [x], add, [1, 4]);

  const { stats } = cseModule(mod);
  // dense appears twice as input → second occurrence is replaced
  expect(stats.replaced).toBeGreaterThan(0);
});

// ─── Test 3: structurally identical CallExprs → true CSE ─────────
test('structurally identical CallExprs (different objects) → replaced=1', () => {
  const x = mkVar('x', [1, 4]);
  // Two separate relu(x) objects with the same structure
  const relu1 = mkCall('nn.relu', [x], [1, 4]);
  const relu2 = mkCall('nn.relu', [x], [1, 4]);  // different object, same hash
  const add   = mkCall('add', [relu1, relu2], [1, 4]);
  const mod   = mkModule('f', [x], add, [1, 4]);

  const { stats } = cseModule(mod);
  expect(stats.replaced).toBe(1);
});

// ─── Test 4: different args → NOT deduplicated ───────────────────
test('CallExprs with different args → not deduplicated (replaced=0)', () => {
  const x = mkVar('x', [1, 4]);
  const c1 = mkConst([1, 2, 3, 4], [1, 4], 'c1');
  const c2 = mkConst([5, 6, 7, 8], [1, 4], 'c2');
  const relu1 = mkCall('nn.relu', [c1], [1, 4]);  // relu(c1)
  const relu2 = mkCall('nn.relu', [c2], [1, 4]);  // relu(c2) — different arg
  const add   = mkCall('add', [relu1, relu2], [1, 4]);
  const mod   = mkModule('f', [x], add, [1, 4]);

  const { stats } = cseModule(mod);
  expect(stats.replaced).toBe(0);
});

// ─── Test 5: verifier passes after CSE ───────────────────────────
test('verifier passes after CSE', () => {
  const x = mkVar('x', [4, 8]);
  const W = mkConst(new Array(8 * 16).fill(0.1), [16, 8], 'W');
  const B = mkConst(new Array(16).fill(0), [1, 16], 'B');
  const dense   = mkCall('nn.dense',    [x, W], [4, 16]);
  const biasAdd = mkCall('nn.bias_add', [dense, B], [4, 16]);
  const relu    = mkCall('nn.relu',     [biasAdd], [4, 16]);
  const mod     = mkModule('f', [x], relu, [4, 16]);

  const { module: cseMod } = cseModule(mod);
  assertHLIRValid(cseMod, 'after CSE');
});

