// ─── Unit tests: dead_code_elimination pass ───────────────────────
import { expect, test } from 'vitest';
import { assertHLIRValid } from '../helpers/verify.js';
import { mkVar, mkConst, mkCall, mkModule } from '../helpers/hlir_builders.js';
import {
  deadCodeElimination,
  wrapWithDeadCode,
} from '../../src/transform/dead_code_elimination.js';
import { LetExpr, VarExpr, TensorType, IRModule, IRFunction } from '../../src/ir/high_level.js';

// ─── Test 1: wrapWithDeadCode + DCE eliminates the dead let ──────
test('wrapWithDeadCode → DCE eliminates dead binding (eliminated=1)', () => {
  const c = mkConst([1, 2, 3, 4], [2, 2], 'c');
  const reluCall = mkCall('nn.relu', [c]);
  const base = mkModule('f', [], reluCall, [2, 2]);

  const withDead = wrapWithDeadCode(base);
  const { module: cleaned, stats } = deadCodeElimination(withDead);

  expect(stats.eliminated).toBe(1);
  expect(stats.totalAfter).toBe(stats.totalBefore - 1);
});

// ─── Test 2: live LetExpr is preserved ───────────────────────────
test('live LetExpr binding is preserved (eliminated=0)', () => {
  // Build: let _tmp = relu(c) in _tmp
  const c = mkConst([1, 2], [1, 2], 'c');
  const reluCall = mkCall('nn.relu', [c]);
  const tmpVar = new VarExpr('_tmp', new TensorType([1, 2]));
  const letBody = new LetExpr(tmpVar, reluCall, tmpVar); // uses _tmp in body

  const mod = new IRModule();
  mod.addFunction(new IRFunction('f', [], letBody, new TensorType([1, 2])));

  const { stats } = deadCodeElimination(mod);
  expect(stats.eliminated).toBe(0);
});

// ─── Test 3: clean module → stats all zero ───────────────────────
test('module without dead code → stats.eliminated = 0', () => {
  const c = mkConst([1.0], [1], 'c');
  const relu = mkCall('nn.relu', [c]);
  const mod = mkModule('f', [], relu, [1]);

  const { stats } = deadCodeElimination(mod);
  expect(stats.eliminated).toBe(0);
});

// ─── Test 4: stats fields are consistent ─────────────────────────
test('stats.eliminated = totalBefore - totalAfter', () => {
  const c = mkConst([1, 2], [1, 2], 'c');
  const relu = mkCall('nn.relu', [c]);
  const base = mkModule('f', [], relu, [1, 2]);
  const withDead = wrapWithDeadCode(base);

  const { stats } = deadCodeElimination(withDead);
  expect(stats.eliminated).toBe(stats.totalBefore - stats.totalAfter);
});

// ─── Test 5: verifier passes after DCE ───────────────────────────
test('verifier passes after DCE', () => {
  const x = mkVar('x', [4, 8]);
  const W = mkConst(new Array(8 * 8).fill(0.1), [8, 8], 'W');
  const B = mkConst(new Array(8).fill(0), [1, 8], 'B');
  const dense   = mkCall('nn.dense',   [x, W], [4, 8]);
  const biasAdd = mkCall('nn.bias_add', [dense, B], [4, 8]);
  const relu    = mkCall('nn.relu',    [biasAdd], [4, 8]);
  const mod = mkModule('f', [x], relu, [4, 8]);

  const withDead = wrapWithDeadCode(mod);
  const { module: cleaned } = deadCodeElimination(withDead);
  assertHLIRValid(cleaned, 'after DCE');
});

