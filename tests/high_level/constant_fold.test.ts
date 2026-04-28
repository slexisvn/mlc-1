// ─── Unit tests: constant_fold pass ───────────────────────────────
import { test } from 'vitest';
import { expect } from 'vitest';
import { assertHLIRValid } from '../helpers/verify.js';
import { mkVar, mkConst, mkCall, mkModule } from '../helpers/hlir_builders.js';
import { constantFold } from '../../src/transform/constant_fold.js';

// ─── Test 1: fold add of two constants ───────────────────────────
test('fold add of two constants → ConstantExpr', () => {
  const c1 = mkConst([1, 2, 3, 4], [2, 2], 'c1');
  const c2 = mkConst([5, 6, 7, 8], [2, 2], 'c2');
  const addCall = mkCall('add', [c1, c2]);
  const mod = mkModule('f', [], addCall, [2, 2]);

  const folded = constantFold(mod);
  const body = folded.getFunction('f')!.body;
  expect(body.kind).toBe('constant');
});

// ─── Test 2: fold relu of a constant ─────────────────────────────
test('fold relu of constant → ConstantExpr with relu applied', () => {
  // relu(-1, -2, 3, 4) → [0, 0, 3, 4]
  const c = mkConst([-1, -2, 3, 4], [2, 2], 'c');
  const reluCall = mkCall('nn.relu', [c]);
  const mod = mkModule('f', [], reluCall, [2, 2]);

  const folded = constantFold(mod);
  const body = folded.getFunction('f')!.body;
  expect(body.kind).toBe('constant');

  // Verify the values are relu-applied
  if (body.kind === 'constant') {
    expect(body.data.data[0]).toBe(0);
    expect(body.data.data[1]).toBe(0);
    expect(body.data.data[2]).toBe(3);
    expect(body.data.data[3]).toBe(4);
  }
});

// ─── Test 3: fold nested: relu(add(c1,c2)) → single constant ─────
test('fold nested relu(add(c1,c2)) → single constant', () => {
  const c1 = mkConst([-3, 2], [1, 2], 'c1');
  const c2 = mkConst([1, -5], [1, 2], 'c2');
  // add: [-2, -3], relu: [0, 0]
  const addCall  = mkCall('add',    [c1, c2]);
  const reluCall = mkCall('nn.relu', [addCall]);
  const mod = mkModule('f', [], reluCall, [1, 2]);

  const folded = constantFold(mod);
  const body = folded.getFunction('f')!.body;
  expect(body.kind).toBe('constant');
  if (body.kind === 'constant') {
    expect(body.data.data[0]).toBe(0);
    expect(body.data.data[1]).toBe(0);
  }
});

// ─── Test 4: no fold when one arg is VarExpr ─────────────────────
test('no fold when arg is VarExpr → CallExpr preserved', () => {
  const x = mkVar('x', [2, 2]);
  const c = mkConst([1, 2, 3, 4], [2, 2], 'c');
  const addCall = mkCall('add', [x, c]);
  const mod = mkModule('f', [x], addCall, [2, 2]);

  const folded = constantFold(mod);
  const body = folded.getFunction('f')!.body;
  expect(body.kind).toBe('call');
});

// ─── Test 5: verifier passes after constant fold ─────────────────
test('verifier passes after constant fold', () => {
  const c1 = mkConst([0.5, 0.5], [1, 2], 'c1');
  const c2 = mkConst([0.1, 0.2], [1, 2], 'c2');
  const addCall = mkCall('add', [c1, c2]);
  const mod = mkModule('f', [], addCall, [1, 2]);

  const folded = constantFold(mod);
  assertHLIRValid(folded, 'after constant fold');
});

test('fold nn.sigmoid / nn.neg / nn.exp / nn.log constants', () => {
  const c = mkConst([0, 1], [1, 2], 'c');
  const sigmoid = constantFold(mkModule('sigmoid', [], mkCall('nn.sigmoid', [c]), [1, 2])).getFunction('sigmoid')!.body;
  const neg = constantFold(mkModule('neg', [], mkCall('nn.neg', [c]), [1, 2])).getFunction('neg')!.body;
  const exp = constantFold(mkModule('exp', [], mkCall('nn.exp', [c]), [1, 2])).getFunction('exp')!.body;
  const logInput = mkConst([1, Math.E], [1, 2], 'log_input');
  const log = constantFold(mkModule('log', [], mkCall('nn.log', [logInput]), [1, 2])).getFunction('log')!.body;

  expect(sigmoid.kind).toBe('constant');
  expect(neg.kind).toBe('constant');
  expect(exp.kind).toBe('constant');
  expect(log.kind).toBe('constant');
});

test('fold nn.dense and nn.bias_add when all args are constants', () => {
  const lhs = mkConst([1, 2, 3, 4], [2, 2], 'lhs');
  const rhs = mkConst([1, 0, 0, 1], [2, 2], 'rhs');
  const bias = mkConst([1, 1, 1, 1], [2, 2], 'bias');
  const dense = mkCall('nn.dense', [lhs, rhs]);
  const biased = mkCall('nn.bias_add', [dense, bias]);
  const folded = constantFold(mkModule('f', [], biased, [2, 2])).getFunction('f')!.body;

  expect(folded.kind).toBe('constant');
});

test('unsupported constant op stays as CallExpr', () => {
  const c = mkConst([1], [1], 'c');
  const call = mkCall('nn.softmax', [c], [1]);
  const folded = constantFold(mkModule('f', [], call, [1])).getFunction('f')!.body;
  expect(folded.kind).toBe('call');
});


