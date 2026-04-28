// ─── Unit tests: shape_infer pass ────────────────────────────────
import { expect, test } from 'vitest';
import { mkVar, mkConst, mkCall, mkModule } from '../helpers/hlir_builders.js';
import { inferModuleShapes } from '../../src/transform/shape_infer.js';

// ─── Test 1: nn.dense shape inference [M,K] × [N,K] → [M,N] ─────
test('nn.dense: [4,32] × [64,32] → outputShape=[4,64]', () => {
  const x = mkVar('x', [4, 32]);
  // W has shape [N, K] = [64, 32], but we just pass as a const with shape [64,32]
  const W = mkConst(new Array(64 * 32).fill(0.1), [64, 32], 'W');
  const denseCall = mkCall('nn.dense', [x, W]);  // no shape set yet
  const mod = mkModule('f', [x], denseCall, [4, 64]);

  const result = inferModuleShapes(mod);

  expect(denseCall.attrs.outputShape).toEqual([4, 64]);
  expect(result.inferred).toBe(1);
  expect(result.totalOps).toBe(1);
});

// ─── Test 2: nn.relu preserves shape ─────────────────────────────
test('nn.relu: [4,64] → outputShape=[4,64] (shape preserved)', () => {
  const x = mkVar('x', [4, 64]);
  const reluCall = mkCall('nn.relu', [x]);
  const mod = mkModule('f', [x], reluCall, [4, 64]);

  inferModuleShapes(mod);
  expect(reluCall.attrs.outputShape).toEqual([4, 64]);
});

// ─── Test 3: nn.bias_add preserves shape ─────────────────────────
test('nn.bias_add: [4,64] → outputShape=[4,64] (shape preserved)', () => {
  const x = mkVar('x', [4, 64]);
  const B = mkConst(new Array(64).fill(0), [1, 64], 'B');
  const biasCall = mkCall('nn.bias_add', [x, B]);
  const mod = mkModule('f', [x], biasCall, [4, 64]);

  inferModuleShapes(mod);
  expect(biasCall.attrs.outputShape).toEqual([4, 64]);
});

// ─── Test 4: reduce_sum collapses to scalar [1] ───────────────────
test('reduce_sum: [4,64] → outputShape=[1] (reduction to scalar)', () => {
  const x = mkVar('x', [4, 64]);
  const sumCall = mkCall('reduce_sum', [x]);
  const mod = mkModule('f', [x], sumCall, [1]);

  inferModuleShapes(mod);
  expect(sumCall.attrs.outputShape).toEqual([1]);
});

// ─── Test 5: chain inference — all ops get shapes ─────────────────
test('chain dense→bias_add→relu: all 3 ops get outputShape', () => {
  const x = mkVar('x', [4, 32]);
  const W = mkConst(new Array(64 * 32).fill(0.1), [64, 32], 'W');
  const B = mkConst(new Array(64).fill(0), [1, 64], 'B');
  const dense   = mkCall('nn.dense',    [x, W]);
  const biasAdd = mkCall('nn.bias_add', [dense, B]);
  const relu    = mkCall('nn.relu',     [biasAdd]);
  const mod     = mkModule('f', [x], relu, [4, 64]);

  const result = inferModuleShapes(mod);

  expect(dense.attrs.outputShape).toEqual([4, 64]);
  expect(biasAdd.attrs.outputShape).toEqual([4, 64]);
  expect(relu.attrs.outputShape).toEqual([4, 64]);
  expect(result.inferred).toBe(3);
  expect(result.totalOps).toBe(3);
});

// ─── Test 6: fused.dense_bias_relu shape ─────────────────────────
test('fused.dense_bias_relu: [4,32] + [64,32] → [4,64]', () => {
  const x = mkVar('x', [4, 32]);
  const W = mkConst(new Array(64 * 32).fill(0.1), [64, 32], 'W');
  const B = mkConst(new Array(64).fill(0), [1, 64], 'B');
  const fused = mkCall('fused.dense_bias_relu', [x, W, B]);
  const mod   = mkModule('f', [x], fused, [4, 64]);

  inferModuleShapes(mod);
  expect(fused.attrs.outputShape).toEqual([4, 64]);
});

test('multiply broadcasts to longer shape', () => {
  const x = mkVar('x', [4, 64]);
  const y = mkConst(new Array(64).fill(1), [64], 'y');
  const mul = mkCall('multiply', [x, y]);
  inferModuleShapes(mkModule('f', [x], mul, [4, 64]));
  expect(mul.attrs.outputShape).toEqual([4, 64]);
});

test('loss ops infer scalar [1]', () => {
  const pred = mkVar('pred', [4, 8]);
  const label = mkVar('label', [4, 8]);
  const ce = mkCall('cross_entropy', [pred, label]);
  const mse = mkCall('mse', [pred, label]);

  inferModuleShapes(mkModule('ce', [pred, label], ce, [1]));
  inferModuleShapes(mkModule('mse', [pred, label], mse, [1]));

  expect(ce.attrs.outputShape).toEqual([1]);
  expect(mse.attrs.outputShape).toEqual([1]);
});

test('unknown op falls back to pre-filled attrs.outputShape', () => {
  const x = mkVar('x', [2, 2]);
  const custom = mkCall('custom.op', [x], [9, 9]);
  const result = inferModuleShapes(mkModule('f', [x], custom, [9, 9]));

  expect(custom.attrs.outputShape).toEqual([9, 9]);
  expect(result.table.includes('custom.op')).toBe(true);
});

