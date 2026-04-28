import { expect, test } from 'vitest';
import { mkConst, mkCall, mkModule, mkVar } from '../helpers/hlir_builders.js';
import { lowerModule, lowerOp } from '../../src/lower/lowering.js';

test('lowerOp(nn.dense) returns dense PrimFunc with accumulator alloc', () => {
  const pf = lowerOp('nn.dense', [[4, 32], [64, 32]]);
  expect(pf !== null).toBe(true);
  expect(pf!.name).toBe('dense');
  expect(pf!.params.length).toBe(3);
  expect(pf!.allocations.length).toBe(1);
  expect(pf!.allocations[0].name).toBe('acc');
  expect(pf!.getLoops().map(l => l.loopVar.name)).toEqual(['i', 'j', 'k']);
});

test('lowerOp lowers fused.dense_bias_relu and optim.sgd_update', () => {
  const fused = lowerOp('fused.dense_bias_relu', [[4, 32], [64, 32], [1, 64]]);
  const sgd = lowerOp('optim.sgd_update', [[8], [8], [1]]);

  expect(fused?.name).toBe('fused_dense_bias_relu');
  expect(fused?.params.length).toBe(4);
  expect(sgd?.name).toBe('sgd_update');
  expect(sgd?.params.length).toBe(3);
});

test('lowerOp returns null for unsupported op', () => {
  const pf = lowerOp('nn.relu', [[4, 64]]);
  expect(pf).toBeNull();
});

test('lowerModule lowers duplicate dense calls and uniquifies names', () => {
  const x = mkVar('x', [4, 8]);
  const w1 = mkConst(new Array(16 * 8).fill(0.1), [16, 8], 'w1');
  const w2 = mkConst(new Array(16 * 8).fill(0.2), [16, 8], 'w2');
  const d1 = mkCall('nn.dense', [x, w1], [4, 16]);
  const d2 = mkCall('nn.dense', [x, w2], [4, 16]);
  const add = mkCall('add', [d1, d2], [4, 16]);
  const funcs = lowerModule(mkModule('f', [x], add, [4, 16]));

  expect(funcs.length).toBe(2);
  expect(funcs[0].name).toBe('dense');
  expect(funcs[1].name).toBe('dense_1');
});

test('lowerModule falls back to default classifier kernels when nothing lowers', () => {
  const x = mkVar('x', [4, 64]);
  const relu = mkCall('nn.relu', [x], [4, 64]);
  const funcs = lowerModule(mkModule('f', [x], relu, [4, 64]));

  expect(funcs.length).toBe(2);
  expect(funcs[0].name).toBe('fused_dense_bias_relu');
  expect(funcs[1].name).toBe('fused_dense_bias');
});

