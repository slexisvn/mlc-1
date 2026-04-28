import { expect, test } from 'vitest';
import { mkConst, mkModule, mkCall } from '../helpers/hlir_builders.js';
import {
  cosineSimilarity,
  measureQuantQuality,
  quantizeModule,
} from '../../src/transform/quantize.js';
import { LetExpr, TensorType, VarExpr } from '../../src/ir/high_level.js';

test('quantizeModule quantizes constants and reports counts', () => {
  const w = mkConst([1.0, -0.5, 0.25, -0.125], [2, 2], 'W');
  const b = mkConst([0.1, -0.2], [1, 2], 'B');
  const add = mkCall('add', [w, b], [2, 2]);
  const mod = mkModule('f', [], add, [2, 2]);

  const result = quantizeModule(mod);

  expect(result.quantizedCount).toBe(2);
  expect(result.totalParams).toBe(6);
  expect(result.quantizedWeights.length).toBe(2);
  expect(result.table.includes('Total params:')).toBe(true);
});

test('quantizeModule handles all-zero tensors with fallback scale', () => {
  const zeros = mkConst([0, 0, 0, 0], [2, 2], 'zeros');
  const mod = mkModule('f', [], zeros, [2, 2]);

  const result = quantizeModule(mod);
  const quant = result.quantizedWeights[0];

  expect(quant.quantParams.scale).toBe(1e-8);
  expect(quant.quantParams.absMax).toBe(0);
  expect(Array.from(quant.quantizedData)).toEqual([0, 0, 0, 0]);
  expect(Array.from(quant.dequantizedData)).toEqual([0, 0, 0, 0]);
});

test('quantizeModule rewrites ConstantExpr data in let-bound expressions', () => {
  const weight = mkConst([0.3, -0.7], [1, 2], 'W');
  const tmp = new VarExpr('_tmp', new TensorType([1, 2]));
  const body = new LetExpr(tmp, weight, tmp);
  const mod = mkModule('f', [], body, [1, 2]);
  const before = Array.from(weight.data.data);

  const result = quantizeModule(mod);
  const after = Array.from(weight.data.data);

  expect(result.quantizedCount).toBe(1);
  expect(after.some((v, i) => v !== before[i])).toBe(true);
});

test('cosineSimilarity returns 1.0 for zero vectors', () => {
  const sim = cosineSimilarity(new Float32Array([0, 0]), new Float32Array([0, 0]));
  expect(sim).toBe(1);
});

test('measureQuantQuality reports cosine and diff statistics', () => {
  const floatOut = new Float32Array([1.0, 2.0, 3.0]);
  const quantOut = new Float32Array([0.9, 2.1, 2.8]);
  const quality = measureQuantQuality(floatOut, quantOut);

  expect(quality.cosineSim).toBeGreaterThan(0.99);
  expect(Number(quality.maxAbsDiff.toFixed(3))).toBe(0.2);
  expect(Number(quality.meanAbsDiff.toFixed(3))).toBe(0.133);
});


