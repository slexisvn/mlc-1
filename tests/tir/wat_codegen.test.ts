import { expect, test } from 'vitest';

import { lowerOp } from '../../src/lower/lowering.js';
import { analyzeWATKernel, codegenWAT } from '../../src/codegen/wat_codegen.js';

test('codegenWAT emits SIMD kernels for fused dense+bias+relu with output width divisible by 4', () => {
  const pf = lowerOp('fused.dense_bias_relu', [[4, 32], [64, 32], [1, 64]]);
  expect(pf).not.toBeNull();

  const kernel = analyzeWATKernel(pf!);
  const wat = codegenWAT([pf!]);

  expect(kernel.mode).toBe('simd-f32x4');
  expect(kernel.weightLayout).toBe('packed-j4k');
  expect(wat.text).toContain('v128.load');
  expect(wat.text).toContain('f32x4.splat');
  expect(wat.text).toContain('v128.store');
  expect(wat.text).toContain('f32x4.max');
});

test('codegenWAT falls back to scalar kernels when output width is not divisible by 4', () => {
  const pf = lowerOp('fused.dense_bias', [[4, 8], [3, 8], [1, 3]]);
  expect(pf).not.toBeNull();

  const kernel = analyzeWATKernel(pf!);
  const wat = codegenWAT([pf!]);

  expect(kernel.mode).toBe('scalar');
  expect(kernel.weightLayout).toBe('row-major');
  expect(wat.text).not.toContain('v128.load');
  expect(wat.text).not.toContain('f32x4.splat');
});
