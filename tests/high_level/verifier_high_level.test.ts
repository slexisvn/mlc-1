import { expect, test } from 'vitest';
import { mkConst, mkModule, mkVar } from '../helpers/hlir_builders.js';
import {
  printVerifyResult,
  verifyHighLevelIR,
} from '../../src/transform/verifier.js';
import { IRFunction, IRModule, TensorType } from '../../src/ir/high_level.js';

test('verifyHighLevelIR accepts bound params and constants', () => {
  const x = mkVar('x', [2, 2]);
  const mod = mkModule('f', [x], x, [2, 2]);
  const result = verifyHighLevelIR(mod);
  expect(result.ok).toBe(true);
  expect(result.errors.length).toBe(0);
});

test('verifyHighLevelIR reports unbound variables', () => {
  const y = mkVar('y', [2, 2]);
  const mod = mkModule('f', [], y, [2, 2]);
  const result = verifyHighLevelIR(mod);

  expect(result.ok).toBe(false);
  expect(result.errors.some(e => e.includes("Unbound variable: 'y'"))).toBe(true);
});

test('verifyHighLevelIR reports constants with empty shape', () => {
  const scalarLike = mkConst([1], [], 'bad_const');
  const mod = mkModule('f', [], scalarLike, []);
  const result = verifyHighLevelIR(mod);

  expect(result.ok).toBe(false);
  expect(result.errors.some(e => e.includes("ConstantExpr 'bad_const' has empty shape"))).toBe(true);
});

test('verifyHighLevelIR warns on empty return type shape', () => {
  const mod = new IRModule();
  mod.addFunction(new IRFunction('f', [], mkConst([1], [1], 'c'), new TensorType([])));
  const result = verifyHighLevelIR(mod);

  expect(result.ok).toBe(true);
  expect(result.warnings.some(w => w.includes("Function 'f' has empty return type shape"))).toBe(true);
});

test('printVerifyResult renders OK status line', () => {
  const mod = mkModule('f', [], mkConst([1], [1], 'c'), [1]);
  const rendered = printVerifyResult('hlir', verifyHighLevelIR(mod));

  expect(rendered.includes('IR Verify [hlir]')).toBe(true);
  expect(rendered.includes('OK')).toBe(true);
});

