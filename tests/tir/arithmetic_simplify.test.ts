// ─── Unit tests: arithmetic_simplify pass ────────────────────────
// Tests the exported simplifyIndex / simplifyValue directly
// (pure expression rewriters, no PrimFunc traversal needed).
import { expect, test } from 'vitest';
import {
  simplifyIndex,
  simplifyValue,
  arithmeticSimplify,
} from '../../src/transform/arithmetic_simplify.js';
import {
  LoopVar,
  VarIndex, ConstIndex, BinOpIndex,
  BinOpExpr, ConstExpr, VarRefExpr, CallExprTIR, MaxExpr, MinExpr,
} from '../../src/ir/low_level.js';
import { mkDenseWithAccFunc } from '../helpers/tir_builders.js';

const i = new LoopVar('i', 'spatial');

// ─── Index simplifications ────────────────────────────────────────

test('index: i + 0 → VarIndex(i)', () => {
  const expr = new BinOpIndex('+', new VarIndex(i), new ConstIndex(0));
  const result = simplifyIndex(expr);
  expect(result instanceof VarIndex).toBe(true);
  expect((result as VarIndex).loopVar.name).toBe('i');
});

test('index: 0 + i → VarIndex(i)', () => {
  const expr = new BinOpIndex('+', new ConstIndex(0), new VarIndex(i));
  const result = simplifyIndex(expr);
  expect(result instanceof VarIndex).toBe(true);
});

test('index: i * 1 → VarIndex(i)', () => {
  const expr = new BinOpIndex('*', new VarIndex(i), new ConstIndex(1));
  const result = simplifyIndex(expr);
  expect(result instanceof VarIndex).toBe(true);
});

test('index: i - 0 → VarIndex(i)', () => {
  const expr = new BinOpIndex('-', new VarIndex(i), new ConstIndex(0));
  const result = simplifyIndex(expr);
  expect(result instanceof VarIndex).toBe(true);
});

test('index: constant folding 3 + 4 → ConstIndex(7)', () => {
  const expr = new BinOpIndex('+', new ConstIndex(3), new ConstIndex(4));
  const result = simplifyIndex(expr);
  expect(result instanceof ConstIndex).toBe(true);
  expect((result as ConstIndex).value).toBe(7);
});

test('index: i * 0 → ConstIndex(0)', () => {
  const expr = new BinOpIndex('*', new VarIndex(i), new ConstIndex(0));
  const result = simplifyIndex(expr);
  expect(result instanceof ConstIndex).toBe(true);
  expect((result as ConstIndex).value).toBe(0);
});

// ─── Value simplifications ────────────────────────────────────────

test('value: v + 0 → same expression (identity)', () => {
  const v = new VarRefExpr(i);
  const expr = new BinOpExpr('+', v, new ConstExpr(0));
  const result = simplifyValue(expr);
  expect(result instanceof VarRefExpr).toBe(true);
});

test('value: v * 1 → same expression (identity)', () => {
  const v = new VarRefExpr(i);
  const expr = new BinOpExpr('*', v, new ConstExpr(1));
  const result = simplifyValue(expr);
  expect(result instanceof VarRefExpr).toBe(true);
});

test('value: v * 0 → ConstExpr(0)', () => {
  const v = new VarRefExpr(i);
  const expr = new BinOpExpr('*', v, new ConstExpr(0));
  const result = simplifyValue(expr);
  expect(result instanceof ConstExpr).toBe(true);
  expect((result as ConstExpr).value).toBe(0);
});

test('value: constant folding 3 * 4 → ConstExpr(12)', () => {
  const expr = new BinOpExpr('*', new ConstExpr(3), new ConstExpr(4));
  const result = simplifyValue(expr);
  expect(result instanceof ConstExpr).toBe(true);
  expect((result as ConstExpr).value).toBe(12);
});

test('value: v * 2 → BinOpExpr(+, v, v) (strength reduction)', () => {
  const v = new VarRefExpr(i);
  const expr = new BinOpExpr('*', v, new ConstExpr(2));
  const result = simplifyValue(expr);
  expect(result instanceof BinOpExpr).toBe(true);
  expect((result as BinOpExpr).op).toBe('+');
});

test('value: Math.sqrt(const) folds to ConstExpr', () => {
  const result = simplifyValue(new CallExprTIR('Math.sqrt', [new ConstExpr(9)]));
  expect(result instanceof ConstExpr).toBe(true);
  expect((result as ConstExpr).value).toBe(3);
});

test('value: max/min on constants fold to ConstExpr', () => {
  const maxResult = simplifyValue(new MaxExpr(new ConstExpr(4), new ConstExpr(7)));
  const minResult = simplifyValue(new MinExpr(new ConstExpr(4), new ConstExpr(7)));
  expect((maxResult as ConstExpr).value).toBe(7);
  expect((minResult as ConstExpr).value).toBe(4);
});

test('arithmeticSimplify rewrites expressions through a full PrimFunc tree', () => {
  const func = mkDenseWithAccFunc(2, 2, 2);
  const simplified = arithmeticSimplify(func);
  expect(simplified !== func).toBe(true);
  expect(simplified.name).toBe(func.name);
});

