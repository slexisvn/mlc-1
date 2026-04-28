import { expect, test } from 'vitest';
import { mkDenseWithAccFunc } from '../helpers/tir_builders.js';
import {
  printVerifyResult,
  verifyLowLevelIR,
} from '../../src/transform/verifier.js';
import {
  AllocNode,
  BufferDecl,
  BufferLoadExpr,
  BufferStoreNode,
  ConstExpr,
  ConstIndex,
  ForNode,
  LoopVar,
  PrimFunc,
} from '../../src/ir/low_level.js';

test('verifyLowLevelIR accepts a valid PrimFunc', () => {
  const func = mkDenseWithAccFunc(2, 4, 2);
  const result = verifyLowLevelIR([func]);
  expect(result.ok).toBe(true);
  expect(result.errors.length).toBe(0);
});

test('verifyLowLevelIR reports non-positive loop extents', () => {
  const out = new BufferDecl('Out', [1], 'global');
  const i = new LoopVar('i', 'spatial');
  const func = new PrimFunc(
    'bad_extent',
    [out],
    new ForNode(i, 0, 0, new BufferStoreNode(out, [new ConstIndex(0)], new ConstExpr(0)), 'none')
  );
  const result = verifyLowLevelIR([func]);

  expect(result.ok).toBe(false);
  expect(result.errors.some(e => e.includes("ForNode 'i' has non-positive extent 0"))).toBe(true);
});

test('verifyLowLevelIR reports undeclared loads and store rank mismatch', () => {
  const out = new BufferDecl('Out', [2, 2], 'global');
  const ghost = new BufferDecl('Ghost', [2, 2], 'global');
  const func = new PrimFunc(
    'bad_buffers',
    [out],
    new BufferStoreNode(
      out,
      [new ConstIndex(0)],
      new BufferLoadExpr(ghost, [new ConstIndex(0), new ConstIndex(1)])
    )
  );
  const result = verifyLowLevelIR([func]);

  expect(result.ok).toBe(false);
  expect(result.errors.some(e => e.includes("BufferStore 'Out' index rank 1 does not match buffer rank 2"))).toBe(true);
  expect(result.errors.some(e => e.includes("BufferLoad references undeclared buffer 'Ghost'"))).toBe(true);
});

test('verifyLowLevelIR warns on unused local allocations', () => {
  const out = new BufferDecl('Out', [1], 'global');
  const scratch = new BufferDecl('scratch', [1], 'local');
  const func = new PrimFunc(
    'unused_alloc',
    [out],
    new AllocNode(scratch, new BufferStoreNode(out, [new ConstIndex(0)], new ConstExpr(1))),
    [scratch]
  );
  const result = verifyLowLevelIR([func]);

  expect(result.ok).toBe(true);
  expect(result.warnings.some(w => w.includes("local allocation 'scratch' is never used"))).toBe(true);
});

test('printVerifyResult renders FAILED output with error and warning lines', () => {
  const out = new BufferDecl('Out', [1], 'global');
  const scratch = new BufferDecl('scratch', [1], 'local');
  const func = new PrimFunc(
    'mixed',
    [out],
    new AllocNode(
      scratch,
      new ForNode(new LoopVar('i', 'spatial'), 0, 0, new BufferStoreNode(out, [new ConstIndex(0)], new ConstExpr(0)), 'none')
    ),
    [scratch]
  );
  const rendered = printVerifyResult('tir', verifyLowLevelIR([func]));

  expect(rendered.includes('FAILED')).toBe(true);
  expect(rendered.includes('ERROR:')).toBe(true);
  expect(rendered.includes('WARN:')).toBe(true);
});

