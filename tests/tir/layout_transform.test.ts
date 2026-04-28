// ─── Unit tests: layout_transform pass ───────────────────────────
import { expect, test } from 'vitest';
import { mkDenseWithAccFunc } from '../helpers/tir_builders.js';
import { layoutTransform } from '../../src/transform/layout_transform.js';

// ─── Test 1: W[32,16] with blockN=16 → applied=true ──────────────
test('W[32,16] blockN=16 (32%16=0 and 32>=16) → applied=true', () => {
  const func = mkDenseWithAccFunc(4, 32, 16);  // W has shape [32, 16]
  const { stats } = layoutTransform(func, 16);
  expect(stats.applied).toBe(true);
});

// ─── Test 2: stats are populated correctly ────────────────────────
test('applied transform: packedShape and packingOps are populated', () => {
  const func = mkDenseWithAccFunc(4, 32, 16);  // W[32,16]
  const { stats } = layoutTransform(func, 16);

  expect(stats.applied).toBe(true);
  expect(stats.packingOps).toBeGreaterThan(0);
  expect(stats.rewrittenLoads).toBeGreaterThan(0);
  expect(stats.blockN).toBe(16);
  // originalShape contains N=32; packedShape = [N/blockN, K, blockN] = [2, K, 16]
  expect(stats.originalShape.includes('32')).toBe(true);
});

// ─── Test 3: N not divisible by blockN → applied=false ───────────
test('W[30,16] blockN=16 (30%16≠0) → applied=false', () => {
  const func = mkDenseWithAccFunc(4, 30, 16);  // W[30,16]
  const { stats } = layoutTransform(func, 16);
  expect(stats.applied).toBe(false);
});

// ─── Test 4: N < blockN → applied=false ──────────────────────────
test('W[8,16] blockN=16 (N < blockN) → applied=false', () => {
  const func = mkDenseWithAccFunc(4, 8, 16);  // W[8,16]
  const { stats } = layoutTransform(func, 16);
  expect(stats.applied).toBe(false);
});

// ─── Test 5: transformed PrimFunc has W_packed buffer ─────────────
test('transformed func adds W_packed as a local allocation', () => {
  const func = mkDenseWithAccFunc(4, 32, 16);  // W[32,16], blockN=16
  const { transformed } = layoutTransform(func, 16);
  const hasWPacked = transformed.allocations.some(a => a.name === 'W_packed');
  expect(hasWPacked).toBe(true);
});

