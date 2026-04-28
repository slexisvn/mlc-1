// ─── Unit tests: op_fusion pass ──────────────────────────────────
import { expect, test } from 'vitest';
import { assertHLIRValid } from '../helpers/verify.js';
import {
  mkVar, mkConst, mkCall, mkModule,
  mkDenseBiasReluModule,
  mkDenseBiasModule,
  mkMultiConsumerModule,
} from '../helpers/hlir_builders.js';
import { fuseOps, fusionStats } from '../../src/transform/op_fusion.js';

// ─── Test 1: dense+bias_add+relu → fused.dense_bias_relu ─────────
test('dense→bias_add→relu chain fuses to fused.dense_bias_relu', () => {
  const { mod } = mkDenseBiasReluModule(4, 16, 8);
  const fused = fuseOps(mod);
  const body = fused.getFunction('forward')!.body;
  // After fusion the body should be a single fused call
  expect(body.kind).toBe('call');
  if (body.kind === 'call') {
    expect(body.op.name).toBe('fused.dense_bias_relu');
  }
});

// ─── Test 2: dense+bias_add (no activation) → fused.dense_bias ───
test('dense→bias_add chain fuses to fused.dense_bias', () => {
  const { mod } = mkDenseBiasModule(4, 16, 8);
  const fused = fuseOps(mod);
  const body = fused.getFunction('forward')!.body;
  expect(body.kind).toBe('call');
  if (body.kind === 'call') {
    expect(body.op.name).toBe('fused.dense_bias');
  }
});

// ─── Test 3: multi-consumer guard → dense is NOT fused ───────────
test('dense with 2 consumers is NOT fused (multi-consumer guard)', () => {
  const { mod } = mkMultiConsumerModule(4, 8, 4);
  const fused = fuseOps(mod);
  const stats = fusionStats(fused);
  // No full dense→bias_add chain can be fused when dense has 2 consumers
  expect(stats.fusedOps).toBe(0);
});

// ─── Test 4: fusionStats reports correct counts ───────────────────
test('fusionStats returns correct fusedOps for a fusible chain', () => {
  const { mod } = mkDenseBiasReluModule(4, 16, 8);
  const fused = fuseOps(mod);
  const stats = fusionStats(fused);
  expect(stats.fusedOps).toBeGreaterThanOrEqual(1);
  expect(stats.fusedGroups.length).toBeGreaterThanOrEqual(1);
});

// ─── Test 5: standalone (unfused) op stays as-is ─────────────────
test('standalone nn.relu (no fusion pattern) stays as-is', () => {
  const x = mkVar('x', [4, 8]);
  const c = mkConst(new Array(4 * 8).fill(0.5), [4, 8], 'c');
  // relu(const) — no dense before it, cannot match any fusion pattern
  const relu = mkCall('nn.relu', [c], [4, 8]);
  const mod  = mkModule('f', [], relu, [4, 8]);

  const fused = fuseOps(mod);
  const body  = fused.getFunction('f')!.body;
  expect(body.kind).toBe('call');
  if (body.kind === 'call') {
    expect(body.op.name).toBe('nn.relu');
  }
});

// ─── Test 6: verifier passes after fusion ────────────────────────
test('verifier passes after fuseOps', () => {
  const { mod } = mkDenseBiasReluModule(4, 16, 8);
  const fused = fuseOps(mod);
  assertHLIRValid(fused, 'after fuseOps');
});

