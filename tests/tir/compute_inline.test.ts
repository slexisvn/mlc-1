// ─── Unit tests: compute_inline pass ─────────────────────────────
import { expect, test } from 'vitest';
import { mkBiasAddProducer, mkReluConsumer } from '../helpers/tir_builders.js';
import { computeInline, demoComputeInline } from '../../src/transform/compute_inline.js';
import { BufferDecl, BufferStoreNode, ConstExpr, ConstIndex, PrimFunc } from '../../src/ir/low_level.js';

// ─── Test 1: successful inline — producer expression substituted ──
test('bias_add inlined into relu → substituted > 0', () => {
  const producer = mkBiasAddProducer(4, 8);  // writes to BiasOut
  const consumer = mkReluConsumer(4, 8);     // reads from BiasOut

  const result = computeInline(producer, consumer, 'BiasOut');

  expect(result.substituted).toBeGreaterThan(0);
  expect(result.producerName).toBe('bias_add');
  expect(result.consumerName).toBe('relu');
});

// ─── Test 2: merged func name is producer_inlined_into_consumer ───
test('inlined func name is bias_add_inlined_into_relu', () => {
  const producer = mkBiasAddProducer(4, 8);
  const consumer = mkReluConsumer(4, 8);

  const { inlined } = computeInline(producer, consumer, 'BiasOut');
  expect(inlined.name).toBe('bias_add_inlined_into_relu');
});

// ─── Test 3: merged params exclude the intermediate buffer ────────
test('inlined func params do not contain BiasOut', () => {
  const producer = mkBiasAddProducer(4, 8);
  const consumer = mkReluConsumer(4, 8);

  const { inlined } = computeInline(producer, consumer, 'BiasOut');
  const hasBiasOut = inlined.params.some(p => p.name === 'BiasOut');
  expect(hasBiasOut).toBe(false);
});

// ─── Test 4: merged params contain producer inputs A and B ────────
test('inlined func params contain producer inputs A and B', () => {
  const producer = mkBiasAddProducer(4, 8);
  const consumer = mkReluConsumer(4, 8);

  const { inlined } = computeInline(producer, consumer, 'BiasOut');
  const hasA = inlined.params.some(p => p.name === 'A');
  const hasB = inlined.params.some(p => p.name === 'B');
  expect(hasA).toBe(true);
  expect(hasB).toBe(true);
});

// ─── Test 5: wrong output buffer name → substituted=0, consumer unchanged ─
test('wrong output buffer name → substituted=0, returns consumer', () => {
  const producer = mkBiasAddProducer(4, 8);
  const consumer = mkReluConsumer(4, 8);

  const result = computeInline(producer, consumer, 'NonExistent');
  expect(result.substituted).toBe(0);
});

test('producer without matching output store returns consumer unchanged', () => {
  const a = new BufferDecl('A', [1], 'global');
  const out = new BufferDecl('Out', [1], 'global');
  const producer = new PrimFunc('producer', [a, out], new BufferStoreNode(out, [new ConstIndex(0)], new ConstExpr(1)));
  const consumer = mkReluConsumer(1, 1);
  const result = computeInline(producer, consumer, 'MissingBuffer');

  expect(result.substituted).toBe(0);
  expect(result.inlined.name).toBe(consumer.name);
});

test('demoComputeInline renders merged-kernel summary text', () => {
  const output = demoComputeInline(2, 3);
  expect(output.includes('After computeInline')).toBe(true);
  expect(output.includes('Intermediate buffer')).toBe(true);
});

