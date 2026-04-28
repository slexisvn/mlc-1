import { expect, test } from 'vitest';
import { buildIR } from '../../src/ir/high_level.js';
import { CrossEntropyLoss } from '../../src/loss/loss.js';
import { Linear, Sequential } from '../../src/model/nn.js';
import { lowerModule } from '../../src/lower/lowering.js';
import { Tracer } from '../../src/trace/tracer.js';
import { inferModuleShapes } from '../../src/transform/shape_infer.js';
import { verifyLowLevelIR } from '../../src/transform/verifier.js';

test('buildIR defaults to the forward root for inference graphs', () => {
  const model = new Sequential([new Linear(4, 3)]);
  const graph = new Tracer().traceInference(model, [2, 4]);
  const irModule = buildIR(graph);
  const main = irModule.getFunction('main');

  expect(main).toBeDefined();
  expect(main!.retType.shape).toEqual([2, 3]);
  expect(main!.body.kind).toBe('call');
  if (main!.body.kind === 'call') {
    expect(main!.body.op.name).toBe('nn.bias_add');
  }
});

test('buildIR can root a training trace at the scalar loss', () => {
  const model = new Sequential([new Linear(4, 3)]);
  const loss = new CrossEntropyLoss();
  const graph = new Tracer().traceTraining(model, loss, [2, 4], [2]);
  const irModule = buildIR(graph, { root: 'loss' });
  const main = irModule.getFunction('main');

  expect(main).toBeDefined();
  expect(main!.params.map(p => p.name)).toEqual(['x', 'target']);
  expect(main!.params[1].type.shape).toEqual([2]);
  expect(main!.retType.shape).toEqual([1]);
  expect(main!.body.kind).toBe('call');
  if (main!.body.kind === 'call') {
    expect(main!.body.op.name).toBe('cross_entropy');
  }
});

test('buildIR rejects loss mode when the graph has no loss root', () => {
  const model = new Sequential([new Linear(4, 3)]);
  const graph = new Tracer().traceInference(model, [2, 4]);

  expect(() => buildIR(graph, { root: 'loss' })).toThrow('missing loss root');
});

test('traceTraining materializes backward nodes and gradient roots', () => {
  const model = new Sequential([new Linear(4, 3)]);
  const loss = new CrossEntropyLoss();
  const graph = new Tracer().traceTraining(model, loss, [2, 4], [2]);
  const backwardOps = graph.nodes.slice(graph.backwardStartIdx).map(node => node.op);
  const firstParam = graph.params.get('param_0');

  expect(backwardOps).toContain('nn.cross_entropy_grad');
  expect(backwardOps).toContain('nn.bias_add_grad');
  expect(backwardOps).toContain('nn.dense_grad_weight');
  expect(firstParam).toBeDefined();
  expect(graph.gradientIds.has(firstParam!.tensor.id)).toBe(true);
});

test('buildIR can root directly at a parameter gradient tensor', () => {
  const model = new Sequential([new Linear(4, 3)]);
  const loss = new CrossEntropyLoss();
  const graph = new Tracer().traceTraining(model, loss, [2, 4], [2]);
  const weight = graph.params.get('param_0');
  if (!weight) throw new Error('expected first parameter');

  const gradId = graph.gradientIds.get(weight.tensor.id);
  if (gradId === undefined) throw new Error('expected gradient root for first parameter');

  const irModule = buildIR(graph, { rootId: gradId });
  const main = irModule.getFunction('main');

  expect(main).toBeDefined();
  expect(main!.body.kind).toBe('call');
  if (main!.body.kind === 'call') {
    expect(main!.body.op.name).toBe('nn.dense_grad_weight');
    expect(main!.retType.shape).toEqual([3, 4]);
  }
});

test('inferModuleShapes infers backward gradient shapes for a parameter root', () => {
  const model = new Sequential([new Linear(4, 3)]);
  const loss = new CrossEntropyLoss();
  const graph = new Tracer().traceTraining(model, loss, [2, 4], [2]);
  const weight = graph.params.get('param_0');
  if (!weight) throw new Error('expected first parameter');

  const gradId = graph.gradientIds.get(weight.tensor.id);
  if (gradId === undefined) throw new Error('expected gradient root for first parameter');

  const irModule = buildIR(graph, { rootId: gradId });
  const result = inferModuleShapes(irModule);
  const main = irModule.getFunction('main');

  expect(result.inferred).toBeGreaterThanOrEqual(2);
  expect(main).toBeDefined();
  expect(main!.body.kind).toBe('call');
  if (main!.body.kind === 'call') {
    expect(main!.body.attrs.outputShape).toEqual([3, 4]);
    expect(main!.body.args[1].kind).toBe('call');
    if (main!.body.args[1].kind === 'call') {
      expect(main!.body.args[1].attrs.outputShape).toEqual([2, 3]);
    }
  }
});

test('lowerModule lowers a backward gradient root into backward kernels', () => {
  const model = new Sequential([new Linear(4, 3)]);
  const loss = new CrossEntropyLoss();
  const graph = new Tracer().traceTraining(model, loss, [2, 4], [2]);
  const weight = graph.params.get('param_0');
  if (!weight) throw new Error('expected first parameter');

  const gradId = graph.gradientIds.get(weight.tensor.id);
  if (gradId === undefined) throw new Error('expected gradient root for first parameter');

  const irModule = buildIR(graph, { rootId: gradId });
  inferModuleShapes(irModule);
  const primFuncs = lowerModule(irModule);
  const verify = verifyLowLevelIR(primFuncs);
  const kernelNames = primFuncs.map(pf => pf.name);

  expect(kernelNames).toContain('softmax_ce_grad');
  expect(kernelNames).toContain('dense_grad_weight');
  expect(verify.ok).toBe(true);
});