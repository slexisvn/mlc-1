import { expect, test } from 'vitest';
import { buildIR } from '../../src/ir/high_level.js';
import { Linear, Sequential } from '../../src/model/nn.js';
import { Tracer } from '../../src/trace/tracer.js';

test('buildIR builds the forward root for inference graphs', () => {
  const model = new Sequential([new Linear(4, 3)]);
  const graph = new Tracer().traceInference(model, [2, 4]);
  const irModule = buildIR(graph);
  const main = irModule.getFunction('main');

  expect(main).toBeDefined();
  expect(main!.params.map(p => p.name)).toEqual(['x']);
  expect(main!.retType.shape).toEqual([2, 3]);
  expect(main!.body.kind).toBe('call');
  if (main!.body.kind === 'call') {
    expect(main!.body.op.name).toBe('nn.bias_add');
    expect(main!.body.attrs.outputShape).toEqual([2, 3]);
  }
});
