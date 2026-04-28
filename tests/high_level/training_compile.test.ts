import { expect, test } from 'vitest';
import { compileTrainingArtifacts } from '../../src/compile/decorator.js';
import { CrossEntropyLoss } from '../../src/loss/loss.js';
import { Linear, ReLU, Sequential } from '../../src/model/nn.js';
import { verifyLowLevelIR } from '../../src/transform/verifier.js';

test('compileTrainingArtifacts produces backward kernels for a 2-layer ReLU classifier', () => {
  const model = new Sequential([
    new Linear(4, 5),
    new ReLU(),
    new Linear(5, 3),
  ]);
  const loss = new CrossEntropyLoss();

  const artifacts = compileTrainingArtifacts(model, loss, [2, 4], [2]);
  const w1 = artifacts.paramGradKernels.get('param_0');
  const b1 = artifacts.paramGradKernels.get('param_1');
  const w2 = artifacts.paramGradKernels.get('param_2');
  const b2 = artifacts.paramGradKernels.get('param_3');

  expect(artifacts.forwardKernels.length).toBeGreaterThan(0);
  expect(artifacts.lossKernels.map(pf => pf.name)).toContain('fused_softmax_ce');
  expect(w1?.map(pf => pf.name)).toContain('dense_grad_weight');
  expect(w1?.map(pf => pf.name)).toContain('relu_grad');
  expect(w1?.map(pf => pf.name)).toContain('dense_grad_data');
  expect(b1?.map(pf => pf.name)).toContain('bias_add_grad');
  expect(w2?.map(pf => pf.name)).toContain('dense_grad_weight');
  expect(b2?.map(pf => pf.name)).toContain('bias_add_grad');
  expect(verifyLowLevelIR(w1 ?? []).ok).toBe(true);
  expect(verifyLowLevelIR(b1 ?? []).ok).toBe(true);
  expect(verifyLowLevelIR(w2 ?? []).ok).toBe(true);
  expect(verifyLowLevelIR(b2 ?? []).ok).toBe(true);
});