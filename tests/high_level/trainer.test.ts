import { expect, test } from 'vitest';

import { engine } from '../../src/autograd/engine.js';
import { CrossEntropyLoss } from '../../src/loss/loss.js';
import { Linear, Module, ReLU } from '../../src/model/nn.js';
import { SGD } from '../../src/optim/sgd.js';
import { Tensor, tensor } from '../../src/tensor/tensor.js';
import { LightningModule, Trainer } from '../../src/training/index.js';

class Classifier extends Module {
  readonly fc1 = new Linear(4, 8);
  readonly relu = new ReLU();
  readonly fc2 = new Linear(8, 3);

  override forward(x: Tensor): Tensor {
    return this.fc2.forward(this.relu.forward(this.fc1.forward(x)));
  }

  override parameters(): Tensor[] {
    return [...this.fc1.parameters(), ...this.fc2.parameters()];
  }

  override children(): Module[] {
    return [this.fc1, this.relu, this.fc2];
  }
}

class LightningClassifier extends LightningModule {
  readonly fc1 = new Linear(4, 8);
  readonly relu = new ReLU();
  readonly fc2 = new Linear(8, 3);
  readonly lossFn = new CrossEntropyLoss();

  override forward(x: Tensor): Tensor {
    return this.fc2.forward(this.relu.forward(this.fc1.forward(x)));
  }

  override parameters(): Tensor[] {
    return [...this.fc1.parameters(), ...this.fc2.parameters()];
  }

  override children(): Module[] {
    return [this.fc1, this.relu, this.fc2];
  }

  override trainingStep(batch: { input: Tensor; target: Tensor }): Tensor {
    return this.lossFn.forward(this.forward(batch.input), batch.target);
  }

  override validationStep(batch: { input: Tensor; target: Tensor }): Tensor {
    return this.lossFn.forward(this.forward(batch.input), batch.target);
  }

  override configureOptimizers(): SGD {
    return new SGD(this.parameters(), { lr: 0.1 });
  }
}

function makeBatch(inputs: number[][], targets: number[]): { input: Tensor; target: Tensor } {
  return {
    input: tensor(inputs),
    target: tensor(targets),
  };
}

test('Trainer eager loop reduces loss and updates parameters', () => {
  const model = new Classifier();
  const optimizer = new SGD(model.parameters(), { lr: 0.1 });
  const lossFn = new CrossEntropyLoss();
  const trainer = new Trainer({ model, optimizer, lossFn, maxSteps: 10 });
  const batch = makeBatch(
    [
      [3, 0, 0, 0],
      [0, 3, 0, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 3],
    ],
    [0, 1, 2, 1],
  );

  const initialParams = model.parameters().map(param => new Float32Array(param.data));
  const losses: number[] = [];

  for (let step = 0; step < 10; step++) {
    const loss = trainer.trainingStep(batch);
    losses.push(loss.data[0]);
  }

  expect(losses.at(-1)).toBeLessThan(losses[0]);
  expect(
    model.parameters().some((param, index) =>
      Array.from(param.data).some((value, i) => value !== initialParams[index][i])
    )
  ).toBe(true);
});

test('Trainer supports LightningModule setup for compact training code', () => {
  const module = new LightningClassifier();
  const trainer = new Trainer({ module, maxSteps: 6 });
  const batch = makeBatch(
    [
      [3, 0, 0, 0],
      [0, 3, 0, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 3],
    ],
    [0, 1, 2, 1],
  );

  const initialLoss = module.lossFn.forward(module.forward(batch.input), batch.target).data[0];
  trainer.fit(Array.from({ length: 6 }, () => batch));
  const finalLoss = module.lossFn.forward(module.forward(batch.input), batch.target).data[0];

  expect(finalLoss).toBeLessThan(initialLoss);
});

test('Trainer fit supports a separate validation pass', () => {
  const module = new LightningClassifier();
  const trainer = new Trainer({ module, maxSteps: 4 });
  const trainBatch = makeBatch(
    [
      [3, 0, 0, 0],
      [0, 3, 0, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 3],
    ],
    [0, 1, 2, 1],
  );
  const valBatch = makeBatch(
    [
      [2, 0, 1, 0],
      [0, 2, 0, 1],
      [1, 0, 2, 0],
      [0, 1, 0, 2],
    ],
    [0, 1, 2, 1],
  );

  trainer.fit(Array.from({ length: 4 }, () => trainBatch), [valBatch]);
  const valLoss = trainer.validationStep(valBatch);

  expect(module.training).toBe(false);
  expect(valLoss.data[0]).toBeGreaterThan(0);
});

test('Trainer train/eval propagates module lifecycle', () => {
  const model = new Classifier();
  const trainer = new Trainer({
    model,
    optimizer: new SGD(model.parameters(), { lr: 0.01 }),
    lossFn: new CrossEntropyLoss(),
  });

  trainer.eval();
  expect(model.training).toBe(false);
  expect(model.fc1.training).toBe(false);
  expect(model.relu.training).toBe(false);
  expect(model.fc2.training).toBe(false);

  trainer.train();
  expect(model.training).toBe(true);
  expect(model.fc1.training).toBe(true);
  expect(model.relu.training).toBe(true);
  expect(model.fc2.training).toBe(true);
});

test('Trainer compiles trained model for post-training inference', () => {
  const model = new Classifier();
  const optimizer = new SGD(model.parameters(), { lr: 0.05 });
  const trainer = new Trainer({
    model,
    optimizer,
    lossFn: new CrossEntropyLoss(),
    maxSteps: 6,
  });
  const batch = makeBatch(
    [
      [1, 0, 0, 1],
      [0, 1, 1, 0],
      [1, 1, 0, 0],
      [0, 0, 1, 1],
    ],
    [0, 1, 0, 2],
  );

  trainer.fit(Array.from({ length: 6 }, () => batch));

  const compiled = trainer.compileModel({ inputShape: batch.input.shape });
  engine.reset();
  const eagerOut = model.forward(batch.input);
  const compiledOut = compiled.forward(batch.input);

  expect(compiledOut.shape).toEqual(eagerOut.shape);
  compiledOut.data.forEach((value, index) => {
    expect(value).toBeCloseTo(eagerOut.data[index], 5);
  });
});

test('Trainer compiles trained model for post-training WAT inference', () => {
  const model = new Classifier();
  const optimizer = new SGD(model.parameters(), { lr: 0.05 });
  const trainer = new Trainer({
    model,
    optimizer,
    lossFn: new CrossEntropyLoss(),
    maxSteps: 6,
  });
  const batch = makeBatch(
    [
      [1, 0, 0, 1],
      [0, 1, 1, 0],
      [1, 1, 0, 0],
      [0, 0, 1, 1],
    ],
    [0, 1, 0, 2],
  );

  trainer.fit(Array.from({ length: 6 }, () => batch));

  const compiled = trainer.compileModel({ inputShape: batch.input.shape, backend: 'wat' });
  engine.reset();
  const eagerOut = model.forward(batch.input);
  const compiledOut = compiled.forward(batch.input);

  expect(compiledOut.shape).toEqual(eagerOut.shape);
  expect(Array.from(compiledOut.data)).toEqual(Array.from(eagerOut.data));
});

