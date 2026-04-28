import { CrossEntropyLoss } from './loss/loss.js';
import { Linear, Module, ReLU } from './model/nn.js';
import { SGD } from './optim/sgd.js';
import { Tensor, tensor } from './tensor/tensor.js';
import { LightningModule, Trainer, type TrainerBatch } from './training/index.js';
import { codegenWAT } from './codegen/wat_codegen.js';
import type { CompiledModule } from './compile/decorator.js';

class ClassifierTask extends LightningModule {
  readonly fc1 = new Linear(4, 128);
  readonly relu1 = new ReLU();
  readonly fc2 = new Linear(128, 256);
  readonly relu2 = new ReLU();
  readonly deadFc = new Linear(128, 64);
  readonly deadRelu = new ReLU();
  readonly fc3 = new Linear(256, 256);
  readonly relu3 = new ReLU();
  readonly fc4 = new Linear(256, 128);
  readonly relu4 = new ReLU();
  readonly fc5 = new Linear(128, 3);
  readonly lossFn = new CrossEntropyLoss();

  override forward(x: Tensor): Tensor {
    const h1 = this.relu1.forward(this.fc1.forward(x));
    const h2 = this.relu2.forward(this.fc2.forward(h1));

    // Intentionally duplicated dead user code:
    // CSE can spot the repeated subgraph, then DCE removes both branches.
    this.deadRelu.forward(this.deadFc.forward(h1));
    this.deadRelu.forward(this.deadFc.forward(h1));

    const h3 = this.relu3.forward(this.fc3.forward(h2));
    const h4 = this.relu4.forward(this.fc4.forward(h3));
    return this.fc5.forward(h4);
  }

  override parameters(): Tensor[] {
    return [
      ...this.fc1.parameters(),
      ...this.fc2.parameters(),
      ...this.deadFc.parameters(),
      ...this.fc3.parameters(),
      ...this.fc4.parameters(),
      ...this.fc5.parameters(),
    ];
  }

  override children(): Module[] {
    return [
      this.fc1,
      this.relu1,
      this.fc2,
      this.relu2,
      this.deadFc,
      this.deadRelu,
      this.fc3,
      this.relu3,
      this.fc4,
      this.relu4,
      this.fc5,
    ];
  }

  override trainingStep(batch: TrainerBatch): Tensor {
    return this.lossFn.forward(this.forward(batch.input), batch.target);
  }

  override validationStep(batch: TrainerBatch): Tensor {
    return this.lossFn.forward(this.forward(batch.input), batch.target);
  }

  override configureOptimizers(): SGD {
    return new SGD(this.parameters(), { lr: 0.08 });
  }
}

function makeBatch(inputs: number[][], targets: number[]): TrainerBatch {
  return {
    input: tensor(inputs),
    target: tensor(targets),
  };
}

function buildTrainBatches(): TrainerBatch[] {
  const batchA = makeBatch(
    [
      [3, 0, 0, 0],
      [0, 3, 0, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 3],
    ],
    [0, 1, 2, 1],
  );

  const batchB = makeBatch(
    [
      [2, 1, 0, 0],
      [0, 2, 1, 0],
      [0, 0, 2, 1],
      [1, 0, 0, 2],
    ],
    [0, 1, 2, 1],
  );

  return [batchA, batchB, batchA, batchB, batchA, batchB, batchA, batchB];
}

function buildValBatches(): TrainerBatch[] {
  return [
    makeBatch(
      [
        [2, 0, 1, 0],
        [0, 2, 0, 1],
        [1, 0, 2, 0],
        [0, 1, 0, 2],
      ],
      [0, 1, 2, 1],
    ),
  ];
}

function evaluateLoss(task: ClassifierTask, batch: TrainerBatch): number {
  const loss = task.lossFn.forward(task.forward(batch.input), batch.target);
  return loss.data[0];
}

function maxAbsDiff(a: Tensor, b: Tensor): number {
  let diff = 0;
  for (let i = 0; i < a.data.length; i++) {
    diff = Math.max(diff, Math.abs(a.data[i] - b.data[i]));
  }
  return diff;
}

function argmaxPerRow(logits: Tensor): number[] {
  const [batch, numClasses] = logits.shape;
  const predictions: number[] = [];

  for (let row = 0; row < batch; row++) {
    let bestClass = 0;
    let bestValue = logits.data[row * numClasses];
    for (let col = 1; col < numClasses; col++) {
      const value = logits.data[row * numClasses + col];
      if (value > bestValue) {
        bestValue = value;
        bestClass = col;
      }
    }
    predictions.push(bestClass);
  }

  return predictions;
}

function labelsFromTarget(target: Tensor): number[] {
  return Array.from(target.data).map((value) => Math.round(value));
}

function accuracy(predictions: number[], labels: number[]): number {
  let correct = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === labels[i]) correct++;
  }
  return correct / predictions.length;
}

function printPredictionSummary(label: string, logits: Tensor, target: Tensor): void {
  const predictions = argmaxPerRow(logits);
  const labels = labelsFromTarget(target);
  console.log(`${label} predictions: [${predictions.join(', ')}]`);
  console.log(`${label} targets:     [${labels.join(', ')}]`);
  console.log(`${label} accuracy:    ${(accuracy(predictions, labels) * 100).toFixed(1)}%`);
}

function benchmarkCompiledForward(
  compiled: CompiledModule,
  input: Tensor,
  warmup = 200,
  reps = 2000,
): { msPerCall: number; kernelNames: string[] } {
  for (let i = 0; i < warmup; i++) compiled.forward(input);

  const start = performance.now();
  for (let i = 0; i < reps; i++) compiled.forward(input);
  const msPerCall = (performance.now() - start) / reps;

  return {
    msPerCall,
    kernelNames: compiled.primFuncs.map((pf) => pf.name),
  };
}

function main(): void {
  const task = new ClassifierTask();
  const trainBatches = buildTrainBatches();
  const valBatches = buildValBatches();
  const trainer = new Trainer({
    module: task,
    maxSteps: trainBatches.length,
  });

  const sampleBatch = trainBatches[0];
  const initialLoss = evaluateLoss(task, sampleBatch);
  const initialValLoss = evaluateLoss(task, valBatches[0]);

  console.log('=== Trainer-first Showcase ===');
  console.log(`Initial loss: ${initialLoss.toFixed(4)}`);
  console.log(`Initial val loss: ${initialValLoss.toFixed(4)}`);

  trainer.fit(trainBatches, valBatches);

  const finalLoss = evaluateLoss(task, sampleBatch);
  const finalValLoss = trainer.validationStep(valBatches[0]).data[0];
  console.log(`Final loss: ${finalLoss.toFixed(4)}`);
  console.log(`Final val loss: ${finalValLoss.toFixed(4)}`);

  trainer.eval();

  const compiledJs = trainer.compileModel({
    inputShape: sampleBatch.input.shape,
    backend: 'js',
    verbose: true,
  });
  const compiledWat = trainer.compileModel({
    inputShape: sampleBatch.input.shape,
    backend: 'wat',
    verbose: true,
  });
  const watModule = codegenWAT(compiledWat.primFuncs);

  const eagerOutput = task.forward(sampleBatch.input);
  const jsOutput = compiledJs.forward(sampleBatch.input);
  const watOutput = compiledWat.forward(sampleBatch.input);
  const eagerValOutput = task.forward(valBatches[0].input);
  const watValOutput = compiledWat.forward(valBatches[0].input);

  console.log('\n=== Post-training Compile ===');
  console.log(`Model mode: ${task.training ? 'train' : 'eval'}`);
  console.log(`Output shape: [${jsOutput.shape.join(', ')}]`);
  console.log(`JS backend diff vs eager: ${maxAbsDiff(eagerOutput, jsOutput).toExponential(3)}`);
  console.log(`WAT backend diff vs eager: ${maxAbsDiff(eagerOutput, watOutput).toExponential(3)}`);
  console.log(`JS kernels: ${compiledJs.primFuncs.map((pf) => pf.name).join(', ')}`);
  console.log(`WAT kernels: ${compiledWat.primFuncs.map((pf) => pf.name).join(', ')}`);
  console.log(`WAT modes: ${watModule.kernels.map((k) => `${k.name}:${k.mode}`).join(', ')}`);

  console.log('\n=== Inference Quality ===');
  printPredictionSummary('Train eager', eagerOutput, sampleBatch.target);
  printPredictionSummary('Train WAT', watOutput, sampleBatch.target);
  printPredictionSummary('Val eager', eagerValOutput, valBatches[0].target);
  printPredictionSummary('Val WAT', watValOutput, valBatches[0].target);

  const jsBench = benchmarkCompiledForward(compiledJs, sampleBatch.input);
  const watBench = benchmarkCompiledForward(compiledWat, sampleBatch.input);
  const speedup = jsBench.msPerCall / watBench.msPerCall;

  console.log('\n=== Backend Benchmark ===');
  console.log(`JS forward: ${jsBench.msPerCall.toFixed(4)} ms/call`);
  console.log(`WAT forward: ${watBench.msPerCall.toFixed(4)} ms/call`);
  console.log(`Speedup: ${speedup.toFixed(2)}x (${speedup >= 1 ? 'WAT faster' : 'JS faster'})`);
}

main();
