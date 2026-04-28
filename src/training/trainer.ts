import { engine } from '../autograd/engine.js';
import {
  type CompileOptions,
  type CompiledModule,
  compileTrained,
} from '../compile/decorator.js';
import { Loss } from '../loss/loss.js';
import { Module } from '../model/nn.js';
import { Optimizer } from '../optim/types.js';
import { Tensor } from '../tensor/tensor.js';
import { LightningModule, type TrainerBatch } from './lightning.js';

export interface TrainerOptions {
  model: Module;
  optimizer: Optimizer;
  lossFn: Loss;
  maxSteps?: number;
}

export interface LightningTrainerOptions {
  module: LightningModule;
  maxSteps?: number;
}

function isLightningOptions(
  opts: TrainerOptions | LightningTrainerOptions,
): opts is LightningTrainerOptions {
  return 'module' in opts;
}

export class Trainer {
  readonly model: Module;
  readonly optimizer: Optimizer;
  readonly lossFn: Loss | null;
  readonly lightningModule: LightningModule | null;
  readonly maxSteps?: number;
  private readonly compiledCache = new Map<string, CompiledModule>();

  constructor(opts: TrainerOptions | LightningTrainerOptions) {
    if (isLightningOptions(opts)) {
      this.model = opts.module;
      this.optimizer = opts.module.optimizer;
      this.lossFn = null;
      this.lightningModule = opts.module;
      this.maxSteps = opts.maxSteps;
      return;
    }

    this.model = opts.model;
    this.optimizer = opts.optimizer;
    this.lossFn = opts.lossFn;
    this.lightningModule = null;
    this.maxSteps = opts.maxSteps;
  }

  train(): void {
    this.model.train();
  }

  eval(): void {
    this.model.eval();
  }

  fit(trainBatches: TrainerBatch[], valBatches?: TrainerBatch[]): void {
    this.train();

    const totalSteps = this.maxSteps === undefined
      ? trainBatches.length
      : Math.min(this.maxSteps, trainBatches.length);

    for (let step = 0; step < totalSteps; step++) {
      this.trainingStep(trainBatches[step]);
    }

    if (valBatches && valBatches.length > 0) {
      this.validate(valBatches);
    }
  }

  trainingStep(batch: TrainerBatch): Tensor {
    this.train();
    engine.reset();

    let loss: Tensor;

    if (this.lightningModule) {
      loss = this.lightningModule.trainingStep(batch);
      this.lightningModule.backward(loss);
    } else {
      const output = this.model.forward(batch.input);
      loss = this.requireLossFn().forward(output, batch.target);
      engine.backward(loss);
    }

    this.optimizer.step();
    this.optimizer.zeroGrad();
    this.compiledCache.clear();
    return loss;
  }

  validate(batches: TrainerBatch[]): Tensor[] {
    this.eval();
    return batches.map(batch => this.validationStep(batch));
  }

  validationStep(batch: TrainerBatch): Tensor {
    this.eval();
    engine.reset();

    if (this.lightningModule) {
      return this.lightningModule.validationStep(batch);
    }

    const output = this.model.forward(batch.input);
    return this.requireLossFn().forward(output, batch.target);
  }

  compileModel(opts: CompileOptions = {}): CompiledModule {
    const backend = opts.backend ?? 'js';
    const useRegTile = opts.useRegTile ?? true;
    const inputShape = opts.inputShape ?? [];
    const key = `${backend}|${useRegTile ? 'reg' : 'plain'}|${inputShape.join('x')}`;

    const cached = this.compiledCache.get(key);
    if (cached) {
      if (opts.verbose) {
        console.log('');
        console.log(`Reusing cached compiled model for backend '${backend}' and input shape [${inputShape.join(', ')}].`);
      }
      return cached;
    }

    const compiled = compileTrained(this.model, opts);
    this.compiledCache.set(key, compiled);
    return compiled;
  }

  private requireLossFn(): Loss {
    if (!this.lossFn) {
      throw new Error('Trainer was created from a LightningModule. Put loss computation inside module.trainingStep().');
    }
    return this.lossFn;
  }
}
