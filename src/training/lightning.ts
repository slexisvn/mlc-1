import { engine } from '../autograd/engine.js';
import { Module } from '../model/nn.js';
import { Optimizer } from '../optim/types.js';
import { Tensor } from '../tensor/tensor.js';

export interface TrainerBatch {
  input: Tensor;
  target: Tensor;
}

export abstract class LightningModule extends Module {
  private _optimizer: Optimizer | null = null;

  abstract trainingStep(batch: TrainerBatch): Tensor;
  abstract configureOptimizers(): Optimizer;

  validationStep(batch: TrainerBatch): Tensor {
    return this.trainingStep(batch);
  }

  get optimizer(): Optimizer {
    if (!this._optimizer) {
      this._optimizer = this.configureOptimizers();
    }
    return this._optimizer;
  }

  backward(loss: Tensor): void {
    engine.backward(loss);
  }
}
