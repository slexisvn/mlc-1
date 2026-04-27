// ═══════════════════════════════════════════════════════════════
//  SGD Optimizer — Stochastic Gradient Descent
//  weight = weight - lr * grad
// ═══════════════════════════════════════════════════════════════

import { GradTensor } from '../autograd/engine.js';

export class SGD {
  params: GradTensor[];
  lr: number;

  constructor(params: GradTensor[], lr: number) {
    this.params = params;
    this.lr = lr;
  }

  step(): void {
    for (const param of this.params) {
      if (param.grad) {
        // In-place update: W = W - lr * grad
        for (let i = 0; i < param.data.size; i++) {
          param.data.data[i] -= this.lr * param.grad.data[i];
        }
      }
    }
  }

  zeroGrad(): void {
    for (const param of this.params) {
      param.zeroGrad();
    }
  }
}
