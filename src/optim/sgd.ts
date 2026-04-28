// ═══════════════════════════════════════════════════════════════
//  SGD Optimizer — Stochastic Gradient Descent
//  weight = weight - lr * grad
// ═══════════════════════════════════════════════════════════════

import { Tensor } from '../tensor/tensor.js';
import { Optimizer } from './types.js';

export interface SGDOptions {
  lr: number;
}

export class SGD implements Optimizer {
  readonly type = 'sgd' as const;
  readonly params: Tensor[];
  readonly lr: number;

  constructor(params: Tensor[], opts: number | SGDOptions) {
    this.params = params;
    this.lr = typeof opts === 'number' ? opts : opts.lr;
  }

  step(): void {
    for (const param of this.params) {
      if (param.grad) {
        // In-place update: W = W - lr * grad
        for (let i = 0; i < param.size; i++) {
          param.data[i] -= this.lr * param.grad.data[i];
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
