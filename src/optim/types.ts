import { Tensor } from '../tensor/tensor.js';

export interface Optimizer {
  readonly type: 'sgd' | 'adam';
  readonly params: Tensor[];
  step(): void;
  zeroGrad(): void;
}
