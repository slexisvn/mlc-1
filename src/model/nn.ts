import { Tensor } from '../tensor/tensor.js';
import {
  denseOp, biasAddOp, reluOp, sigmoidOp, tanhOp,
  leakyReluOp, softmaxOp, batchNormOp
} from '../autograd/grad_ops.js';

export abstract class Module {
  training = true;

  abstract forward(x: Tensor): Tensor;

  parameters(): Tensor[] {
    return [];
  }

  children(): Module[] {
    return [];
  }

  train(): this {
    this.training = true;
    for (const child of this.children()) child.train();
    return this;
  }

  eval(): this {
    this.training = false;
    for (const child of this.children()) child.eval();
    return this;
  }
}

export class Linear extends Module {
  weight: Tensor;
  bias: Tensor;
  inFeatures: number;
  outFeatures: number;

  constructor(inFeatures: number, outFeatures: number) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    const scale = Math.sqrt(2 / (inFeatures + outFeatures));
    this.weight = Tensor.randn([outFeatures, inFeatures], true).mul(scale);
    this.weight.requiresGrad = true;
    this.bias = Tensor.zeros([1, outFeatures], true);
  }

  forward(x: Tensor): Tensor {
    return biasAddOp(denseOp(x, this.weight), this.bias);
  }

  parameters(): Tensor[] {
    return [this.weight, this.bias];
  }
}

export class ReLU extends Module {
  forward(x: Tensor): Tensor { return reluOp(x); }
}
export class Sigmoid extends Module {
  forward(x: Tensor): Tensor { return sigmoidOp(x); }
}
export class Tanh extends Module {
  forward(x: Tensor): Tensor { return tanhOp(x); }
}
export class LeakyReLU extends Module {
  alpha: number;
  constructor(alpha = 0.01) { super(); this.alpha = alpha; }
  forward(x: Tensor): Tensor { return leakyReluOp(x, this.alpha); }
}
export class Softmax extends Module {
  axis: number;
  constructor(axis = -1) { super(); this.axis = axis; }
  forward(x: Tensor): Tensor { return softmaxOp(x, this.axis); }
}

export class Sequential extends Module {
  layers: Module[];
  constructor(layers: Module[]) { super(); this.layers = layers; }
  forward(x: Tensor): Tensor {
    let out = x;
    for (const layer of this.layers) out = layer.forward(out);
    return out;
  }
  parameters(): Tensor[] {
    const params: Tensor[] = [];
    for (const layer of this.layers) params.push(...layer.parameters());
    return params;
  }
  override children(): Module[] { return this.layers; }
}

export class BatchNorm extends Module {
  numFeatures: number;
  eps: number;
  gamma: Tensor;
  beta: Tensor;

  constructor(numFeatures: number, eps = 1e-5) {
    super();
    this.numFeatures = numFeatures;
    this.eps = eps;
    this.gamma = Tensor.ones([1, numFeatures], true);
    this.beta = Tensor.zeros([1, numFeatures], true);
  }

  forward(x: Tensor): Tensor {
    return batchNormOp(x, this.gamma, this.beta, this.eps);
  }

  parameters(): Tensor[] {
    return [this.gamma, this.beta];
  }
}
