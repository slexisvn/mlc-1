// ═══════════════════════════════════════════════════════════════
//  Neural Network Modules — PyTorch-like API
//  Each module defines forward() using GradTensor ops.
//  Autograd automatically tracks backward.
// ═══════════════════════════════════════════════════════════════

import { NDArray } from '../tensor/ndarray.js';
import { GradTensor } from '../autograd/engine.js';
import {
  denseOp, biasAddOp, reluOp, sigmoidOp, tanhOp,
  leakyReluOp, softmaxOp
} from '../autograd/grad_ops.js';

export abstract class Module {
  abstract forward(x: GradTensor): GradTensor;

  parameters(): GradTensor[] {
    return [];
  }
}

export class Linear extends Module {
  weight: GradTensor;
  bias: GradTensor;
  inFeatures: number;
  outFeatures: number;

  constructor(inFeatures: number, outFeatures: number) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;

    // Xavier initialization: W ~ N(0, sqrt(2 / (in + out)))
    const scale = Math.sqrt(2 / (inFeatures + outFeatures));
    const wData = NDArray.randn([outFeatures, inFeatures]).mul(scale);
    const bData = NDArray.zeros([1, outFeatures]);

    this.weight = new GradTensor(wData, true);
    this.bias = new GradTensor(bData, true);
  }

  forward(x: GradTensor): GradTensor {
    // x: [batch, in_features]
    // weight: [out_features, in_features]
    // output = x @ weight^T + bias = [batch, out_features]
    const out = denseOp(x, this.weight);
    return biasAddOp(out, this.bias);
  }

  parameters(): GradTensor[] {
    return [this.weight, this.bias];
  }
}

export class ReLU extends Module {
  forward(x: GradTensor): GradTensor {
    return reluOp(x);
  }
}

export class Sigmoid extends Module {
  forward(x: GradTensor): GradTensor {
    return sigmoidOp(x);
  }
}

export class Tanh extends Module {
  forward(x: GradTensor): GradTensor {
    return tanhOp(x);
  }
}

export class LeakyReLU extends Module {
  alpha: number;
  constructor(alpha = 0.01) {
    super();
    this.alpha = alpha;
  }
  forward(x: GradTensor): GradTensor {
    return leakyReluOp(x, this.alpha);
  }
}

export class Softmax extends Module {
  axis: number;
  constructor(axis = -1) {
    super();
    this.axis = axis;
  }
  forward(x: GradTensor): GradTensor {
    return softmaxOp(x, this.axis);
  }
}

export class Sequential extends Module {
  layers: Module[];

  constructor(layers: Module[]) {
    super();
    this.layers = layers;
  }

  forward(x: GradTensor): GradTensor {
    let out = x;
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out;
  }

  parameters(): GradTensor[] {
    const params: GradTensor[] = [];
    for (const layer of this.layers) {
      params.push(...layer.parameters());
    }
    return params;
  }
}
