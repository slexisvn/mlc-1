import { Tensor } from '../tensor/tensor.js';
import { engine } from './engine.js';

export function denseOp(x: Tensor, w: Tensor): Tensor {
  const result = x.matmul(w.transpose()).withGrad(x.requiresGrad || w.requiresGrad);
  engine.record('nn.dense', [x, w], result, [x, w], (gradOut, [savedX, savedW]) => {
    const dX = gradOut.matmul(savedW);
    const dW = gradOut.transpose().matmul(savedX);
    return [dX, dW];
  });
  return result;
}

export function matmulOp(a: Tensor, b: Tensor): Tensor {
  const result = a.matmul(b).withGrad(a.requiresGrad || b.requiresGrad);
  engine.record('matmul', [a, b], result, [a, b], (gradOut, [savedA, savedB]) => {
    const dA = gradOut.matmul(savedB.transpose());
    const dB = savedA.transpose().matmul(gradOut);
    return [dA, dB];
  });
  return result;
}

export function biasAddOp(x: Tensor, bias: Tensor): Tensor {
  const result = x.add(bias).withGrad(x.requiresGrad || bias.requiresGrad);
  engine.record('bias_add', [x, bias], result, [], gradOut => [gradOut, gradOut.sum(0)]);
  return result;
}

export function reluOp(x: Tensor): Tensor {
  const out = x.isMeta ? Tensor.meta(x.shape, x.requiresGrad) : Tensor.zeros(x.shape, true);
  if (!x.isMeta) {
    for (let i = 0; i < x.size; i++) out.data[i] = Math.max(x.data[i], 0);
  }
  engine.record('relu', [x], out, [x], (gradOut, [savedX]) => [gradOut.mul(savedX.gt(0))]);
  return out;
}

export function sigmoidOp(x: Tensor): Tensor {
  const out = x.isMeta ? Tensor.meta(x.shape, x.requiresGrad) : Tensor.zeros(x.shape, true);
  if (!x.isMeta) {
    for (let i = 0; i < x.size; i++) out.data[i] = 1 / (1 + Math.exp(-x.data[i]));
  }
  engine.record('sigmoid', [x], out, [out], (gradOut, [savedY]) => {
    const oneMinusY = Tensor.ones(savedY.shape).sub(savedY);
    return [gradOut.mul(savedY.mul(oneMinusY))];
  });
  return out;
}

export function tanhOp(x: Tensor): Tensor {
  const out = x.isMeta ? Tensor.meta(x.shape, x.requiresGrad) : Tensor.zeros(x.shape, true);
  if (!x.isMeta) {
    for (let i = 0; i < x.size; i++) out.data[i] = Math.tanh(x.data[i]);
  }
  engine.record('tanh', [x], out, [out], (gradOut, [savedY]) => {
    const factor = Tensor.ones(savedY.shape).sub(savedY.mul(savedY));
    return [gradOut.mul(factor)];
  });
  return out;
}

export function leakyReluOp(x: Tensor, alpha = 0.01): Tensor {
  const out = x.isMeta ? Tensor.meta(x.shape, x.requiresGrad) : Tensor.zeros(x.shape, true);
  if (!x.isMeta) {
    for (let i = 0; i < x.size; i++) out.data[i] = x.data[i] > 0 ? x.data[i] : alpha * x.data[i];
  }
  engine.record('leaky_relu', [x], out, [x], (gradOut, [savedX]) => {
    if (savedX.isMeta) return [Tensor.meta(savedX.shape)];
    const gradInput = Tensor.zeros(savedX.shape);
    for (let i = 0; i < savedX.size; i++) gradInput.data[i] = savedX.data[i] > 0 ? gradOut.data[i] : alpha * gradOut.data[i];
    return [gradInput];
  });
  return out;
}

export function softmaxOp(x: Tensor, axis = -1): Tensor {
  const a = axis < 0 ? x.ndim + axis : axis;
  if (x.isMeta) {
    const out = Tensor.meta(x.shape, x.requiresGrad);
    engine.record('softmax', [x], out, [out], gradOut => [Tensor.meta(gradOut.shape)]);
    return out;
  }
  const maxVals = x.max(a, true);
  const exps = x.sub(maxVals).exp();
  const out = exps.div(exps.sum(a, true)).withGrad(true);
  engine.record('softmax', [x], out, [out], (gradOut, [savedY]) => {
    const dot = gradOut.mul(savedY).sum(a, true);
    return [savedY.mul(gradOut.sub(dot))];
  });
  return out;
}

export function logOp(x: Tensor): Tensor {
  const result = x.log().withGrad(x.requiresGrad);
  engine.record('log', [x], result, [x], (gradOut, [savedX]) => [gradOut.div(savedX)]);
  return result;
}

export function expOp(x: Tensor): Tensor {
  const out = x.exp().withGrad(x.requiresGrad);
  engine.record('exp', [x], out, [out], (gradOut, [savedY]) => [gradOut.mul(savedY)]);
  return out;
}

export function sumOp(x: Tensor, axis?: number): Tensor {
  const result = x.sum(axis).withGrad(x.requiresGrad);
  const inputShape = [...x.shape];
  engine.record('sum', [x], result, [], gradOut => {
    if (x.isMeta) return [Tensor.meta(inputShape)];
    let expanded = gradOut;
    if (axis !== undefined) {
      const expandedShape = [...inputShape];
      expandedShape[axis < 0 ? inputShape.length + axis : axis] = 1;
      expanded = gradOut.reshape(expandedShape);
    }
    const grad = Tensor.zeros(inputShape);
    for (let i = 0; i < grad.size; i++) grad.data[i] = expanded.data[i % expanded.size];
    return [grad];
  });
  return result;
}

export function meanOp(x: Tensor, axis?: number): Tensor {
  const n = axis !== undefined ? x.shape[axis < 0 ? x.ndim + axis : axis] : x.size;
  const result = x.mean(axis).withGrad(x.requiresGrad);
  const inputShape = [...x.shape];
  engine.record('mean', [x], result, [], gradOut => {
    if (x.isMeta) return [Tensor.meta(inputShape)];
    let expanded = gradOut;
    if (axis !== undefined) {
      const expandedShape = [...inputShape];
      expandedShape[axis < 0 ? inputShape.length + axis : axis] = 1;
      expanded = gradOut.reshape(expandedShape);
    }
    const grad = Tensor.zeros(inputShape);
    for (let i = 0; i < grad.size; i++) grad.data[i] = expanded.data[i % expanded.size] / n;
    return [grad];
  });
  return result;
}

export function mulOp(a: Tensor, b: Tensor): Tensor {
  const result = a.mul(b).withGrad(a.requiresGrad || b.requiresGrad);
  engine.record('multiply', [a, b], result, [a, b], (gradOut, [savedA, savedB]) => [gradOut.mul(savedB), gradOut.mul(savedA)]);
  return result;
}

export function addOp(a: Tensor, b: Tensor): Tensor {
  const result = a.add(b).withGrad(a.requiresGrad || b.requiresGrad);
  engine.record('add', [a, b], result, [], gradOut => [gradOut.clone(), gradOut.clone()]);
  return result;
}

export function subOp(a: Tensor, b: Tensor): Tensor {
  const result = a.sub(b).withGrad(a.requiresGrad || b.requiresGrad);
  engine.record('subtract', [a, b], result, [], gradOut => [gradOut.clone(), gradOut.neg()]);
  return result;
}

export function negOp(a: Tensor): Tensor {
  const result = a.neg().withGrad(a.requiresGrad);
  engine.record('neg', [a], result, [], gradOut => [gradOut.neg()]);
  return result;
}

export function batchNormOp(x: Tensor, gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor {
  const [b, c] = x.shape;
  if (x.isMeta || gamma.isMeta || beta.isMeta) {
    const out = Tensor.meta([b, c], x.requiresGrad || gamma.requiresGrad || beta.requiresGrad);
    engine.record('batch_norm', [x, gamma, beta], out, [x, gamma, beta], () => [
      Tensor.meta([b, c]),
      Tensor.meta([1, c]),
      Tensor.meta([1, c]),
    ]);
    return out;
  }

  const mean = new Float32Array(c);
  const variance = new Float32Array(c);
  for (let col = 0; col < c; col++) {
    let sum = 0;
    for (let row = 0; row < b; row++) sum += x.data[row * c + col];
    mean[col] = sum / b;
  }
  for (let col = 0; col < c; col++) {
    let sq = 0;
    for (let row = 0; row < b; row++) {
      const diff = x.data[row * c + col] - mean[col];
      sq += diff * diff;
    }
    variance[col] = sq / b;
  }
  const xHat = new Float32Array(b * c);
  const stdInv = new Float32Array(c);
  for (let col = 0; col < c; col++) stdInv[col] = 1 / Math.sqrt(variance[col] + eps);
  for (let row = 0; row < b; row++) {
    for (let col = 0; col < c; col++) {
      xHat[row * c + col] = (x.data[row * c + col] - mean[col]) * stdInv[col];
    }
  }
  const out = Tensor.zeros([b, c], true);
  for (let row = 0; row < b; row++) {
    for (let col = 0; col < c; col++) {
      const idx = row * c + col;
      out.data[idx] = gamma.data[col] * xHat[idx] + beta.data[col];
    }
  }
  const savedXHat = new Tensor(new Float32Array(xHat), [b, c]);
  engine.record('batch_norm', [x, gamma, beta], out, [x, gamma, savedXHat], dY => {
    const dGamma = Tensor.zeros([1, c]);
    const dBeta = Tensor.zeros([1, c]);
    const dX = Tensor.zeros([b, c]);
    for (let col = 0; col < c; col++) {
      for (let row = 0; row < b; row++) {
        const idx = row * c + col;
        dGamma.data[col] += dY.data[idx] * savedXHat.data[idx];
        dBeta.data[col] += dY.data[idx];
        dX.data[idx] = dY.data[idx] * gamma.data[col];
      }
    }
    return [dX, dGamma, dBeta];
  });
  return out;
}
