// ═══════════════════════════════════════════════════════════════
//  Loss Functions
//  - CrossEntropyLoss:    multi-class (softmax + NLL internally)
//  - BCEWithLogitsLoss:   binary & multi-label (sigmoid + BCE)
//  - MSELoss:             regression baseline
// ═══════════════════════════════════════════════════════════════

import { NDArray } from '../tensor/ndarray.js';
import { GradTensor, engine } from '../autograd/engine.js';

export abstract class Loss {
  abstract forward(prediction: GradTensor, target: GradTensor): GradTensor;
}

// ─── CrossEntropyLoss ───
// Combines LogSoftmax + NLLLoss in a numerically stable way.
// Input:  logits [batch, num_classes],  target [batch] (class indices as float)
// Output: scalar loss
//
// Forward: loss = -logits[target] + log(sum(exp(logits)))
// Backward: dLogits = softmax(logits) - one_hot(target)
//
// This is the canonical "fused softmax + cross entropy" that MLC optimizes.
export class CrossEntropyLoss extends Loss {
  forward(logits: GradTensor, target: GradTensor): GradTensor {
    const [batch, numClasses] = logits.data.shape;

    // Numerically stable log-softmax
    const maxVals = logits.data.max(1, true);
    const shifted = logits.data.sub(maxVals);
    const exps = shifted.exp();
    const sumExps = exps.sum(1, true);
    const logSumExp = sumExps.log().add(maxVals);

    // Softmax (saved for backward)
    const softmax = exps.div(sumExps);

    // NLL: -logits[target_class] + logSumExp
    let totalLoss = 0;
    for (let b = 0; b < batch; b++) {
      const targetClass = Math.round(target.data.data[b]);
      totalLoss += -logits.data.data[b * numClasses + targetClass]
        + logSumExp.data[b];
    }
    totalLoss /= batch;

    const lossData = NDArray.fromArray([totalLoss], [1]);
    const result = new GradTensor(lossData, true);

    // Build one_hot for backward
    const oneHot = NDArray.zeros([batch, numClasses]);
    for (let b = 0; b < batch; b++) {
      const targetClass = Math.round(target.data.data[b]);
      oneHot.data[b * numClasses + targetClass] = 1;
    }

    engine.record('cross_entropy', [logits, target], result,
      [softmax, oneHot],
      (gradOut, [savedSoftmax, savedOneHot]) => {
        // dLogits = (softmax - one_hot) / batch * gradOut
        const dLogits = savedSoftmax.sub(savedOneHot).mul(gradOut.data[0] / batch);
        return [dLogits, NDArray.zeros(target.data.shape)]; // no grad for target
      }
    );

    return result;
  }
}

// ─── BCEWithLogitsLoss ───
// Combines Sigmoid + BCE in a numerically stable way.
// Input: logits [batch, *], target [batch, *] (0 or 1, float)
// Output: scalar loss
//
// Forward: loss = max(x,0) - x*t + log(1 + exp(-|x|))
// Backward: dX = sigmoid(x) - target
export class BCEWithLogitsLoss extends Loss {
  forward(logits: GradTensor, target: GradTensor): GradTensor {
    const size = logits.data.size;

    // Numerically stable BCE: max(x,0) - x*t + log(1+exp(-|x|))
    let totalLoss = 0;
    const sigmoid = NDArray.zeros(logits.data.shape);
    for (let i = 0; i < size; i++) {
      const x = logits.data.data[i];
      const t = target.data.data[i];
      const absX = Math.abs(x);
      totalLoss += Math.max(x, 0) - x * t + Math.log(1 + Math.exp(-absX));
      sigmoid.data[i] = 1 / (1 + Math.exp(-x));
    }
    totalLoss /= size;

    const lossData = NDArray.fromArray([totalLoss], [1]);
    const result = new GradTensor(lossData, true);

    engine.record('bce_with_logits', [logits, target], result,
      [sigmoid, target.data],
      (gradOut, [savedSigmoid, savedTarget]) => {
        // dX = (sigmoid(x) - target) / size * gradOut
        const dLogits = savedSigmoid.sub(savedTarget).mul(gradOut.data[0] / size);
        return [dLogits, NDArray.zeros(savedTarget.shape)];
      }
    );

    return result;
  }
}

// ─── MSELoss ───
// Forward: loss = mean((pred - target)²)
// Backward: dPred = 2 * (pred - target) / n
export class MSELoss extends Loss {
  forward(prediction: GradTensor, target: GradTensor): GradTensor {
    const diff = prediction.data.sub(target.data);
    const squared = diff.mul(diff);
    const loss = squared.mean();

    const lossData = loss;
    const result = new GradTensor(lossData, true);

    engine.record('mse', [prediction, target], result,
      [prediction.data, target.data],
      (gradOut, [savedPred, savedTarget]) => {
        const n = savedPred.size;
        const dPred = savedPred.sub(savedTarget).mul(2 / n).mul(gradOut.data[0]);
        return [dPred, NDArray.zeros(savedTarget.shape)];
      }
    );

    return result;
  }
}
