import { Tensor } from '../tensor/tensor.js';
import { engine } from '../autograd/engine.js';

export abstract class Loss {
  abstract forward(prediction: Tensor, target: Tensor): Tensor;
}

export class CrossEntropyLoss extends Loss {
  forward(logits: Tensor, target: Tensor): Tensor {
    const [batch, numClasses] = logits.shape;
    if (logits.isMeta || target.isMeta) {
      const result = Tensor.meta([1], true);
      engine.record('cross_entropy', [logits, target], result, [Tensor.meta(logits.shape), Tensor.meta(logits.shape)], () => [
        Tensor.meta(logits.shape),
        Tensor.meta(target.shape),
      ]);
      return result;
    }

    const maxVals = logits.max(1, true);
    const exps = logits.sub(maxVals).exp();
    const sumExps = exps.sum(1, true);
    const logSumExp = sumExps.log().add(maxVals);
    const softmax = exps.div(sumExps);

    let totalLoss = 0;
    for (let b = 0; b < batch; b++) {
      const targetClass = Math.round(target.data[b]);
      totalLoss += -logits.data[b * numClasses + targetClass] + logSumExp.data[b];
    }
    totalLoss /= batch;

    const result = Tensor.fromArray([totalLoss], [1], true);
    const oneHot = Tensor.zeros([batch, numClasses]);
    for (let b = 0; b < batch; b++) {
      oneHot.data[b * numClasses + Math.round(target.data[b])] = 1;
    }
    engine.record('cross_entropy', [logits, target], result, [softmax, oneHot], (gradOut, [savedSoftmax, savedOneHot]) => {
      const dLogits = savedSoftmax.sub(savedOneHot).mul(gradOut.data[0] / batch);
      return [dLogits, Tensor.zeros(target.shape)];
    });
    return result;
  }
}

export class BCEWithLogitsLoss extends Loss {
  forward(logits: Tensor, target: Tensor): Tensor {
    if (logits.isMeta || target.isMeta) {
      const result = Tensor.meta([1], true);
      engine.record('bce_with_logits', [logits, target], result, [Tensor.meta(logits.shape), Tensor.meta(target.shape)], () => [
        Tensor.meta(logits.shape),
        Tensor.meta(target.shape),
      ]);
      return result;
    }
    const size = logits.size;
    let totalLoss = 0;
    const sigmoid = Tensor.zeros(logits.shape);
    for (let i = 0; i < size; i++) {
      const x = logits.data[i];
      const t = target.data[i];
      totalLoss += Math.max(x, 0) - x * t + Math.log(1 + Math.exp(-Math.abs(x)));
      sigmoid.data[i] = 1 / (1 + Math.exp(-x));
    }
    totalLoss /= size;
    const result = Tensor.fromArray([totalLoss], [1], true);
    engine.record('bce_with_logits', [logits, target], result, [sigmoid, target], (gradOut, [savedSigmoid, savedTarget]) => {
      const dLogits = savedSigmoid.sub(savedTarget).mul(gradOut.data[0] / size);
      return [dLogits, Tensor.zeros(savedTarget.shape)];
    });
    return result;
  }
}

export class MSELoss extends Loss {
  forward(prediction: Tensor, target: Tensor): Tensor {
    const diff = prediction.sub(target);
    const loss = diff.mul(diff).mean();
    loss.requiresGrad = true;
    engine.record('mse', [prediction, target], loss, [prediction, target], (gradOut, [savedPred, savedTarget]) => {
      const dPred = savedPred.sub(savedTarget).mul(2 / savedPred.size).mul(gradOut.data[0]);
      return [dPred, Tensor.zeros(savedTarget.shape)];
    });
    return loss;
  }
}
