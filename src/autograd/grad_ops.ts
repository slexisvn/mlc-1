// ═══════════════════════════════════════════════════════════════
//  Gradient Operations — Forward + Backward for each op
//  Each function performs the forward computation and records
//  the backward function into the autograd tape.
// ═══════════════════════════════════════════════════════════════

import { NDArray } from '../tensor/ndarray.js';
import { GradTensor, engine } from './engine.js';

// ─── DENSE ───
// Forward: C = A @ B^T          A:[M,K] B:[N,K] → C:[M,N]
export function denseOp(X: GradTensor, W: GradTensor): GradTensor {
  const result = new GradTensor(X.data.matmul(W.data.transpose()), X.requiresGrad || W.requiresGrad);
  engine.record('nn.dense', [X, W], result, [X.data, W.data],
    (gradOut, [savedX, savedW]) => {
      const dX = gradOut.matmul(savedW); // [M,N] @ [N,K] = [M,K]
      const dW = gradOut.transpose().matmul(savedX); // [N,M] @ [M,K] = [N,K]
      return [dX, dW];
    }
  );
  return result;
}

// ─── MATMUL ───
// Forward: C = A @ B            A:[M,K] B:[K,N] → C:[M,N]
// Backward: dA = dC @ B^T       dB = A^T @ dC
export function matmulOp(A: GradTensor, B: GradTensor): GradTensor {
  const result = new GradTensor(A.data.matmul(B.data), true);
  engine.record('matmul', [A, B], result, [A.data, B.data],
    (gradOut, [savedA, savedB]) => {
      const dA = gradOut.matmul(savedB.transpose());
      const dB = savedA.transpose().matmul(gradOut);
      return [dA, dB];
    }
  );
  return result;
}

// ─── BIAS ADD ───
// Forward: Y = X + bias         X:[M,N] bias:[N] → Y:[M,N]
// Backward: dX = dY             dbias = sum(dY, axis=0)
export function biasAddOp(X: GradTensor, bias: GradTensor): GradTensor {
  const result = new GradTensor(X.data.add(bias.data), true);
  engine.record('bias_add', [X, bias], result, [],
    (gradOut) => {
      const dX = gradOut;
      const dBias = gradOut.sum(0);
      return [dX, dBias];
    }
  );
  return result;
}

// ─── RELU ───
// Forward: Y = max(X, 0)
// Backward: dX = dY * (X > 0)
export function reluOp(X: GradTensor): GradTensor {
  const outData = NDArray.zeros(X.data.shape);
  for (let i = 0; i < X.data.size; i++) {
    outData.data[i] = Math.max(X.data.data[i], 0);
  }
  const result = new GradTensor(outData, true);
  engine.record('relu', [X], result, [X.data],
    (gradOut, [savedX]) => {
      const mask = savedX.gt(0);
      return [gradOut.mul(mask)];
    }
  );
  return result;
}

// ─── SIGMOID ───
// Forward: Y = 1 / (1 + exp(-X))
// Backward: dX = dY * Y * (1 - Y)
export function sigmoidOp(X: GradTensor): GradTensor {
  const outData = NDArray.zeros(X.data.shape);
  for (let i = 0; i < X.data.size; i++) {
    const ex = Math.exp(-X.data.data[i]);
    outData.data[i] = 1 / (1 + ex);
  }
  const result = new GradTensor(outData, true);
  engine.record('sigmoid', [X], result, [outData],
    (gradOut, [savedY]) => {
      // dX = dY * Y * (1 - Y)
      const oneMinusY = NDArray.ones(savedY.shape).sub(savedY);
      return [gradOut.mul(savedY.mul(oneMinusY))];
    }
  );
  return result;
}

// ─── TANH ───
// Forward: Y = tanh(X)
// Backward: dX = dY * (1 - Y²)
export function tanhOp(X: GradTensor): GradTensor {
  const outData = NDArray.zeros(X.data.shape);
  for (let i = 0; i < X.data.size; i++) {
    outData.data[i] = Math.tanh(X.data.data[i]);
  }
  const result = new GradTensor(outData, true);
  engine.record('tanh', [X], result, [outData],
    (gradOut, [savedY]) => {
      // dX = dY * (1 - Y²)
      const ySquared = savedY.mul(savedY);
      const factor = NDArray.ones(savedY.shape).sub(ySquared);
      return [gradOut.mul(factor)];
    }
  );
  return result;
}

// ─── LEAKY RELU ───
// Forward: Y = X if X > 0, else alpha * X
// Backward: dX = dY if X > 0, else alpha * dY
export function leakyReluOp(X: GradTensor, alpha = 0.01): GradTensor {
  const outData = NDArray.zeros(X.data.shape);
  for (let i = 0; i < X.data.size; i++) {
    outData.data[i] = X.data.data[i] > 0 ? X.data.data[i] : alpha * X.data.data[i];
  }
  const result = new GradTensor(outData, true);
  engine.record('leaky_relu', [X], result, [X.data],
    (gradOut, [savedX]) => {
      const gradInput = NDArray.zeros(savedX.shape);
      for (let i = 0; i < savedX.size; i++) {
        gradInput.data[i] = savedX.data[i] > 0 ? gradOut.data[i] : alpha * gradOut.data[i];
      }
      return [gradInput];
    }
  );
  return result;
}

// ─── SOFTMAX ───
// Forward: Y[i] = exp(X[i] - max(X)) / sum(exp(X - max(X)))
// Backward: dX[i] = Y[i] * (dY[i] - sum(dY * Y))
export function softmaxOp(X: GradTensor, axis = -1): GradTensor {
  const a = axis < 0 ? X.data.ndim + axis : axis;
  // Numerically stable softmax
  const maxVals = X.data.max(a, true);
  const shifted = X.data.sub(maxVals);
  const exps = shifted.exp();
  const sumExps = exps.sum(a, true);
  const outData = exps.div(sumExps);

  const result = new GradTensor(outData, true);
  engine.record('softmax', [X], result, [outData],
    (gradOut, [savedY]) => {
      // dX[i] = Y[i] * (dY[i] - sum(dY * Y, axis))
      const dotProduct = gradOut.mul(savedY).sum(a, true);
      return [savedY.mul(gradOut.sub(dotProduct))];
    }
  );
  return result;
}

// ─── LOG ───
// Forward: Y = log(X)
// Backward: dX = dY / X
export function logOp(X: GradTensor): GradTensor {
  const result = new GradTensor(X.data.log(), true);
  engine.record('log', [X], result, [X.data],
    (gradOut, [savedX]) => [gradOut.div(savedX)]
  );
  return result;
}

// ─── EXP ───
// Forward: Y = exp(X)
// Backward: dX = dY * Y
export function expOp(X: GradTensor): GradTensor {
  const outData = X.data.exp();
  const result = new GradTensor(outData, true);
  engine.record('exp', [X], result, [outData],
    (gradOut, [savedY]) => [gradOut.mul(savedY)]
  );
  return result;
}

// ─── SUM ───
// Forward: Y = sum(X, axis)
// Backward: dX = broadcast(dY, X.shape)
export function sumOp(X: GradTensor, axis?: number): GradTensor {
  const result = new GradTensor(X.data.sum(axis), true);
  const inputShape = [...X.data.shape];
  engine.record('sum', [X], result, [],
    (gradOut) => {
      // Broadcast grad back to input shape
      let expanded = gradOut;
      if (axis !== undefined) {
        const expandedShape = [...inputShape];
        expandedShape[axis < 0 ? inputShape.length + axis : axis] = 1;
        expanded = gradOut.reshape(expandedShape);
      }
      // Broadcast to full input shape
      const result = NDArray.zeros(inputShape);
      for (let i = 0; i < result.size; i++) {
        result.data[i] = expanded.data[i % expanded.size];
      }
      return [result];
    }
  );
  return result;
}

// ─── MEAN ───
export function meanOp(X: GradTensor, axis?: number): GradTensor {
  const n = axis !== undefined
    ? X.data.shape[axis < 0 ? X.data.ndim + axis : axis]
    : X.data.size;
  const result = new GradTensor(X.data.mean(axis), true);
  const inputShape = [...X.data.shape];
  engine.record('mean', [X], result, [],
    (gradOut) => {
      let expanded = gradOut;
      if (axis !== undefined) {
        const expandedShape = [...inputShape];
        expandedShape[axis < 0 ? inputShape.length + axis : axis] = 1;
        expanded = gradOut.reshape(expandedShape);
      }
      const grad = NDArray.zeros(inputShape);
      for (let i = 0; i < grad.size; i++) {
        grad.data[i] = expanded.data[i % expanded.size] / n;
      }
      return [grad];
    }
  );
  return result;
}

// ─── MULTIPLY (element-wise) ───
// Forward: Y = A * B
// Backward: dA = dY * B,  dB = dY * A
export function mulOp(A: GradTensor, B: GradTensor): GradTensor {
  const result = new GradTensor(A.data.mul(B.data), true);
  engine.record('multiply', [A, B], result, [A.data, B.data],
    (gradOut, [savedA, savedB]) => [gradOut.mul(savedB), gradOut.mul(savedA)]
  );
  return result;
}

// ─── ADD ───
export function addOp(A: GradTensor, B: GradTensor): GradTensor {
  const result = new GradTensor(A.data.add(B.data), true);
  engine.record('add', [A, B], result, [],
    (gradOut) => [gradOut.clone(), gradOut.clone()]
  );
  return result;
}

// ─── SUBTRACT ───
export function subOp(A: GradTensor, B: GradTensor): GradTensor {
  const result = new GradTensor(A.data.sub(B.data), true);
  engine.record('subtract', [A, B], result, [],
    (gradOut) => [gradOut.clone(), gradOut.neg()]
  );
  return result;
}

// ─── NEG ───
export function negOp(A: GradTensor): GradTensor {
  const result = new GradTensor(A.data.neg(), true);
  engine.record('neg', [A], result, [],
    (gradOut) => [gradOut.neg()]
  );
  return result;
}

// ─── BATCH NORM ────────────────────────────────────────────────
// What it does (MLC concept):
//   Normalize a mini-batch of activations to zero mean and unit variance,
//   then scale/shift with learnable parameters γ (scale) and β (shift).
//
//   Training forward:
//     μ = mean(x, axis=0)             [1, C]  per-feature mean
//     σ² = var(x, axis=0)             [1, C]  per-feature variance
//     x̂ = (x - μ) / sqrt(σ² + ε)    [B, C]  normalized
//     y  = γ * x̂ + β                 [B, C]  rescaled
//
//   Backward (through normalization + scale/shift):
//     dγ = sum(dY * x̂, axis=0)
//     dβ = sum(dY, axis=0)
//     dx̂ = dY * γ
//     dvar = sum(dx̂ * (x - μ) * -0.5 * (σ² + ε)^(-3/2), axis=0)
//     dmean = sum(dx̂ * -1/sqrt(σ²+ε), axis=0) + dvar * mean(-2(x-μ), axis=0)
//     dx = dx̂ / sqrt(σ²+ε) + dvar * 2(x-μ)/N + dmean/N
//
//   Why it matters:
//     - Stabilizes and accelerates training (reduces covariate shift)
//     - Enables much higher learning rates
//     - Each call is a "batch_norm" node in the IR; shape-preserving
//
// Inputs:
//   x    : [B, C]  — input activations
//   gamma: [1, C]  — learnable scale
//   beta : [1, C]  — learnable shift
//   eps  : scalar  — numerical stability (default 1e-5)
// ──────────────────────────────────────────────────────────────
export function batchNormOp(
  x: GradTensor,
  gamma: GradTensor,
  beta: GradTensor,
  eps = 1e-5
): GradTensor {
  const [B, C] = x.data.shape;

  // Compute per-feature mean and variance
  const mean = new Float32Array(C);
  const variance = new Float32Array(C);

  for (let c = 0; c < C; c++) {
    let sum = 0;
    for (let b = 0; b < B; b++) sum += x.data.data[b * C + c];
    mean[c] = sum / B;
  }
  for (let c = 0; c < C; c++) {
    let sq = 0;
    for (let b = 0; b < B; b++) {
      const diff = x.data.data[b * C + c] - mean[c];
      sq += diff * diff;
    }
    variance[c] = sq / B;
  }

  // x̂ = (x - μ) / sqrt(σ² + ε)
  const xHatData = new Float32Array(B * C);
  const stdInv = new Float32Array(C);
  for (let c = 0; c < C; c++) stdInv[c] = 1 / Math.sqrt(variance[c] + eps);

  for (let b = 0; b < B; b++) {
    for (let c = 0; c < C; c++) {
      xHatData[b * C + c] = (x.data.data[b * C + c] - mean[c]) * stdInv[c];
    }
  }

  // y = γ * x̂ + β
  const yData = new Float32Array(B * C);
  for (let b = 0; b < B; b++) {
    for (let c = 0; c < C; c++) {
      const idx = b * C + c;
      yData[idx] = gamma.data.data[c] * xHatData[idx] + beta.data.data[c];
    }
  }

  const result = new GradTensor(new NDArray(yData, [B, C]), true);

  // Save intermediates for backward
  const savedX = x.data;
  const savedGamma = gamma.data;
  const savedXHat = new NDArray(new Float32Array(xHatData), [B, C]);
  const savedStdInv = new Float32Array(stdInv);
  const savedMean = new Float32Array(mean);

  engine.record('batch_norm', [x, gamma, beta], result,
    [savedX, savedGamma, savedXHat],
    (dY, [_sx, _sg, xHat]) => {
      const dYd = dY.data;

      // dγ = sum(dY * x̂, axis=0): [1, C]
      const dGamma = new Float32Array(C);
      // dβ = sum(dY, axis=0): [1, C]
      const dBeta = new Float32Array(C);
      // dx̂ = dY * γ
      const dXhat = new Float32Array(B * C);

      for (let c = 0; c < C; c++) {
        for (let b = 0; b < B; b++) {
          const idx = b * C + c;
          dGamma[c] += dYd[idx] * xHat.data[idx];
          dBeta[c] += dYd[idx];
          dXhat[idx] = dYd[idx] * savedGamma.data[c];
        }
      }

      // dvar = sum(dx̂ * (x - μ) * -0.5 * stdInv³, axis=0)
      const dVar = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        const si3 = savedStdInv[c] * savedStdInv[c] * savedStdInv[c];
        for (let b = 0; b < B; b++) {
          const diff = savedX.data[b * C + c] - savedMean[c];
          dVar[c] += dXhat[b * C + c] * diff * (-0.5) * si3;
        }
      }

      // dmean = sum(dx̂ * -stdInv, axis=0) + dvar * mean(-2(x-μ), axis=0)
      const dMean = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        for (let b = 0; b < B; b++) {
          dMean[c] += dXhat[b * C + c] * (-savedStdInv[c]);
          dMean[c] += dVar[c] * (-2 * (savedX.data[b * C + c] - savedMean[c])) / B;
        }
      }

      // dx = dx̂ * stdInv + dvar * 2(x-μ)/N + dmean/N
      const dX = new Float32Array(B * C);
      for (let b = 0; b < B; b++) {
        for (let c = 0; c < C; c++) {
          const idx = b * C + c;
          const diff = savedX.data[idx] - savedMean[c];
          dX[idx] = dXhat[idx] * savedStdInv[c]
                  + dVar[c] * 2 * diff / B
                  + dMean[c] / B;
        }
      }

      return [
        new NDArray(dX, [B, C]),
        new NDArray(dGamma, [1, C]),
        new NDArray(dBeta, [1, C])
      ];
    }
  );

  return result;
}
