// ═══════════════════════════════════════════════════════════════
//  Adam Optimizer — Adaptive Moment Estimation
//
//  Why this matters (MLC concept):
//    Optimizers are the core learning algorithm. Adam is the
//    default optimizer for most deep learning workloads because:
//      - Adaptive per-parameter learning rates (better than SGD for non-convex)
//      - First moment (m): exponential moving average of gradients
//      - Second moment (v): exponential moving average of squared gradients
//      - Bias correction: prevents large updates in early steps
//
//  Update rule:
//    m_t = β₁·m_{t-1} + (1-β₁)·g_t          ← 1st moment (mean)
//    v_t = β₂·v_{t-1} + (1-β₂)·g_t²          ← 2nd moment (variance)
//    m̂_t = m_t / (1 - β₁^t)                  ← bias-corrected mean
//    v̂_t = v_t / (1 - β₂^t)                  ← bias-corrected variance
//    W_t = W_{t-1} - lr · m̂_t / (√v̂_t + ε)  ← weight update
//
//  Compared to SGD:
//    SGD:  W = W - lr·g
//    Adam: W = W - lr·(m̂/√v̂) — effectively normalizes the gradient
//          by its own variance. This is why Adam converges faster.
//
//  Default hyperparameters (from the original paper, Kingma & Ba 2015):
//    β₁ = 0.9, β₂ = 0.999, ε = 1e-8
// ═══════════════════════════════════════════════════════════════

import { Tensor } from '../tensor/tensor.js';
import { Optimizer } from './types.js';

export interface AdamOptions {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
}

export class Adam implements Optimizer {
  readonly type = 'adam' as const;
  readonly params: Tensor[];
  private lr: number;
  private beta1: number;
  private beta2: number;
  private eps: number;
  private step_t: number;

  /** First moment (momentum) buffers — one per param */
  private m: Float32Array[];
  /** Second moment (RMS) buffers — one per param */
  private v: Float32Array[];

  constructor(
    params: Tensor[],
    opts: number | AdamOptions = 0.001,
    legacyBeta1 = 0.9,
    legacyBeta2 = 0.999,
    legacyEps = 1e-8
  ) {
    const normalized = typeof opts === 'number'
      ? { lr: opts, beta1: legacyBeta1, beta2: legacyBeta2, eps: legacyEps }
      : {
          lr: opts.lr ?? 0.001,
          beta1: opts.beta1 ?? 0.9,
          beta2: opts.beta2 ?? 0.999,
          eps: opts.eps ?? 1e-8,
        };

    this.params = params;
    this.lr = normalized.lr;
    this.beta1 = normalized.beta1;
    this.beta2 = normalized.beta2;
    this.eps = normalized.eps;
    this.step_t = 0;

    // Initialize moment buffers to zeros — same shape as each parameter
    this.m = params.map(p => new Float32Array(p.size));
    this.v = params.map(p => new Float32Array(p.size));
  }

  step(): void {
    this.step_t++;
    const { beta1, beta2, eps, lr, step_t } = this;

    // Bias correction factors — cancel the zero-initialization bias in m, v
    // In early steps (t=1), β₁^1 = 0.9, so correction = 1/(1-0.9) = 10×
    // This prevents the optimizer from making tiny steps at the start.
    const bc1 = 1 - Math.pow(beta1, step_t);   // 1 - β₁^t
    const bc2 = 1 - Math.pow(beta2, step_t);   // 1 - β₂^t

    for (let i = 0; i < this.params.length; i++) {
      const param = this.params[i];
      if (!param.grad) continue;

      const g = param.grad.data;   // gradient vector
      const m = this.m[i];         // 1st moment buffer
      const v = this.v[i];         // 2nd moment buffer
      const w = param.data;   // weight vector

      for (let j = 0; j < w.length; j++) {
        const gj = g[j];

        // Update biased first moment: m = β₁·m + (1-β₁)·g
        m[j] = beta1 * m[j] + (1 - beta1) * gj;

        // Update biased second moment: v = β₂·v + (1-β₂)·g²
        v[j] = beta2 * v[j] + (1 - beta2) * gj * gj;

        // Bias-corrected estimates
        const m_hat = m[j] / bc1;
        const v_hat = v[j] / bc2;

        // Weight update: w = w - lr · m̂ / (√v̂ + ε)
        w[j] -= lr * m_hat / (Math.sqrt(v_hat) + eps);
      }
    }
  }

  zeroGrad(): void {
    for (const param of this.params) {
      param.zeroGrad();
    }
  }

  /** Current step count (useful for logging) */
  get currentStep(): number {
    return this.step_t;
  }

  /** Effective learning rate at current step (accounting for bias correction) */
  get effectiveLR(): number {
    if (this.step_t === 0) return this.lr; // before any steps, bias correction = 1
    const bc1 = 1 - Math.pow(this.beta1, this.step_t);
    const bc2 = 1 - Math.pow(this.beta2, this.step_t);
    return this.lr * Math.sqrt(bc2) / bc1;
  }
}
