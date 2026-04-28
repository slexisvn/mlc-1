// ═══════════════════════════════════════════════════════════════
//  Autograd Engine — Reverse-mode Automatic Differentiation
//  Records forward ops into tape, then backward() traverses
//  the tape in reverse to compute gradients.
// ═══════════════════════════════════════════════════════════════

import { NDArray } from '../tensor/ndarray.js';

export type GradFn = (gradOutput: NDArray, savedTensors: NDArray[]) => NDArray[];

export interface TapeEntry {
  op: string;
  inputs: GradTensor[];
  output: GradTensor;
  savedTensors: NDArray[];
  gradFn: GradFn;
}

export interface BackwardTraceNode {
  id: number;
  op: string;
  inputs: number[];
  outputShape: number[];
  attrs: Record<string, any>;
}

let _nextId = 0;

export class GradTensor {
  id: number;
  data: NDArray;
  grad: NDArray | null = null;
  requiresGrad: boolean;
  _creator: TapeEntry | null = null;

  constructor(data: NDArray, requiresGrad = false) {
    this.id = _nextId++;
    this.data = data;
    this.requiresGrad = requiresGrad;
  }

  get shape(): number[] { return this.data.shape; }

  zeroGrad(): void {
    this.grad = null;
  }
}

export class AutogradEngine {
  tape: TapeEntry[] = [];
  backwardTrace: BackwardTraceNode[] = [];
  gradientTensorIds: Map<number, number> = new Map();
  backwardSeedId?: number;

  record(
    op: string,
    inputs: GradTensor[],
    output: GradTensor,
    savedTensors: NDArray[],
    gradFn: GradFn
  ): void {
    const entry: TapeEntry = { op, inputs, output, savedTensors, gradFn };
    output._creator = entry;
    this.tape.push(entry);
  }

  backward(loss: GradTensor): void {
    this.backwardTrace = [];
    this.gradientTensorIds = new Map();

    // 1. Initialize loss gradient = 1.0 (scalar loss)
    const lossSeed = new GradTensor(NDArray.ones(loss.data.shape), false);
    this.backwardSeedId = lossSeed.id;
    loss.grad = lossSeed.data;
    const gradTensors = new Map<number, GradTensor>();
    gradTensors.set(loss.id, lossSeed);
    this.gradientTensorIds.set(loss.id, lossSeed.id);

    // 2. Traverse tape in reverse (reverse topological order)
    for (let i = this.tape.length - 1; i >= 0; i--) {
      const entry = this.tape[i];
      const gradOut = entry.output.grad;
      if (!gradOut) continue;
      const gradOutTensor = gradTensors.get(entry.output.id);
      if (!gradOutTensor) continue;

      // 3. Compute gradients for this op's inputs
      const gradInputs = entry.gradFn(gradOut, entry.savedTensors);

      // 4. Accumulate gradients to inputs
      for (let j = 0; j < entry.inputs.length; j++) {
        const inp = entry.inputs[j];
        if (!inp.requiresGrad && !inp._creator) continue;

        const gi = gradInputs[j];
        if (!gi) continue;

        // Handle shape mismatch from broadcasting — reduce grad to match input shape
        const reduced = this.reduceBroadcastGrad(gi, inp.data.shape);

        const contribution = this.materializeGradientContribution(
          entry,
          j,
          gradOutTensor,
          gi,
          reduced
        );

        if (inp.grad) {
          inp.grad = inp.grad.add(reduced);
          const prevGradTensor = gradTensors.get(inp.id);
          if (!prevGradTensor) continue;
          const accumulated = new GradTensor(inp.grad, false);
          this.backwardTrace.push({
            id: accumulated.id,
            op: 'add',
            inputs: [prevGradTensor.id, contribution.id],
            outputShape: [...accumulated.data.shape],
            attrs: {},
          });
          gradTensors.set(inp.id, accumulated);
          this.gradientTensorIds.set(inp.id, accumulated.id);
        } else {
          inp.grad = reduced.clone();
          gradTensors.set(inp.id, contribution);
          this.gradientTensorIds.set(inp.id, contribution.id);
        }
      }
    }
  }

  private materializeGradientContribution(
    entry: TapeEntry,
    inputIndex: number,
    gradOutTensor: GradTensor,
    rawGrad: NDArray,
    reducedGrad: NDArray
  ): GradTensor {
    const spec = this.getBackwardOpSpec(entry, inputIndex, gradOutTensor);

    let currentTensor: GradTensor;
    if (spec.kind === 'passthrough') {
      currentTensor = gradOutTensor;
    } else {
      currentTensor = new GradTensor(rawGrad, false);
      this.backwardTrace.push({
        id: currentTensor.id,
        op: spec.op,
        inputs: spec.inputIds,
        outputShape: [...rawGrad.shape],
        attrs: {},
      });
    }

    if (arraysEqual(currentTensor.data.shape, reducedGrad.shape)) {
      return currentTensor;
    }

    const reducedTensor = new GradTensor(reducedGrad, false);
    this.backwardTrace.push({
      id: reducedTensor.id,
      op: 'reduce_sum_to_shape',
      inputs: [currentTensor.id],
      outputShape: [...reducedGrad.shape],
      attrs: {},
    });
    return reducedTensor;
  }

  private getBackwardOpSpec(
    entry: TapeEntry,
    inputIndex: number,
    gradOutTensor: GradTensor
  ): { kind: 'node'; op: string; inputIds: number[] } | { kind: 'passthrough' } {
    switch (entry.op) {
      case 'nn.dense':
        return inputIndex === 0
          ? { kind: 'node', op: 'nn.dense_grad_data', inputIds: [gradOutTensor.id, entry.inputs[1].id] }
          : { kind: 'node', op: 'nn.dense_grad_weight', inputIds: [entry.inputs[0].id, gradOutTensor.id] };
      case 'bias_add':
        return inputIndex === 0
          ? { kind: 'passthrough' }
          : { kind: 'node', op: 'nn.bias_add_grad', inputIds: [gradOutTensor.id] };
      case 'relu':
        return { kind: 'node', op: 'nn.relu_grad', inputIds: [gradOutTensor.id, entry.inputs[0].id] };
      case 'sigmoid':
        return { kind: 'node', op: 'nn.sigmoid_grad', inputIds: [gradOutTensor.id, entry.output.id] };
      case 'tanh':
        return { kind: 'node', op: 'nn.tanh_grad', inputIds: [gradOutTensor.id, entry.output.id] };
      case 'leaky_relu':
        return { kind: 'node', op: 'nn.leaky_relu_grad', inputIds: [gradOutTensor.id, entry.inputs[0].id] };
      case 'softmax':
        return { kind: 'node', op: 'nn.softmax_grad', inputIds: [gradOutTensor.id, entry.output.id] };
      case 'cross_entropy':
        return inputIndex === 0
          ? { kind: 'node', op: 'nn.cross_entropy_grad', inputIds: [entry.inputs[0].id, entry.inputs[1].id] }
          : { kind: 'node', op: 'zero_like', inputIds: [entry.inputs[1].id] };
      case 'bce_with_logits':
        return inputIndex === 0
          ? { kind: 'node', op: 'nn.bce_grad', inputIds: [entry.inputs[0].id, entry.inputs[1].id] }
          : { kind: 'node', op: 'zero_like', inputIds: [entry.inputs[1].id] };
      case 'mse':
        return inputIndex === 0
          ? { kind: 'node', op: 'mse_grad', inputIds: [entry.inputs[0].id, entry.inputs[1].id] }
          : { kind: 'node', op: 'zero_like', inputIds: [entry.inputs[1].id] };
      case 'add':
        return { kind: 'passthrough' };
      case 'subtract':
        return inputIndex === 0
          ? { kind: 'passthrough' }
          : { kind: 'node', op: 'neg', inputIds: [gradOutTensor.id] };
      case 'multiply':
        return inputIndex === 0
          ? { kind: 'node', op: 'multiply', inputIds: [gradOutTensor.id, entry.inputs[1].id] }
          : { kind: 'node', op: 'multiply', inputIds: [gradOutTensor.id, entry.inputs[0].id] };
      case 'exp':
        return { kind: 'node', op: 'multiply', inputIds: [gradOutTensor.id, entry.output.id] };
      case 'neg':
        return { kind: 'node', op: 'neg', inputIds: [gradOutTensor.id] };
      default:
        return { kind: 'node', op: `${entry.op}_grad_${inputIndex}`, inputIds: [gradOutTensor.id, ...entry.inputs.map(inp => inp.id)] };
    }
  }

  private reduceBroadcastGrad(grad: NDArray, targetShape: number[]): NDArray {
    if (arraysEqual(grad.shape, targetShape)) return grad;

    // If sizes are the same, just reshape
    const gradSize = grad.shape.reduce((a, b) => a * b, 1);
    const targetSize = targetShape.reduce((a, b) => a * b, 1);
    if (gradSize === targetSize) {
      return new NDArray(new Float32Array(grad.data), targetShape);
    }

    // If grad is smaller than target (e.g., loss=scalar, but target shape is [1,10],
    // broadcast the gradient)
    if (gradSize < targetSize) {
      const result = NDArray.zeros(targetShape);
      for (let i = 0; i < targetSize; i++) {
        result.data[i] = grad.data[i % gradSize];
      }
      return result;
    }

    // Grad is larger than target — need to sum/reduce
    let result = grad;
    const gradDims = result.shape.length;
    const targetDims = targetShape.length;
    const offset = gradDims - targetDims;

    // Sum over leading dimensions that don't exist in target
    for (let i = 0; i < offset; i++) {
      result = result.sum(0);
    }

    // Sum over dimensions where target is 1 (broadcast dims)
    for (let i = 0; i < targetShape.length; i++) {
      if (i < result.shape.length && targetShape[i] === 1 && result.shape[i] !== 1) {
        result = result.sum(i, true);
      }
    }

    // Final reshape if needed
    const resultSize = result.shape.reduce((a, b) => a * b, 1);
    if (resultSize === targetSize) {
      return new NDArray(new Float32Array(result.data), targetShape);
    }

    return result;
  }

  reset(): void {
    this.tape = [];
    this.backwardTrace = [];
    this.gradientTensorIds = new Map();
    this.backwardSeedId = undefined;
  }
}

function arraysEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Global engine instance
export const engine = new AutogradEngine();
