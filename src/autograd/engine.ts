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
    // 1. Initialize loss gradient = 1.0 (scalar loss)
    loss.grad = NDArray.ones(loss.data.shape);

    // 2. Traverse tape in reverse (reverse topological order)
    for (let i = this.tape.length - 1; i >= 0; i--) {
      const entry = this.tape[i];
      const gradOut = entry.output.grad;
      if (!gradOut) continue;

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

        if (inp.grad) {
          inp.grad = inp.grad.add(reduced);
        } else {
          inp.grad = reduced.clone();
        }
      }
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
  }
}

function arraysEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Global engine instance
export const engine = new AutogradEngine();
