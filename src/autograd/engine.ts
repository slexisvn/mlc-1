import { Tensor } from '../tensor/tensor.js';

export type GradFn = (gradOutput: Tensor, savedTensors: Tensor[]) => Tensor[];

export interface TapeEntry {
  op: string;
  inputs: Tensor[];
  output: Tensor;
  savedTensors: Tensor[];
  gradFn: GradFn;
}

export class AutogradEngine {
  tape: TapeEntry[] = [];

  record(
    op: string,
    inputs: Tensor[],
    output: Tensor,
    savedTensors: Tensor[],
    gradFn: GradFn
  ): void {
    const entry: TapeEntry = { op, inputs, output, savedTensors, gradFn };
    output._creator = entry;
    this.tape.push(entry);
  }

  backward(loss: Tensor): void {
    const lossSeed = loss.isMeta ? Tensor.meta(loss.shape) : Tensor.ones(loss.shape);
    loss.grad = lossSeed;

    for (let i = this.tape.length - 1; i >= 0; i--) {
      const entry = this.tape[i];
      const gradOut = entry.output.grad;
      if (!gradOut) continue;

      const gradInputs = entry.gradFn(gradOut, entry.savedTensors);

      for (let j = 0; j < entry.inputs.length; j++) {
        const inp = entry.inputs[j];
        if (!inp.requiresGrad && !inp._creator) continue;

        const gradInput = gradInputs[j];
        if (!gradInput) continue;

        const reduced = this.reduceBroadcastGrad(gradInput, inp.shape);
        if (inp.grad) {
          inp.grad = inp.grad.add(reduced);
        } else {
          inp.grad = reduced.clone();
        }
      }
    }
  }

  private reduceBroadcastGrad(grad: Tensor, targetShape: number[]): Tensor {
    if (arraysEqual(grad.shape, targetShape)) return grad;
    if (grad.isMeta) return Tensor.meta(targetShape);

    const gradSize = grad.size;
    const targetSize = targetShape.reduce((a, b) => a * b, 1);

    if (gradSize === targetSize) {
      return new Tensor(new Float32Array(grad.data), targetShape);
    }

    if (gradSize < targetSize) {
      const result = Tensor.zeros(targetShape);
      for (let i = 0; i < targetSize; i++) result.data[i] = grad.data[i % gradSize];
      return result;
    }

    let result = grad;
    const gradDims = result.shape.length;
    const targetDims = targetShape.length;
    const offset = gradDims - targetDims;

    for (let i = 0; i < offset; i++) result = result.sum(0);
    for (let i = 0; i < targetShape.length; i++) {
      if (i < result.shape.length && targetShape[i] === 1 && result.shape[i] !== 1) {
        result = result.sum(i, true);
      }
    }

    const resultSize = result.shape.reduce((a, b) => a * b, 1);
    if (resultSize === targetSize) return new Tensor(new Float32Array(result.data), targetShape);
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

export const engine = new AutogradEngine();
