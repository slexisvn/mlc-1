export interface TapeEntryLike {
  op: string;
  inputs: Tensor[];
  output: Tensor;
  savedTensors: Tensor[];
  gradFn: (gradOutput: Tensor, savedTensors: Tensor[]) => Tensor[];
}

export class Tensor {
  static nextId = 0;

  id: number;
  data: Float32Array;
  shape: number[];
  strides: number[];
  requiresGrad: boolean;
  grad: Tensor | null = null;
  _creator: TapeEntryLike | null = null;
  readonly isMeta: boolean;

  constructor(data: Float32Array, shape: number[], requiresGrad = false, isMeta = false) {
    this.id = Tensor.nextId++;
    this.data = data;
    this.shape = shape;
    this.strides = Tensor.computeStrides(shape);
    this.requiresGrad = requiresGrad;
    this.isMeta = isMeta;
  }

  static computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  static meta(shape: number[], requiresGrad = false): MetaTensor {
    return new MetaTensor(shape, requiresGrad);
  }

  static zeros(shape: number[], requiresGrad = false): Tensor {
    return new Tensor(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape, requiresGrad);
  }

  static ones(shape: number[], requiresGrad = false): Tensor {
    return new Tensor(new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1), shape, requiresGrad);
  }

  static full(shape: number[], value: number, requiresGrad = false): Tensor {
    return new Tensor(new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(value), shape, requiresGrad);
  }

  static rand(shape: number[], requiresGrad = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = Math.random();
    return new Tensor(data, shape, requiresGrad);
  }

  static randn(shape: number[], requiresGrad = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i += 2) {
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1));
      data[i] = r * Math.cos(2 * Math.PI * u2);
      if (i + 1 < size) data[i + 1] = r * Math.sin(2 * Math.PI * u2);
    }
    return new Tensor(data, shape, requiresGrad);
  }

  static fromArray(arr: number[], shape?: number[], requiresGrad = false): Tensor {
    return new Tensor(new Float32Array(arr), shape || [arr.length], requiresGrad);
  }

  static fromNestedArray(value: NestedNumberArray, requiresGrad = false): Tensor {
    const { flat, shape } = flattenNestedArray(value);
    return Tensor.fromArray(flat, shape, requiresGrad);
  }

  get size(): number { return this.shape.reduce((a, b) => a * b, 1); }
  get ndim(): number { return this.shape.length; }

  withGrad(requiresGrad = true): Tensor {
    const clone = this.clone();
    clone.requiresGrad = requiresGrad;
    return clone;
  }

  zeroGrad(): void {
    this.grad = null;
  }

  private flatIndex(indices: number[]): number {
    let idx = 0;
    for (let i = 0; i < indices.length; i++) idx += indices[i] * this.strides[i];
    return idx;
  }

  get(indices: number[]): number {
    return this.data[this.flatIndex(indices)];
  }

  set(indices: number[], value: number): void {
    this.data[this.flatIndex(indices)] = value;
  }

  clone(): Tensor {
    return this.isMeta
      ? Tensor.meta([...this.shape], this.requiresGrad)
      : new Tensor(new Float32Array(this.data), [...this.shape], this.requiresGrad);
  }

  reshape(newShape: number[]): Tensor {
    const size = newShape.reduce((a, b) => a * b, 1);
    if (size !== this.size) throw new Error(`Cannot reshape ${this.shape} to ${newShape}`);
    if (this.isMeta) return Tensor.meta(newShape, this.requiresGrad);
    return new Tensor(new Float32Array(this.data), newShape, this.requiresGrad);
  }

  transpose(): Tensor {
    if (this.ndim !== 2) throw new Error('transpose() only for 2D');
    const [m, n] = this.shape;
    if (this.isMeta) return Tensor.meta([n, m], this.requiresGrad);
    const result = Tensor.zeros([n, m], this.requiresGrad);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        result.data[j * m + i] = this.data[i * n + j];
      }
    }
    return result;
  }

  add(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      if (this.isMeta) return Tensor.meta([...this.shape], this.requiresGrad);
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) result[i] = this.data[i] + other;
      return new Tensor(result, [...this.shape], this.requiresGrad);
    }
    return this.broadcastBinaryOp(other, (a, b) => a + b);
  }

  sub(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      if (this.isMeta) return Tensor.meta([...this.shape], this.requiresGrad);
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) result[i] = this.data[i] - other;
      return new Tensor(result, [...this.shape], this.requiresGrad);
    }
    return this.broadcastBinaryOp(other, (a, b) => a - b);
  }

  mul(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      if (this.isMeta) return Tensor.meta([...this.shape], this.requiresGrad);
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) result[i] = this.data[i] * other;
      return new Tensor(result, [...this.shape], this.requiresGrad);
    }
    return this.broadcastBinaryOp(other, (a, b) => a * b);
  }

  div(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      if (this.isMeta) return Tensor.meta([...this.shape], this.requiresGrad);
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) result[i] = this.data[i] / other;
      return new Tensor(result, [...this.shape], this.requiresGrad);
    }
    return this.broadcastBinaryOp(other, (a, b) => a / b);
  }

  neg(): Tensor {
    if (this.isMeta) return Tensor.meta([...this.shape], this.requiresGrad);
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = -this.data[i];
    return new Tensor(result, [...this.shape], this.requiresGrad);
  }

  exp(): Tensor {
    if (this.isMeta) return Tensor.meta([...this.shape], this.requiresGrad);
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = Math.exp(this.data[i]);
    return new Tensor(result, [...this.shape], this.requiresGrad);
  }

  log(): Tensor {
    if (this.isMeta) return Tensor.meta([...this.shape], this.requiresGrad);
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = Math.log(this.data[i]);
    return new Tensor(result, [...this.shape], this.requiresGrad);
  }

  abs(): Tensor {
    if (this.isMeta) return Tensor.meta([...this.shape], this.requiresGrad);
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = Math.abs(this.data[i]);
    return new Tensor(result, [...this.shape], this.requiresGrad);
  }

  gt(value: number): Tensor {
    if (this.isMeta) return Tensor.meta([...this.shape], false);
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = this.data[i] > value ? 1 : 0;
    return new Tensor(result, [...this.shape], false);
  }

  sum(axis?: number, keepDim = false): Tensor {
    if (axis === undefined) {
      if (this.isMeta) return Tensor.meta(keepDim ? new Array(this.ndim).fill(1) : [1], this.requiresGrad);
      let s = 0;
      for (let i = 0; i < this.size; i++) s += this.data[i];
      return keepDim ? Tensor.fromArray([s], new Array(this.ndim).fill(1), this.requiresGrad) : Tensor.fromArray([s], [1], this.requiresGrad);
    }
    if (axis < 0) axis += this.ndim;
    return this.reduceAxis(axis, keepDim, arr => arr.reduce((a, b) => a + b, 0));
  }

  max(axis?: number, keepDim = false): Tensor {
    if (axis === undefined) {
      if (this.isMeta) return Tensor.meta(keepDim ? new Array(this.ndim).fill(1) : [1], this.requiresGrad);
      let m = -Infinity;
      for (let i = 0; i < this.size; i++) if (this.data[i] > m) m = this.data[i];
      return keepDim ? Tensor.fromArray([m], new Array(this.ndim).fill(1), this.requiresGrad) : Tensor.fromArray([m], [1], this.requiresGrad);
    }
    if (axis < 0) axis += this.ndim;
    return this.reduceAxis(axis, keepDim, arr => arr.reduce((a, b) => Math.max(a, b), -Infinity));
  }

  mean(axis?: number, keepDim = false): Tensor {
    if (axis === undefined) {
      if (this.isMeta) return Tensor.meta(keepDim ? new Array(this.ndim).fill(1) : [1], this.requiresGrad);
      let s = 0;
      for (let i = 0; i < this.size; i++) s += this.data[i];
      const out = s / this.size;
      return keepDim ? Tensor.fromArray([out], new Array(this.ndim).fill(1), this.requiresGrad) : Tensor.fromArray([out], [1], this.requiresGrad);
    }
    if (axis < 0) axis += this.ndim;
    const n = this.shape[axis];
    return this.reduceAxis(axis, keepDim, arr => arr.reduce((a, b) => a + b, 0) / n);
  }

  matmul(other: Tensor): Tensor {
    if (this.ndim !== 2 || other.ndim !== 2) throw new Error('matmul requires 2D arrays');
    const [m, k] = this.shape;
    const [k2, n] = other.shape;
    if (k !== k2) throw new Error('matmul shape mismatch');
    if (this.isMeta || other.isMeta) return Tensor.meta([m, n], this.requiresGrad || other.requiresGrad);
    const result = Tensor.zeros([m, n], this.requiresGrad || other.requiresGrad);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let kk = 0; kk < k; kk++) {
          sum += this.data[i * k + kk] * other.data[kk * n + j];
        }
        result.data[i * n + j] = sum;
      }
    }
    return result;
  }

  allClose(other: Tensor, rtol = 1e-5): boolean {
    if (this.isMeta || other.isMeta) return this.shape.join(',') === other.shape.join(',');
    if (this.shape.length !== other.shape.length) return false;
    if (this.shape.some((dim, i) => dim !== other.shape[i])) return false;
    for (let i = 0; i < this.size; i++) {
      if (Math.abs(this.data[i] - other.data[i]) > rtol) return false;
    }
    return true;
  }

  toString(): string {
    if (this.isMeta) return `MetaTensor(shape=[${this.shape}], dtype=float32)`;
    if (this.ndim === 2 && this.shape[0] <= 8 && this.shape[1] <= 8) {
      const rows: string[] = [];
      for (let i = 0; i < this.shape[0]; i++) {
        const row = Array.from(this.data.slice(i * this.shape[1], (i + 1) * this.shape[1]));
        rows.push(`  [${row.map(v => Number(v.toFixed(4))).join(', ')}]`);
      }
      return `Tensor(shape=[${this.shape}]):\n[\n${rows.join('\n')}\n]`;
    }
    return `Tensor(shape=[${this.shape}], size=${this.size})`;
  }

  private reduceAxis(axis: number, keepDim: boolean, reducer: (arr: number[]) => number): Tensor {
    const outShape = this.shape.filter((_, i) => i !== axis);
    if (outShape.length === 0) outShape.push(1);
    const finalShape = keepDim
      ? this.shape.map((dim, i) => (i === axis ? 1 : dim))
      : outShape;
    if (this.isMeta) return Tensor.meta(finalShape, this.requiresGrad);

    const axisSize = this.shape[axis];
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const result = new Float32Array(finalShape.reduce((a, b) => a * b, 1));

    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        const values: number[] = [];
        for (let a = 0; a < axisSize; a++) {
          const idx = o * axisSize * inner + a * inner + i;
          values.push(this.data[idx]);
        }
        result[o * inner + i] = reducer(values);
      }
    }
    return new Tensor(result, finalShape, this.requiresGrad);
  }

  private broadcastBinaryOp(other: Tensor, fn: (a: number, b: number) => number): Tensor {
    const outShape = broadcastShape(this.shape, other.shape);
    if (this.isMeta || other.isMeta) return Tensor.meta(outShape, this.requiresGrad || other.requiresGrad);
    const result = Tensor.zeros(outShape, this.requiresGrad || other.requiresGrad);
    const aStrides = broadcastStrides(this.shape, outShape);
    const bStrides = broadcastStrides(other.shape, outShape);
    const outStrides = Tensor.computeStrides(outShape);
    const total = outShape.reduce((a, b) => a * b, 1);
    for (let flat = 0; flat < total; flat++) {
      let remaining = flat;
      let aIdx = 0;
      let bIdx = 0;
      for (let d = 0; d < outShape.length; d++) {
        const coord = Math.floor(remaining / outStrides[d]);
        remaining %= outStrides[d];
        aIdx += coord * aStrides[d];
        bIdx += coord * bStrides[d];
      }
      result.data[flat] = fn(this.data[aIdx], other.data[bIdx]);
    }
    return result;
  }
}

export class MetaTensor extends Tensor {
  constructor(shape: number[], requiresGrad = false) {
    super(new Float32Array(0), shape, requiresGrad, true);
  }
}

export type NestedNumberArray = number[] | NestedNumberArray[];

export function tensor(value: NestedNumberArray, requiresGrad = false): Tensor {
  return Tensor.fromNestedArray(value, requiresGrad);
}

export function zeros(shape: number[], requiresGrad = false): Tensor {
  return Tensor.zeros(shape, requiresGrad);
}

export function ones(shape: number[], requiresGrad = false): Tensor {
  return Tensor.ones(shape, requiresGrad);
}

export function full(shape: number[], value: number, requiresGrad = false): Tensor {
  return Tensor.full(shape, value, requiresGrad);
}

export function rand(shape: number[], requiresGrad = false): Tensor {
  return Tensor.rand(shape, requiresGrad);
}

export function randn(shape: number[], requiresGrad = false): Tensor {
  return Tensor.randn(shape, requiresGrad);
}

function flattenNestedArray(value: NestedNumberArray): { flat: number[]; shape: number[] } {
  if (!Array.isArray(value)) {
    throw new Error('tensor() expects an array or nested array');
  }
  const shape = inferShape(value);
  const flat: number[] = [];
  fillFlat(value, shape, flat, 0);
  return { flat, shape };
}

function inferShape(value: NestedNumberArray): number[] {
  if (!Array.isArray(value)) return [];
  const length = value.length;
  if (length === 0) return [0];
  if (!Array.isArray(value[0])) return [length];
  const childShape = inferShape(value[0] as NestedNumberArray);
  for (let i = 1; i < value.length; i++) {
    const nextShape = inferShape(value[i] as NestedNumberArray);
    if (nextShape.length !== childShape.length || nextShape.some((dim, idx) => dim !== childShape[idx])) {
      throw new Error('Cannot build tensor from ragged nested arrays');
    }
  }
  return [length, ...childShape];
}

function fillFlat(value: NestedNumberArray, shape: number[], flat: number[], depth: number): void {
  if (depth === shape.length - 1) {
    for (const item of value as number[]) flat.push(item);
    return;
  }
  for (const item of value as NestedNumberArray[]) {
    fillFlat(item, shape, flat, depth + 1);
  }
}

function broadcastShape(a: number[], b: number[]): number[] {
  const maxLen = Math.max(a.length, b.length);
  const result: number[] = [];
  for (let i = 0; i < maxLen; i++) {
    const da = i < a.length ? a[a.length - 1 - i] : 1;
    const db = i < b.length ? b[b.length - 1 - i] : 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`Cannot broadcast shapes ${a} and ${b}`);
    }
    result.unshift(Math.max(da, db));
  }
  return result;
}

function broadcastStrides(origShape: number[], targetShape: number[]): number[] {
  const origStrides = Tensor.computeStrides(origShape);
  const result = new Array(targetShape.length).fill(0);
  const offset = targetShape.length - origShape.length;
  for (let i = 0; i < origShape.length; i++) {
    result[i + offset] = origShape[i] === 1 ? 0 : origStrides[i];
  }
  return result;
}
