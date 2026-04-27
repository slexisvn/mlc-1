// ═══════════════════════════════════════════════════════════════
//  NDArray — N-dimensional array backed by Float32Array
//  Core data structure for the entire MLC framework
// ═══════════════════════════════════════════════════════════════

export class NDArray {
  data: Float32Array;
  shape: number[];
  strides: number[];

  constructor(data: Float32Array, shape: number[]) {
    this.data = data;
    this.shape = shape;
    this.strides = NDArray.computeStrides(shape);
  }

  // ─── Static Constructors ───

  static computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  static zeros(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    return new NDArray(new Float32Array(size), shape);
  }

  static ones(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(1);
    return new NDArray(data, shape);
  }

  static full(shape: number[], value: number): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(value);
    return new NDArray(data, shape);
  }

  static rand(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = Math.random();
    return new NDArray(data, shape);
  }

  static randn(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    // Box-Muller transform
    for (let i = 0; i < size; i += 2) {
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1));
      data[i] = r * Math.cos(2 * Math.PI * u2);
      if (i + 1 < size) data[i + 1] = r * Math.sin(2 * Math.PI * u2);
    }
    return new NDArray(data, shape);
  }

  static fromArray(arr: number[], shape?: number[]): NDArray {
    const data = new Float32Array(arr);
    return new NDArray(data, shape || [arr.length]);
  }

  // ─── Properties ───

  get size(): number { return this.data.length; }
  get ndim(): number { return this.shape.length; }

  // ─── Element Access ───

  private flatIndex(indices: number[]): number {
    let idx = 0;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * this.strides[i];
    }
    return idx;
  }

  get(indices: number[]): number {
    return this.data[this.flatIndex(indices)];
  }

  set(indices: number[], val: number): void {
    this.data[this.flatIndex(indices)] = val;
  }

  // ─── Shape Operations ───

  reshape(newShape: number[]): NDArray {
    const size = newShape.reduce((a, b) => a * b, 1);
    if (size !== this.size) throw new Error(`Cannot reshape ${this.shape} to ${newShape}`);
    return new NDArray(new Float32Array(this.data), newShape);
  }

  transpose(): NDArray {
    if (this.ndim !== 2) throw new Error('transpose() only for 2D');
    const [M, N] = this.shape;
    const result = NDArray.zeros([N, M]);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        result.data[j * M + i] = this.data[i * N + j];
      }
    }
    return result;
  }

  // ─── Arithmetic (element-wise) ───

  add(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) result[i] = this.data[i] + other;
      return new NDArray(result, [...this.shape]);
    }
    // Broadcasting support
    return this.broadcastBinaryOp(other, (a, b) => a + b);
  }

  sub(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) result[i] = this.data[i] - other;
      return new NDArray(result, [...this.shape]);
    }
    return this.broadcastBinaryOp(other, (a, b) => a - b);
  }

  mul(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) result[i] = this.data[i] * other;
      return new NDArray(result, [...this.shape]);
    }
    return this.broadcastBinaryOp(other, (a, b) => a * b);
  }

  div(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) result[i] = this.data[i] / other;
      return new NDArray(result, [...this.shape]);
    }
    return this.broadcastBinaryOp(other, (a, b) => a / b);
  }

  neg(): NDArray {
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = -this.data[i];
    return new NDArray(result, [...this.shape]);
  }

  // ─── Broadcasting binary op ───

  private broadcastBinaryOp(other: NDArray, fn: (a: number, b: number) => number): NDArray {
    const outShape = this.broadcastShape(this.shape, other.shape);
    const result = NDArray.zeros(outShape);
    const aStrides = this.broadcastStrides(this.shape, outShape);
    const bStrides = this.broadcastStrides(other.shape, outShape);

    const totalSize = outShape.reduce((a, b) => a * b, 1);
    const outStrides = NDArray.computeStrides(outShape);

    for (let flatIdx = 0; flatIdx < totalSize; flatIdx++) {
      // Compute multi-dimensional index
      let remaining = flatIdx;
      let aIdx = 0, bIdx = 0;
      for (let d = 0; d < outShape.length; d++) {
        const coord = Math.floor(remaining / outStrides[d]);
        remaining %= outStrides[d];
        aIdx += coord * aStrides[d];
        bIdx += coord * bStrides[d];
      }
      result.data[flatIdx] = fn(this.data[aIdx], other.data[bIdx]);
    }
    return result;
  }

  private broadcastShape(a: number[], b: number[]): number[] {
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

  private broadcastStrides(origShape: number[], targetShape: number[]): number[] {
    const origStrides = NDArray.computeStrides(origShape);
    const result = new Array(targetShape.length).fill(0);
    const offset = targetShape.length - origShape.length;
    for (let i = 0; i < origShape.length; i++) {
      result[i + offset] = origShape[i] === 1 ? 0 : origStrides[i];
    }
    return result;
  }

  // ─── Reduction ───

  sum(axis?: number, keepDim = false): NDArray {
    if (axis === undefined) {
      let s = 0;
      for (let i = 0; i < this.size; i++) s += this.data[i];
      return keepDim ? NDArray.fromArray([s], new Array(this.ndim).fill(1)) : NDArray.fromArray([s]);
    }
    if (axis < 0) axis += this.ndim;
    return this.reduceAxis(axis, keepDim, (arr) => {
      let s = 0;
      for (const v of arr) s += v;
      return s;
    });
  }

  max(axis?: number, keepDim = false): NDArray {
    if (axis === undefined) {
      let m = -Infinity;
      for (let i = 0; i < this.size; i++) if (this.data[i] > m) m = this.data[i];
      return keepDim ? NDArray.fromArray([m], new Array(this.ndim).fill(1)) : NDArray.fromArray([m]);
    }
    if (axis < 0) axis += this.ndim;
    return this.reduceAxis(axis, keepDim, (arr) => {
      let m = -Infinity;
      for (const v of arr) if (v > m) m = v;
      return m;
    });
  }

  mean(axis?: number, keepDim = false): NDArray {
    if (axis === undefined) {
      let s = 0;
      for (let i = 0; i < this.size; i++) s += this.data[i];
      return keepDim
        ? NDArray.fromArray([s / this.size], new Array(this.ndim).fill(1))
        : NDArray.fromArray([s / this.size]);
    }
    if (axis < 0) axis += this.ndim;
    const n = this.shape[axis];
    return this.reduceAxis(axis, keepDim, (arr) => {
      let s = 0;
      for (const v of arr) s += v;
      return s / n;
    });
  }

  private reduceAxis(axis: number, keepDim: boolean, reducer: (arr: number[]) => number): NDArray {
    const outShape = this.shape.filter((_, i) => i !== axis);
    if (outShape.length === 0) outShape.push(1);
    const axisSize = this.shape[axis];
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float32Array(outSize);

    const outerSize = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const innerSize = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);

    for (let o = 0; o < outerSize; o++) {
      for (let inner = 0; inner < innerSize; inner++) {
        const values: number[] = [];
        for (let a = 0; a < axisSize; a++) {
          values.push(this.data[o * axisSize * innerSize + a * innerSize + inner]);
        }
        result[o * innerSize + inner] = reducer(values);
      }
    }

    const finalShape = keepDim
      ? this.shape.map((s, i) => i === axis ? 1 : s)
      : outShape;
    return new NDArray(result, finalShape);
  }

  // ─── Math ───

  exp(): NDArray {
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = Math.exp(this.data[i]);
    return new NDArray(result, [...this.shape]);
  }

  log(): NDArray {
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = Math.log(this.data[i] + 1e-12);
    return new NDArray(result, [...this.shape]);
  }

  abs(): NDArray {
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = Math.abs(this.data[i]);
    return new NDArray(result, [...this.shape]);
  }

  // ─── Comparison ───

  gt(value: number): NDArray {
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) result[i] = this.data[i] > value ? 1 : 0;
    return new NDArray(result, [...this.shape]);
  }

  // ─── Matmul ───

  matmul(other: NDArray): NDArray {
    if (this.ndim !== 2 || other.ndim !== 2) throw new Error('matmul requires 2D arrays');
    const [M, K1] = this.shape;
    const [K2, N] = other.shape;
    if (K1 !== K2) throw new Error(`matmul shape mismatch: [${M},${K1}] @ [${K2},${N}]`);
    const result = NDArray.zeros([M, N]);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K1; k++) {
          sum += this.data[i * K1 + k] * other.data[k * N + j];
        }
        result.data[i * N + j] = sum;
      }
    }
    return result;
  }

  // ─── Utility ───

  clone(): NDArray {
    return new NDArray(new Float32Array(this.data), [...this.shape]);
  }

  allClose(other: NDArray, rtol = 1e-5): boolean {
    if (this.size !== other.size) return false;
    for (let i = 0; i < this.size; i++) {
      const diff = Math.abs(this.data[i] - other.data[i]);
      const scale = Math.max(Math.abs(this.data[i]), Math.abs(other.data[i]), 1e-8);
      if (diff / scale > rtol && diff > 1e-6) return false;
    }
    return true;
  }

  toString(): string {
    if (this.ndim === 1) {
      const vals = Array.from(this.data).map(v => v.toFixed(4)).join(', ');
      return `[${vals}]`;
    }
    if (this.ndim === 2) {
      const [M, N] = this.shape;
      const rows: string[] = [];
      for (let i = 0; i < Math.min(M, 4); i++) {
        const row = Array.from(this.data.slice(i * N, i * N + Math.min(N, 8)))
          .map(v => v.toFixed(4))
          .join(', ');
        rows.push(`  [${row}${N > 8 ? ', ...' : ''}]`);
      }
      if (M > 4) rows.push('  ...');
      return `NDArray(shape=[${this.shape}]):\n[\n${rows.join('\n')}\n]`;
    }
    return `NDArray(shape=[${this.shape}], size=${this.size})`;
  }
}
