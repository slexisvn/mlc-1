// ═══════════════════════════════════════════════════════════════
//  Runtime Executor — Executes compiled MLC code
//  Manages memory, runs forward/backward, benchmarks.
// ═══════════════════════════════════════════════════════════════

import { NDArray } from '../tensor/ndarray.js';
import { PrimFunc } from '../ir/low_level.js';
import { compile, compileRegTile } from '../codegen/js_codegen.js';

export interface MemoryPlan {
  buffers: Map<string, { offset: number; size: number; shape: number[] }>;
  totalBytes: number;
}

export interface BenchResult {
  mean: number;
  median: number;
  min: number;
  max: number;
  iterations: number;
}

export class RuntimeModule {
  private functions: Map<string, Function> = new Map();
  private primFuncs: PrimFunc[] = [];
  private params: Map<string, NDArray>;

  constructor(primFuncs: PrimFunc[], params: Map<string, NDArray>, useRegTile = false) {
    this.primFuncs = primFuncs;
    this.params = params;

    // Compile all PrimFuncs
    for (const pf of primFuncs) {
      try {
        const fn = useRegTile ? compileRegTile(pf) : compile(pf);
        this.functions.set(pf.name, fn);
      } catch (e) {
        console.error(`Failed to compile ${pf.name}:`, e);
      }
    }
  }

  // Run forward inference for a 2-layer classifier
  forward(input: NDArray): NDArray {
    const paramEntries = [...this.params.values()];

    // Find functions by prefix  
    const findFn = (prefix: string): Function | undefined => {
      for (const [name, fn] of this.functions) {
        if (name === prefix || name.startsWith(prefix)) return fn;
      }
      return undefined;
    };

    // Layer 1: fused_dense_bias_relu (or similar activation)
    const layer1Fn = findFn('fused_dense_bias_relu')
      || findFn('fused_dense_bias_sigmoid')
      || findFn('fused_dense_bias_tanh')
      || findFn('fused_dense_bias');

    if (!layer1Fn || paramEntries.length < 2) {
      throw new Error('Cannot run forward: missing compiled functions or params');
    }

    const W1 = paramEntries[0]; // weight [out, in]
    const b1 = paramEntries[1]; // bias [1, out]
    const hiddenSize = W1.shape[0];
    const hidden = new Float32Array(input.shape[0] * hiddenSize);

    layer1Fn(input.data, W1.data, b1.data, hidden);

    // Layer 2: fused_dense_bias (find the second one, not the first)
    if (paramEntries.length < 4) {
      return new NDArray(hidden, [input.shape[0], hiddenSize]);
    }

    // Find layer 2 function — it's the second fused_dense_bias or a standalone
    let layer2Fn: Function | undefined;
    const funcNames = [...this.functions.keys()];
    for (const name of funcNames) {
      if (name.startsWith('fused_dense_bias') 
          && !name.includes('relu') 
          && !name.includes('sigmoid')
          && !name.includes('tanh')) {
        // Pick the LAST one (layer 2)
        layer2Fn = this.functions.get(name);
      }
    }

    if (!layer2Fn) {
      return new NDArray(hidden, [input.shape[0], hiddenSize]);
    }

    const W2 = paramEntries[2]; 
    const b2 = paramEntries[3];
    const outSize = W2.shape[0];
    const output = new Float32Array(input.shape[0] * outSize);

    layer2Fn(hidden, W2.data, b2.data, output);

    return new NDArray(output, [input.shape[0], outSize]);
  }

  // Run a single function by name
  runFunction(name: string, ...buffers: Float32Array[]): void {
    const fn = this.functions.get(name);
    if (!fn) throw new Error(`Function '${name}' not found`);
    fn(...buffers);
  }

  // Benchmark forward pass
  benchmarkForward(input: NDArray, iterations: number): BenchResult {
    // Warmup
    for (let i = 0; i < 5; i++) {
      try { this.forward(input); } catch { break; }
    }

    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      this.forward(input);
      const end = performance.now();
      times.push(end - start);
    }

    return this.computeStats(times, iterations);
  }

  // Benchmark a specific compiled function
  benchmarkFunction(name: string, buffers: Float32Array[], iterations: number): BenchResult {
    const fn = this.functions.get(name);
    if (!fn) throw new Error(`Function '${name}' not found`);

    // Warmup
    for (let i = 0; i < 5; i++) fn(...buffers);

    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      fn(...buffers);
      const end = performance.now();
      times.push(end - start);
    }

    return this.computeStats(times, iterations);
  }

  private computeStats(times: number[], iterations: number): BenchResult {
    times.sort((a, b) => a - b);
    const sum = times.reduce((a, b) => a + b, 0);
    return {
      mean: sum / times.length,
      median: times[Math.floor(times.length / 2)],
      min: times[0],
      max: times[times.length - 1],
      iterations,
    };
  }

  // List compiled functions
  listFunctions(): string[] {
    return [...this.functions.keys()];
  }
}

// ─── Naive forward for verification ───

export function naiveForward(
  input: NDArray,
  params: NDArray[],
  activation: 'relu' | 'sigmoid' | 'tanh' | 'leaky_relu' | 'none' = 'relu'
): NDArray {
  // params = [W1, b1, W2, b2, ...]
  let x = input;

  for (let i = 0; i < params.length; i += 2) {
    const W = params[i];   // [out, in]
    const b = params[i + 1]; // [1, out]

    // Matmul: x @ W^T
    x = x.matmul(W.transpose());

    // Bias add
    x = x.add(b);

    // Apply activation (except on last layer)
    if (i + 2 < params.length) {
      switch (activation) {
        case 'relu':
          x = new NDArray(
            new Float32Array(x.data.map(v => Math.max(v, 0))),
            [...x.shape]
          );
          break;
        case 'sigmoid':
          x = new NDArray(
            new Float32Array(x.data.map(v => 1 / (1 + Math.exp(-v)))),
            [...x.shape]
          );
          break;
        case 'tanh':
          x = new NDArray(
            new Float32Array(x.data.map(v => Math.tanh(v))),
            [...x.shape]
          );
          break;
        case 'leaky_relu':
          x = new NDArray(
            new Float32Array(x.data.map(v => v > 0 ? v : 0.01 * v)),
            [...x.shape]
          );
          break;
        case 'none':
          break;
      }
    }
  }

  return x;
}
