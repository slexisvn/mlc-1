// ═══════════════════════════════════════════════════════════════
//  Runtime Executor — Executes compiled MLC code
//  Supports JavaScript and WebAssembly backends for inference.
// ═══════════════════════════════════════════════════════════════

import { codegenWAT, type WATKernelInfo } from '../codegen/wat_codegen.js';
import { compile, compileRegTile } from '../codegen/js_codegen.js';
import { PrimFunc } from '../ir/low_level.js';
import { Tensor } from '../tensor/tensor.js';

const wabtPkg = await import('wabt');
const wabtFactory = (wabtPkg as { default: () => Promise<any> }).default;
const wabt = await wabtFactory();

export type RuntimeBackend = 'js' | 'wat';

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

interface WasmKernelPlan {
  name: string;
  fn: CallableFunction;
  inputOffset: number;
  weightOffset: number;
  biasOffset: number;
  outputOffset: number;
  outputShape: number[];
  mode: WATKernelInfo['mode'];
  weightLayout: WATKernelInfo['weightLayout'];
  packedWeightBytes: number;
  originalWeightBytes: number;
}

export interface WasmRuntimeDebugInfo {
  paramsUploadedOnce: boolean;
  paramBytes: number;
  activationBytes: number;
  totalBytes: number;
  totalPages: number;
  kernels: Array<{
    name: string;
    mode: WATKernelInfo['mode'];
    weightLayout: WATKernelInfo['weightLayout'];
    inputOffset: number;
    weightOffset: number;
    biasOffset: number;
    outputOffset: number;
    outputShape: number[];
    originalWeightBytes: number;
    packedWeightBytes: number;
  }>;
}

export class RuntimeModule {
  private compiledFunctions: Function[] = [];
  private primFuncs: PrimFunc[] = [];
  private params: Map<string, Tensor>;
  private paramEntries: Tensor[];
  private backend: RuntimeBackend;
  private wasmInstance: WebAssembly.Instance | null = null;
  private wasmMemory: WebAssembly.Memory | null = null;
  private wasmMemoryView: Float32Array | null = null;
  private wasmKernelPlans: WasmKernelPlan[] = [];
  private wasmDebugInfo: WasmRuntimeDebugInfo | null = null;

  constructor(
    primFuncs: PrimFunc[],
    params: Map<string, Tensor>,
    backend: RuntimeBackend = 'js',
    useRegTile = false,
  ) {
    this.primFuncs = primFuncs;
    this.params = params;
    this.paramEntries = [...params.values()];
    this.backend = backend;

    if (backend === 'wat') {
      this.initializeWasmRuntime();
      return;
    }

    for (const pf of primFuncs) {
      try {
        const fn = useRegTile ? compileRegTile(pf) : compile(pf);
        this.compiledFunctions.push(fn);
      } catch (e) {
        console.error(`Failed to compile ${pf.name}:`, e);
        this.compiledFunctions.push(() => {
          throw new Error(`Compiled kernel '${pf.name}' is unavailable`);
        });
      }
    }
  }

  forward(input: Tensor): Tensor {
    if (this.backend === 'wat') {
      return this.forwardWasm(input);
    }
    return this.forwardJs(input);
  }

  runFunction(name: string, ...buffers: Float32Array[]): void {
    const index = this.primFuncs.findIndex(pf => pf.name === name);
    if (index === -1) throw new Error(`Function '${name}' not found`);
    const fn = this.compiledFunctions[index];
    fn(...buffers);
  }

  benchmarkForward(input: Tensor, iterations: number): BenchResult {
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

  benchmarkFunction(name: string, buffers: Float32Array[], iterations: number): BenchResult {
    const index = this.primFuncs.findIndex(pf => pf.name === name);
    if (index === -1) throw new Error(`Function '${name}' not found`);
    const fn = this.compiledFunctions[index];

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

  listFunctions(): string[] {
    if (this.backend === 'wat' && this.wasmInstance) {
      return this.primFuncs.map(pf => pf.name);
    }
    return this.primFuncs.map(pf => pf.name);
  }

  getWasmDebugInfo(): WasmRuntimeDebugInfo | null {
    return this.wasmDebugInfo;
  }

  private forwardJs(input: Tensor): Tensor {
    if (this.compiledFunctions.length === 0 || this.paramEntries.length < 2) {
      throw new Error('Cannot run forward: missing compiled functions or params');
    }
    if (this.paramEntries.length < this.compiledFunctions.length * 2) {
      throw new Error('Cannot run forward: parameter count does not match compiled kernels');
    }

    let current = input.data;
    let currentShape = [...input.shape];

    for (let i = 0; i < this.compiledFunctions.length; i++) {
      const weight = this.paramEntries[i * 2];
      const bias = this.paramEntries[i * 2 + 1];
      if (!weight || !bias) {
        throw new Error(`Missing parameters for compiled kernel ${i}`);
      }

      const outParam = this.primFuncs[i].params.find(param => param.name === 'Out');
      if (!outParam) {
        throw new Error(`Missing Out buffer for compiled kernel ${this.primFuncs[i].name}`);
      }

      const output = new Float32Array(outParam.shape.reduce((a, b) => a * b, 1));
      this.compiledFunctions[i](current, weight.data, bias.data, output);
      current = output;
      currentShape = [...outParam.shape];
    }

    return new Tensor(current, currentShape);
  }

  private initializeWasmRuntime(): void {
    const watModule = codegenWAT(this.primFuncs);
    const wasmSrc = wabt.parseWat('runtime.wat', watModule.text);
    const { buffer } = wasmSrc.toBinary({});
    wasmSrc.destroy();

    const module = new WebAssembly.Module(buffer);
    this.wasmInstance = new WebAssembly.Instance(module, {});
    this.wasmMemory = this.wasmInstance.exports.memory as WebAssembly.Memory;
    this.wasmKernelPlans = this.buildWasmKernelPlans(watModule.kernels);
    const totalBytes = this.wasmKernelPlans.reduce((maxBytes, plan) => {
      const outputBytes = plan.outputShape.reduce((a, b) => a * b, 1) * 4;
      return Math.max(maxBytes, plan.outputOffset + outputBytes);
    }, 0);
    this.ensureWasmMemoryCapacity(totalBytes);
    this.wasmMemoryView = new Float32Array(this.wasmMemory.buffer);
    this.preloadWasmParams();
    this.wasmDebugInfo = {
      paramsUploadedOnce: true,
      paramBytes: this.wasmKernelPlans.reduce((sum, plan) => sum + plan.packedWeightBytes + this.biasBytes(plan), 0),
      activationBytes: this.computeActivationBytes(),
      totalBytes,
      totalPages: Math.ceil(totalBytes / 65536),
      kernels: this.wasmKernelPlans.map(plan => ({
        name: plan.name,
        mode: plan.mode,
        weightLayout: plan.weightLayout,
        inputOffset: plan.inputOffset,
        weightOffset: plan.weightOffset,
        biasOffset: plan.biasOffset,
        outputOffset: plan.outputOffset,
        outputShape: [...plan.outputShape],
        originalWeightBytes: plan.originalWeightBytes,
        packedWeightBytes: plan.packedWeightBytes,
      })),
    };
  }

  private forwardWasm(input: Tensor): Tensor {
    if (!this.wasmInstance || !this.wasmMemoryView || !this.wasmMemory) {
      throw new Error('WAT runtime is not initialized');
    }
    if (this.paramEntries.length < 2 || this.primFuncs.length === 0) {
      throw new Error('Cannot run WAT forward: missing params or kernels');
    }
    if (this.wasmKernelPlans.length === 0) {
      throw new Error('Cannot run WAT forward: missing kernel execution plan');
    }

    const firstInputOffset = this.wasmKernelPlans[0].inputOffset / 4;
    this.wasmMemoryView.set(input.data, firstInputOffset);

    for (const plan of this.wasmKernelPlans) {
      plan.fn(plan.inputOffset, plan.weightOffset, plan.biasOffset, plan.outputOffset);
    }

    const lastPlan = this.wasmKernelPlans[this.wasmKernelPlans.length - 1];
    const outputLength = lastPlan.outputShape.reduce((a, b) => a * b, 1);
    const start = lastPlan.outputOffset / 4;
    const output = this.wasmMemoryView.slice(start, start + outputLength);
    return new Tensor(output, [...lastPlan.outputShape]);
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

  private ensureWasmMemoryCapacity(requiredBytes: number): void {
    if (!this.wasmMemory) return;

    const currentBytes = this.wasmMemory.buffer.byteLength;
    if (requiredBytes <= currentBytes) return;

    const pageBytes = 65536;
    const missingPages = Math.ceil((requiredBytes - currentBytes) / pageBytes);
    this.wasmMemory.grow(missingPages);
  }

  private buildWasmKernelPlans(kernelInfos: WATKernelInfo[]): WasmKernelPlan[] {
    if (!this.wasmInstance) {
      throw new Error('WAT runtime requires an instantiated WebAssembly module');
    }
    if (this.paramEntries.length < this.primFuncs.length * 2) {
      throw new Error('Cannot build WAT kernel plans: parameter count does not match compiled kernels');
    }

    const plans: WasmKernelPlan[] = [];
    let cursor = 0;
    const align = (value: number, boundary = 16) => Math.ceil(value / boundary) * boundary;

    for (let i = 0; i < this.primFuncs.length; i++) {
      const pf = this.primFuncs[i];
      const kernelInfo = kernelInfos[i];
      const weight = this.paramEntries[i * 2];
      const bias = this.paramEntries[i * 2 + 1];
      const outParam = pf.params.find(param => param.name === 'Out');
      if (!weight || !bias || !outParam) {
        throw new Error(`Missing WAT kernel resources for ${pf.name}`);
      }

      const fn = this.wasmInstance.exports[pf.name] as CallableFunction | undefined;
      if (!fn) {
        throw new Error(`WAT export '${pf.name}' not found`);
      }

      const inputOffset = i === 0 ? align(cursor) : plans[i - 1].outputOffset;
      if (i === 0) {
        cursor = inputOffset + pf.params.find(param => param.name === 'A')!.shape.reduce((a, b) => a * b, 1) * 4;
      } else {
        cursor = plans[i - 1].outputOffset + plans[i - 1].outputShape.reduce((a, b) => a * b, 1) * 4;
      }

      const packedWeight = kernelInfo.mode === 'simd-f32x4'
        ? packWeightForWasmSimd(weight)
        : weight.data;
      const originalWeightBytes = weight.data.length * 4;
      const packedWeightBytes = packedWeight.length * 4;
      const weightOffset = align(cursor);
      cursor = weightOffset + packedWeightBytes;
      const biasOffset = align(cursor);
      cursor = biasOffset + bias.data.length * 4;
      const outputOffset = align(cursor);
      cursor = outputOffset + outParam.shape.reduce((a, b) => a * b, 1) * 4;

      plans.push({
        name: pf.name,
        fn,
        inputOffset,
        weightOffset,
        biasOffset,
        outputOffset,
        outputShape: [...outParam.shape],
        mode: kernelInfo.mode,
        weightLayout: kernelInfo.weightLayout,
        packedWeightBytes,
        originalWeightBytes,
      });
    }

    return plans;
  }

  private preloadWasmParams(): void {
    if (!this.wasmMemoryView) return;

    for (let i = 0; i < this.wasmKernelPlans.length; i++) {
      const plan = this.wasmKernelPlans[i];
      const weight = this.paramEntries[i * 2];
      const bias = this.paramEntries[i * 2 + 1];
      const packedWeight = plan.mode === 'simd-f32x4'
        ? packWeightForWasmSimd(weight)
        : weight.data;

      this.wasmMemoryView.set(packedWeight, plan.weightOffset / 4);
      this.wasmMemoryView.set(bias.data, plan.biasOffset / 4);
    }
  }

  private computeActivationBytes(): number {
    if (this.wasmKernelPlans.length === 0) return 0;
    const firstInputBytes = this.wasmKernelPlans[0].inputOffset === 0
      ? this.primFuncs[0].params.find(param => param.name === 'A')!.shape.reduce((a, b) => a * b, 1) * 4
      : 0;
    return firstInputBytes + this.wasmKernelPlans.reduce(
      (sum, plan) => sum + plan.outputShape.reduce((a, b) => a * b, 1) * 4,
      0,
    );
  }

  private biasBytes(plan: WasmKernelPlan): number {
    const kernelIndex = this.wasmKernelPlans.indexOf(plan);
    return this.paramEntries[kernelIndex * 2 + 1].data.length * 4;
  }
}

function packWeightForWasmSimd(weight: Tensor): Float32Array {
  const [N, K] = weight.shape;
  const blocks = Math.ceil(N / 4);
  const packed = new Float32Array(blocks * K * 4);

  for (let block = 0; block < blocks; block++) {
    for (let k = 0; k < K; k++) {
      for (let lane = 0; lane < 4; lane++) {
        const j = block * 4 + lane;
        const dst = (block * K + k) * 4 + lane;
        packed[dst] = j < N ? weight.data[j * K + k] : 0;
      }
    }
  }

  return packed;
}
