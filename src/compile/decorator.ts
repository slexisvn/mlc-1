// ═══════════════════════════════════════════════════════════════
//  MLC Compile Decorator
//
//  Wraps any Module with the full MLC inference pipeline:
//    traceInference → buildIR → constantFold → fuseOps
//    → deadCodeElimination → cseModule
//    → lowerModule → arithmeticSimplify → storageRewrite
//    → RuntimeModule
//
//  API:
//    // Function-style (eager if inputShape given, lazy otherwise)
//    const net = mlcCompile(model, { inputShape: [4, 32] });
//    const out  = net.forward(input);
//
//    // Class decorator (always lazy — shape inferred from first forward())
//    @compile({ inputShape: [4, 32] })
//    class Classifier extends Sequential { ... }
// ═══════════════════════════════════════════════════════════════

import { NDArray } from '../tensor/ndarray.js';
import { GradTensor } from '../autograd/engine.js';
import { type Module } from '../model/nn.js';
import { Tracer } from '../trace/tracer.js';
import { buildIR } from '../ir/high_level.js';
import { constantFold } from '../transform/constant_fold.js';
import { fuseOps } from '../transform/op_fusion.js';
import { deadCodeElimination } from '../transform/dead_code_elimination.js';
import { cseModule } from '../transform/cse.js';
import { lowerModule } from '../lower/lowering.js';
import { arithmeticSimplify } from '../transform/arithmetic_simplify.js';
import { storageRewrite } from '../transform/storage_rewrite.js';
import { RuntimeModule } from '../runtime/executor.js';
import { PrimFunc } from '../ir/low_level.js';

// ─── Public Interfaces ────────────────────────────────────────

export interface CompileOptions {
  /** Provide to compile eagerly before any forward() call. */
  inputShape?: number[];
  /** Enable register-tiling in the JS codegen. Default: true. */
  useRegTile?: boolean;
  /** Print pipeline stats (kernel names, timing) after compilation. Default: false. */
  verbose?: boolean;
}

export interface CompileStats {
  inputShape: number[];
  numKernels: number;
  kernelNames: string[];
  /** Wall-clock time spent in the MLC pipeline (ms). */
  compilationMs: number;
  /** Date.now() when compilation finished. */
  compiledAt: number;
}

// ─── CompiledModule ───────────────────────────────────────────

export class CompiledModule {
  private _model: Module;
  private _opts: Required<CompileOptions>;
  private _runtime: RuntimeModule | null = null;
  private _stats: CompileStats | null = null;
  /** PrimFuncs produced by the last compilation (exposed for inspection). */
  primFuncs: PrimFunc[] = [];

  constructor(model: Module, opts: CompileOptions = {}) {
    this._model = model;
    this._opts = {
      inputShape: opts.inputShape ?? [],
      useRegTile: opts.useRegTile ?? true,
      verbose:    opts.verbose    ?? false,
    };

    // Eager: compile now if inputShape is provided
    if (this._opts.inputShape.length > 0) {
      this._compile(this._opts.inputShape);
    }
  }

  // ── Public API ──────────────────────────────────────────────

  /**
   * Run compiled forward pass.
   * On the first call of a lazy-compiled module the MLC pipeline
   * runs automatically using the shape of `input`.
   */
  forward(input: NDArray | GradTensor): NDArray {
    const nd = input instanceof GradTensor ? input.data : input;

    if (!this._runtime) {
      this._compile(nd.shape);
    }

    return this._runtime!.forward(nd);
  }

  /** Returns compilation metadata. Null if not yet compiled (lazy, pre-first-call). */
  getStats(): CompileStats | null {
    return this._stats;
  }

  /**
   * Force re-trace and recompile with a new input shape.
   * Needed when batch size or input dimensions change.
   */
  recompile(inputShape: number[]): void {
    this._runtime = null;
    this._stats   = null;
    this.primFuncs = [];
    this._compile(inputShape);
  }

  // ── Private ──────────────────────────────────────────────────

  private _compile(inputShape: number[]): void {
    const t0 = performance.now();

    // 1. Trace inference graph
    const tracer   = new Tracer();
    const graph    = tracer.traceInference(this._model, inputShape);

    // 2. High-level IR + graph-level optimizations
    const irMod    = buildIR(graph);
    const folded   = constantFold(irMod);
    const fused    = fuseOps(folded);
    const { module: dceMod } = deadCodeElimination(fused);
    const { module: cseMod } = cseModule(dceMod);

    // 3. Lower to TensorIR (loop nests)
    const primFuncsRaw = lowerModule(cseMod);

    // 4. TensorIR passes
    //    arithmeticSimplify: safe always
    //    storageRewrite: safe for JS backend (promotes size-1 buffers to scalars)
    const optimized = primFuncsRaw.map(pf =>
      storageRewrite(arithmeticSimplify(pf)).func
    );

    this.primFuncs = optimized;

    // 5. Build params map: param_0 → W0, param_1 → B0, ...
    const params    = this._model.parameters();
    const paramsMap = new Map<string, NDArray>();
    params.forEach((p, i) => paramsMap.set(`param_${i}`, p.data));

    // 6. Instantiate RuntimeModule
    this._runtime = new RuntimeModule(optimized, paramsMap, this._opts.useRegTile);

    const compilationMs = performance.now() - t0;

    this._stats = {
      inputShape:    inputShape.slice(),
      numKernels:    optimized.length,
      kernelNames:   optimized.map(pf => pf.name),
      compilationMs,
      compiledAt:    Date.now(),
    };

    if (this._opts.verbose) {
      console.log(`[mlcCompile] Compiled in ${compilationMs.toFixed(1)}ms`);
      console.log(`  inputShape : [${inputShape.join(', ')}]`);
      console.log(`  kernels    : ${optimized.length}`);
      for (const pf of optimized) {
        const sig = pf.params.map(p => `${p.name}[${p.shape.join('×')}]`).join(', ');
        console.log(`    ${pf.name}(${sig})`);
      }
    }
  }
}

// ─── mlcCompile() ────────────────────────────────────────────

/**
 * Wrap a Module with the MLC inference pipeline.
 *
 * @param model   Any Module (Sequential, custom class, etc.)
 * @param opts    Optional compile options
 * @returns       CompiledModule — call .forward(input) to run
 *
 * @example
 *   const net = mlcCompile(model, { inputShape: [4, 32] });
 *   const out = net.forward(input);  // runs compiled kernels
 */
export function mlcCompile(model: Module, opts: CompileOptions = {}): CompiledModule {
  return new CompiledModule(model, opts);
}

// ─── @compile class decorator (Stage 3 TS5) ──────────────────

type Constructor<T = object> = new (...args: any[]) => T;

/**
 * Class decorator that transparently replaces `forward()` with the
 * MLC-compiled version. Compilation is always lazy (triggered on the
 * first `forward()` call so the input shape is known).
 *
 * @example
 *   @compile({ verbose: true })
 *   class Classifier extends Sequential { ... }
 *
 *   const net = new Classifier([new Linear(32, 64), new ReLU(), new Linear(64, 8)]);
 *   const out = net.forward(input);  // compiles on first call
 */
export function compile(opts: CompileOptions = {}): <T extends Constructor<Module>>(Base: T) => T {
  return function <T extends Constructor<Module>>(Base: T): T {
    return class extends Base {
      private _compiled: CompiledModule | null = null;

      override forward(x: GradTensor): GradTensor {
        const nd = x instanceof GradTensor ? x.data : x as unknown as NDArray;

        if (!this._compiled) {
          // Lazy: infer shape from the first real input.
          // IMPORTANT: pass a shim whose forward() delegates to Base.prototype.forward,
          // NOT to this.forward (the patched version). Without this, CompiledModule's
          // tracer calls model.forward() → hits this override again → infinite recursion.
          const self = this;
          const baseShim: Module = {
            forward(inp: GradTensor): GradTensor {
              return Base.prototype.forward.call(self, inp);
            },
            parameters(): GradTensor[] {
              return self.parameters();
            },
          };
          this._compiled = new CompiledModule(baseShim, {
            ...opts,
            inputShape: nd.shape,
          });
        }

        const out = this._compiled.forward(nd);
        // Wrap NDArray back into a GradTensor (no grad required — inference only)
        return new GradTensor(out, false);
      }

      /** Access compile stats (null before first forward()). */
      getCompileStats(): CompileStats | null {
        return this._compiled?.getStats() ?? null;
      }
    } as T;
  };
}
