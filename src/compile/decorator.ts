// ═══════════════════════════════════════════════════════════════
//  MLC Compile Decorator
//
//  Wraps any Module with the forward-only MLC inference pipeline:
//    traceInference → buildIR → constantFold → cseModule (analysis)
//    → deadCodeElimination → fuseOps
//    → lowerModule → arithmeticSimplify → storageRewrite
//    → RuntimeModule
// ═══════════════════════════════════════════════════════════════

import { Tensor } from '../tensor/tensor.js';
import { type Module } from '../model/nn.js';
import { Tracer, type TraceGraph } from '../trace/tracer.js';
import { buildIR, type Expr } from '../ir/high_level.js';
import { inferModuleShapes } from '../transform/shape_infer.js';
import { constantFold } from '../transform/constant_fold.js';
import { fuseOps, fusionStats } from '../transform/op_fusion.js';
import { deadCodeElimination } from '../transform/dead_code_elimination.js';
import { cseModule } from '../transform/cse.js';
import { lowerModule } from '../lower/lowering.js';
import { arithmeticSimplify } from '../transform/arithmetic_simplify.js';
import { storageRewrite } from '../transform/storage_rewrite.js';
import { RuntimeModule } from '../runtime/executor.js';
import { PrimFunc } from '../ir/low_level.js';
import type { RuntimeBackend } from '../runtime/executor.js';
import { printPhaseBanner, printSubSection, printHighLevelIR, printTIR } from '../utils/printer.js';
import { printVerifyResult, verifyHighLevelIR, verifyLowLevelIR } from '../transform/verifier.js';
import { codegenWAT } from '../codegen/wat_codegen.js';

export interface CompileOptions {
  inputShape?: number[]; 
  useRegTile?: boolean;
  verbose?: boolean;
  backend?: RuntimeBackend;
}

interface SharedCompileArtifacts {
  graph: TraceGraph;
  simplifiedPrimFuncs: PrimFunc[];
  runtimeParamNames: string[];
}

const MODEL_COMPILE_CACHE_ID = Symbol('mlcCompileCacheId');
const sharedCompileCache = new Map<string, Map<string, SharedCompileArtifacts>>();
let nextModelCompileCacheId = 0;

function traceCompilerPhase(phase: number, title: string): void {
  printPhaseBanner(phase, title);
}

function logVerification(label: string, result: ReturnType<typeof verifyHighLevelIR> | ReturnType<typeof verifyLowLevelIR>): void {
  console.log(printVerifyResult(label, result));
}

function optimizeIRToPrimFuncs(
  irMod: ReturnType<typeof buildIR>,
  verbose: boolean,
): { primFuncs: PrimFunc[]; runtimeParamNames: string[] } {
  if (verbose) {
    traceCompilerPhase(2, 'High-Level IR Construction');
    console.log(printHighLevelIR(irMod));
    printSubSection('2.1 Shape Inference');
  }

  const shapeResult = inferModuleShapes(irMod);
  if (verbose) {
    console.log(shapeResult.table);
    console.log(`  Inferred: ${shapeResult.inferred}/${shapeResult.totalOps} ops\n`);
    logVerification('Phase 2 high-level IR', verifyHighLevelIR(irMod));
    traceCompilerPhase(3, 'Graph-Level Optimizations');
    printSubSection('3.1 Constant Folding');
  }

  const folded = constantFold(irMod);
  if (verbose) {
    console.log('  Applied constant folding pass');
    printSubSection('3.2 Common Subexpression Elimination (CSE)');
  }

  const { stats: cseStats } = cseModule(folded);
  if (verbose) {
    console.log(`  Checked: ${cseStats.checked} nodes`);
    console.log(`  Replaced: ${cseStats.replaced} duplicate(s)`);
    console.log(`  LetExpr bindings created: ${cseStats.bindings} (IR linearized)`);
    console.log('  Note: CSE is reported from the tree IR; lowering still continues from the dead-code-pruned graph to keep fusion stable.');
    printSubSection('3.3 Dead Code Elimination');
  }

  const { module: dceMod, stats: dceStats } = deadCodeElimination(folded);
  if (verbose) {
    console.log(`  Before: ${dceStats.totalBefore} ops`);
    console.log(`  After:  ${dceStats.totalAfter} ops`);
    console.log(`  Eliminated: ${dceStats.eliminated} dead node(s)`);
    printSubSection('3.4 Operator Fusion');
  }

  const fused = fuseOps(dceMod);
  if (verbose) {
    console.log('\n  After fusion:');
    console.log(printHighLevelIR(fused));
    const stats = fusionStats(fused);
    console.log(`  Summary: ${stats.totalOps} total ops, ${stats.fusedOps} fused`);
    for (const group of stats.fusedGroups) {
      console.log(`    ${group}`);
    }
    logVerification('Phase 3 high-level IR', verifyHighLevelIR(fused));
    traceCompilerPhase(4, 'Operator Lowering (→ TensorIR / Loop Nests)');
  }

  const primFuncsRaw = lowerModule(fused);
  if (verbose) {
    console.log(`  Lowered ${primFuncsRaw.length} PrimFunc(s):\n`);
    for (const pf of primFuncsRaw) {
      console.log(printTIR(pf));
      console.log('');
    }
    logVerification('Phase 4 low-level IR', verifyLowLevelIR(primFuncsRaw));
  }

  const simplified = primFuncsRaw.map((pf) => {
    return arithmeticSimplify(pf);
  });

  const uniquified = uniquifyPrimFuncNames(simplified);
  return {
    primFuncs: uniquified,
    runtimeParamNames: collectRuntimeParamNames(fused.getFunction('main')?.body),
  };
}

function collectRuntimeParamNames(expr: Expr | undefined): string[] {
  if (!expr) return [];

  const names: string[] = [];

  function visit(node: Expr): void {
    if (node.kind === 'constant') {
      if (node.name.startsWith('param_')) names.push(node.name);
      return;
    }
    if (node.kind === 'var') return;
    if (node.kind === 'let') {
      visit(node.body);
      return;
    }
    for (const arg of node.args) visit(arg);
  }

  visit(expr);
  return names;
}

function uniquifyPrimFuncNames(primFuncs: PrimFunc[]): PrimFunc[] {
  const counts = new Map<string, number>();
  return primFuncs.map((pf) => {
    const nextCount = counts.get(pf.name) ?? 0;
    counts.set(pf.name, nextCount + 1);
    if (nextCount === 0) return pf;
    const cloned = pf.clone();
    cloned.name = `${pf.name}_${nextCount}`;
    return cloned;
  });
}

function compileBackendPrimFuncs(
  simplifiedPrimFuncs: PrimFunc[],
  backend: RuntimeBackend,
  verbose: boolean,
  reusedFrontend: boolean,
): PrimFunc[] {
  if (verbose) {
    traceCompilerPhase(5, 'TensorIR Passes');
    printSubSection('5.1 Arithmetic Simplification');
    console.log(reusedFrontend
      ? '  Reused cached arithmetic-simplified PrimFunc(s) from the previous compile.'
      : '  Arithmetic simplification already applied during this compile.');
  }

  if (backend === 'wat') {
    if (verbose) {
      printSubSection('5.2 Storage Rewrite');
      console.log('  WAT backend: skipped storage rewrite to preserve the current WAT runtime pipeline.');
      printSubSection('5.3 WebAssembly Text (WAT)');
      const watModule = codegenWAT(simplifiedPrimFuncs);
      const pages = Math.ceil(watModule.totalBytes / 65536);
      console.log(`  WAT module: ${simplifiedPrimFuncs.length} kernel(s), ${pages} page(s) of linear memory`);
      console.log(`  Exports: ${watModule.exports.join(', ')}`);
      for (const kernel of watModule.kernels) {
        console.log(`  ${kernel.name}: ${kernel.mode} (${kernel.weightLayout}) — ${kernel.note}`);
      }
      console.log('');
      console.log(watModule.text);
    }
    return simplifiedPrimFuncs.map(pf => pf.clone());
  }

  if (verbose) {
    printSubSection('5.2 Storage Rewrite');
  }
  return simplifiedPrimFuncs.map((pf) => {
    const { func, stats } = storageRewrite(pf);
    if (verbose) {
      if (stats.promotedToScalar.length > 0) {
        console.log(`  ${pf.name}: promoted [${stats.promotedToScalar.join(', ')}] to scalar`);
        console.log(`    alloc: ${stats.originalAllocBytes}B → ${stats.optimizedAllocBytes}B`);
      } else {
        console.log(`  ${pf.name}: no scalar promotion needed`);
      }
    }
    return func;
  });
}

function getModelCompileCacheId(model: Module): string {
  const taggedModel = model as Module & { [MODEL_COMPILE_CACHE_ID]?: string };
  if (!taggedModel[MODEL_COMPILE_CACHE_ID]) {
    taggedModel[MODEL_COMPILE_CACHE_ID] = `model_${nextModelCompileCacheId++}`;
  }
  return taggedModel[MODEL_COMPILE_CACHE_ID]!;
}

function getSharedCompileCache(model: Module): Map<string, SharedCompileArtifacts> {
  const modelCacheId = getModelCompileCacheId(model);
  let cache = sharedCompileCache.get(modelCacheId);
  if (!cache) {
    cache = new Map<string, SharedCompileArtifacts>();
    sharedCompileCache.set(modelCacheId, cache);
  }
  return cache;
}

export class CompiledModule {
  private _model: Module;
  private _opts: Required<CompileOptions>;
  private _runtime: RuntimeModule | null = null;
  primFuncs: PrimFunc[] = [];

  constructor(model: Module, opts: CompileOptions = {}) {
    this._model = model;
    this._opts = {
      inputShape: opts.inputShape ?? [],
      useRegTile: opts.useRegTile ?? true,
      verbose: opts.verbose ?? false,
      backend: opts.backend ?? 'js',
    };

    if (this._opts.inputShape.length > 0) {
      this._compile(this._opts.inputShape);
    }
  }

  forward(input: Tensor): Tensor {
    if (!this._runtime) {
      this._compile(input.shape);
    }
    return this._runtime!.forward(input);
  }
  recompile(inputShape: number[]): void {
    this._runtime = null;
    this.primFuncs = [];
    this._compile(inputShape);
  }

  private _compile(inputShape: number[]): void {
    const t0 = performance.now();
    const shapeKey = inputShape.join('x');
    const compileCache = getSharedCompileCache(this._model);
    const cached = compileCache.get(shapeKey);

    let shared: SharedCompileArtifacts;
    if (cached) {
      shared = cached;
      if (this._opts.verbose) {
        console.log('');
        console.log(`Reusing cached frontend compile artifacts for input shape [${inputShape.join(', ')}].`);
        console.log('Shared phases 1-4 are skipped; only backend-specific work is shown below.');
      }
    } else {
      const tracer = new Tracer();
      if (this._opts.verbose) {
        traceCompilerPhase(1, 'Model Tracing (Capture Computation Graph)');
      }
      const graph = tracer.traceInference(this._model, inputShape);
      if (this._opts.verbose) {
        console.log(Tracer.printGraph(graph));
      }
      const irMod = buildIR(graph);
      const optimized = optimizeIRToPrimFuncs(irMod, this._opts.verbose);
      shared = {
        graph,
        simplifiedPrimFuncs: optimized.primFuncs,
        runtimeParamNames: optimized.runtimeParamNames,
      };
      compileCache.set(shapeKey, shared);
    }

    this.primFuncs = compileBackendPrimFuncs(
      shared.simplifiedPrimFuncs,
      this._opts.backend,
      this._opts.verbose,
      Boolean(cached),
    );

    const paramsMap = new Map<string, Tensor>();
    for (const name of shared.runtimeParamNames) {
      const traced = shared.graph.params.get(name);
      if (traced) {
        paramsMap.set(name, traced.tensor);
      }
    }

    if (this._opts.verbose) {
      traceCompilerPhase(6, 'Runtime Build');
    }
    this._runtime = new RuntimeModule(this.primFuncs, paramsMap, this._opts.backend, this._opts.useRegTile);

    const compilationMs = performance.now() - t0;
    if (this._opts.verbose) {
      console.log(`  Backend: ${this._opts.backend}`);
      console.log(`  Input shape: [${inputShape.join(', ')}]`);
      console.log(`  Final kernels: ${this.primFuncs.map((pf) => pf.name).join(', ')}`);
      if (this._opts.backend === 'wat') {
        const wasmInfo = this._runtime.getWasmDebugInfo();
        if (wasmInfo) {
          console.log(`  Params preloaded once: ${wasmInfo.paramsUploadedOnce ? 'yes' : 'no'}`);
          console.log(`  WASM memory: params=${wasmInfo.paramBytes}B activations=${wasmInfo.activationBytes}B total=${wasmInfo.totalBytes}B (${wasmInfo.totalPages} page(s))`);
          for (const kernel of wasmInfo.kernels) {
            console.log(`  ${kernel.name}: ${kernel.mode}, weight=${kernel.originalWeightBytes}B → ${kernel.packedWeightBytes}B, out=[${kernel.outputShape.join(', ')}]`);
          }
        }
      }
      console.log(`  Compile time: ${compilationMs.toFixed(1)}ms`);
    }
  }
}

export function mlcCompile(model: Module, opts: CompileOptions = {}): CompiledModule {
  return new CompiledModule(model, opts);
}

export function compileTrained(model: Module, opts: CompileOptions = {}): CompiledModule {
  return mlcCompile(model, opts);
}

type Constructor<T = object> = new (...args: any[]) => T;

export function compile(opts: CompileOptions = {}): <T extends Constructor<Module>>(Base: T) => T {
  return function <T extends Constructor<Module>>(Base: T): T {
    return class extends Base {
      private _compiled: CompiledModule | null = null;

      private _createBaseShim(): Module {
        const self = this;
        return {
          training: self.training,
          forward(inp: Tensor): Tensor {
            return Base.prototype.forward.call(self, inp);
          },
          parameters(): Tensor[] {
            return self.parameters();
          },
          children(): Module[] {
            return self.children();
          },
          train(): Module {
            self.train();
            return this;
          },
          eval(): Module {
            self.eval();
            return this;
          },
        };
      }

      override forward(x: Tensor): Tensor {
        if (!this._compiled) {
          this._compiled = new CompiledModule(this._createBaseShim(), {
            ...opts,
            inputShape: x.shape,
          });
        }
        return this._compiled.forward(x);
      }
    } as T;
  };
}
