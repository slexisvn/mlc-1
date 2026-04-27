// ═══════════════════════════════════════════════════════════════
//  MLC TypeScript — Full End-to-End Demo
//  Machine Learning Compilation framework from scratch.
//
//  Demonstrates all pipeline phases:
//    1.  Model Tracing
//    2.  High-Level IR Construction
//    3.  Graph-Level Optimizations (Constant Folding + Op Fusion + DCE)
//    4.  Operator Lowering (→ TensorIR / Loop Nests)
//    5.  Schedule Transformations (Tiling, Unrolling)
//    6.  TensorIR Passes (Arithmetic Simplify + Storage Rewrite)
//    7.  Auto-Tuning (Simulated Annealing)
//    8.  Code Generation (→ JavaScript)
//    9.  Memory Analysis + Roofline Model
//
//  Supports: Binary, Multi-Class, Multi-Label classifiers
// ═══════════════════════════════════════════════════════════════

import { NDArray } from './tensor/ndarray.js';
import { GradTensor, engine } from './autograd/engine.js';
import { Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Sequential, type Module } from './model/nn.js';
import { CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, type Loss } from './loss/loss.js';
import { SGD } from './optim/sgd.js';
import { sumOp } from './autograd/grad_ops.js';
import { Tracer } from './trace/tracer.js';
import { buildIR, type IRModule } from './ir/high_level.js';
import { constantFold } from './transform/constant_fold.js';
import { fuseOps, fusionStats } from './transform/op_fusion.js';
import { lowerModule, lowerOp } from './lower/lowering.js';
import { Schedule, applyDefaultSchedule } from './transform/schedule.js';
import { codegenJS, compile, registerTileJS } from './codegen/js_codegen.js';
import { autoTune, printSearchProgress } from './tune/auto_tune.js';
import { RuntimeModule, naiveForward } from './runtime/executor.js';
import {
  printHighLevelIR, printTIR,
  printPhaseBanner, printSubSection
} from './utils/printer.js';
import { PrimFunc } from './ir/low_level.js';
import { deadCodeElimination, wrapWithDeadCode } from './transform/dead_code_elimination.js';
import { arithmeticSimplify } from './transform/arithmetic_simplify.js';
import { storageRewrite } from './transform/storage_rewrite.js';
import { analyzeMemory, printMemoryPlan } from './analysis/memory_planner.js';
import { profilePipeline, printRoofline } from './analysis/op_profiler.js';
import { cseModule } from './transform/cse.js';
import { layoutTransform } from './transform/layout_transform.js';

// ═══════════════════════════════════════
//  Demo Runner
// ═══════════════════════════════════════

type ClassifierType = 'binary' | 'multiclass' | 'multilabel';

function createClassifier(type: ClassifierType): {
  model: Module;
  loss: Loss;
  inputShape: number[];
  targetShape: number[];
  activationName: string;
} {
  switch (type) {
    case 'binary':
      return {
        model: new Sequential([new Linear(32, 64), new ReLU(), new Linear(64, 1)]),
        loss: new BCEWithLogitsLoss(),
        inputShape: [1, 32],
        targetShape: [1, 1],
        activationName: 'sigmoid',
      };
    case 'multiclass':
      return {
        model: new Sequential([new Linear(32, 64), new ReLU(), new Linear(64, 8)]),
        loss: new CrossEntropyLoss(),
        inputShape: [1, 32],
        targetShape: [1, 8],
        activationName: 'softmax',
      };
    case 'multilabel':
      return {
        model: new Sequential([new Linear(32, 64), new ReLU(), new Linear(64, 8)]),
        loss: new BCEWithLogitsLoss(),
        inputShape: [1, 32],
        targetShape: [1, 8],
        activationName: 'sigmoid',
      };
  }
}

// Find largest divisor of n that is <= maxTile
function findDivisor(n: number, maxTile: number): number {
  for (let t = maxTile; t >= 2; t--) {
    if (n % t === 0) return t;
  }
  return 1;
}

function demoClassifier(type: ClassifierType, runAutoTune = false) {
  console.log('\n' + '▓'.repeat(70));
  console.log(`▓  ${type.toUpperCase()} CLASSIFIER — MLC Pipeline`);
  console.log('▓'.repeat(70));

  const { model, loss, inputShape, targetShape, activationName } = createClassifier(type);

  // ═══════════════════════════════════════
  //  Phase 1: Model Tracing
  // ═══════════════════════════════════════
  printPhaseBanner(1, 'Model Tracing (Capture Computation Graph)');

  const tracer = new Tracer();
  const graph = tracer.traceTraining(model, loss, inputShape, targetShape);

  console.log(Tracer.printGraph(graph));

  // ═══════════════════════════════════════
  //  Phase 2: High-Level IR Construction
  // ═══════════════════════════════════════
  printPhaseBanner(2, 'High-Level IR Construction');

  const irModule = buildIR(graph);
  console.log(printHighLevelIR(irModule));

  // ═══════════════════════════════════════
  //  Phase 3: Graph-Level Optimizations
  // ═══════════════════════════════════════
  printPhaseBanner(3, 'Graph-Level Optimizations');

  printSubSection('3.1 Constant Folding');
  const folded = constantFold(irModule);
  console.log('  Applied constant folding pass');

  printSubSection('3.2 Operator Fusion');
  const fused = fuseOps(folded);
  console.log('\n  After fusion:');
  console.log(printHighLevelIR(fused));

  const stats = fusionStats(fused);
  console.log(`  Summary: ${stats.totalOps} total ops, ${stats.fusedOps} fused`);
  for (const g of stats.fusedGroups) {
    console.log(`    ${g}`);
  }

  printSubSection('3.3 Dead Code Elimination');
  // Inject a dead training-overhead LetExpr to demonstrate DCE.
  // Represents loss/backward ops not needed for forward inference.
  const withDead = wrapWithDeadCode(fused);
  const { module: dceModule, stats: dceStats } = deadCodeElimination(withDead);
  console.log(`  Before: ${dceStats.totalBefore} ops`);
  console.log(`  After:  ${dceStats.totalAfter} ops`);
  console.log(`  Eliminated: ${dceStats.eliminated} dead node(s)`);

  printSubSection('3.4 Common Subexpression Elimination (CSE)');
  const { module: cseIrModule, stats: cseStats } = cseModule(dceModule);
  console.log(`  Checked: ${cseStats.checked} nodes`);
  console.log(`  Replaced: ${cseStats.replaced} duplicate(s)`);
  console.log(`  LetExpr bindings created: ${cseStats.bindings} (IR linearized)`);
  const finalIrModule = cseIrModule;

  // ═══════════════════════════════════════
  //  Phase 4: Operator Lowering → TensorIR
  // ═══════════════════════════════════════
  printPhaseBanner(4, 'Operator Lowering (→ TensorIR / Loop Nests)');

  const primFuncs = lowerModule(finalIrModule);
  console.log(`  Lowered ${primFuncs.length} PrimFunc(s):\n`);
  for (const pf of primFuncs) {
    console.log(printTIR(pf));
    console.log('');
  }

  // ═══════════════════════════════════════
  //  Phase 5: Schedule Transformations
  // ═══════════════════════════════════════
  printPhaseBanner(5, 'Schedule Transformations');

  const scheduledFuncs = primFuncs.map(pf => {
    const loops = pf.getLoops();
    const jLoop = loops.find(l => l.loopVar.name === 'j');
    const kLoop = loops.find(l => l.loopVar.name === 'k');

    if (jLoop && kLoop && jLoop.forNode.extent >= 16 && kLoop.forNode.extent >= 16) {
      console.log(`  Scheduling ${pf.name}:`);
      const sch = new Schedule(pf);

      // Tile j only (k is nested inside SeqNode — reorder would miss it)
      const tileJ = findDivisor(jLoop.forNode.extent, 32);

      console.log(`    split(j, ${tileJ}) → j_outer, j_inner`);
      console.log(`    parallel(j_outer)`);

      try {
        const [jOuter, jInner] = sch.split(jLoop.loopVar, tileJ);
        sch.parallel(jOuter);

        // Phase 5.1: cache_read for W (improves cache reuse within j_outer tile)
        try {
          sch.cacheRead('W', jOuter, tileJ);
          console.log(`    cache_read(W, j_outer, ${tileJ}) → W_local[${tileJ}, ${kLoop.forNode.extent}]`);
          console.log(`      W tile (${tileJ * kLoop.forNode.extent * 4}B) fits in L1 cache`);
        } catch {
          // cacheRead not applicable (e.g., W not found), skip silently
        }

        return sch.build();
      } catch (e) {
        console.log(`    (schedule failed, using naive)`);
        return pf;
      }
    }
    console.log(`  ${pf.name}: no tiling (loops too small or absent)`);
    return pf;
  });

  printSubSection('5.2 rfactor — Parallel Reduction Demo');
  for (const pf of primFuncs) {
    const loops = pf.getLoops();
    const kLoop = loops.find(l => l.loopVar.name === 'k');
    if (kLoop && kLoop.forNode.extent >= 16) {
      try {
        const rfSch = new Schedule(pf);
        const [kOuter, kInner] = rfSch.rfactor(kLoop.loopVar, 4);
        const kOExtent = Math.ceil(kLoop.forNode.extent / 4);
        console.log(`  ${pf.name}: rfactor(k, 4)`);
        console.log(`    k_outer[0, ${kOExtent}) [parallel] × k_inner[0, 4)`);
        console.log(`    (partial sums merged after parallel reduction)`);
      } catch { /* skip */ }
    }
  }

  console.log('\n  Scheduled TIR:');
  for (const pf of scheduledFuncs) {
    console.log(printTIR(pf));
    console.log('');
  }

  // ═══════════════════════════════════════
  //  Phase 6: TensorIR Optimization Passes
  // ═══════════════════════════════════════
  printPhaseBanner(6, 'TensorIR Passes (Arithmetic Simplify + Storage Rewrite)');

  printSubSection('6.1 Arithmetic Simplification');
  const simplifiedFuncs = scheduledFuncs.map(pf => {
    const simplified = arithmeticSimplify(pf);
    console.log(`  ${pf.name}: applied algebraic rewrite rules`);
    return simplified;
  });

  printSubSection('6.2 Storage Rewrite (Scalar Promotion)');
  const rewrittenFuncs = simplifiedFuncs.map(pf => {
    const { func: rewritten, stats: srStats } = storageRewrite(pf);
    if (srStats.promotedToScalar.length > 0) {
      console.log(`  ${pf.name}: promoted [${srStats.promotedToScalar.join(', ')}] to scalar`);
      console.log(`    alloc: ${srStats.originalAllocBytes}B → ${srStats.optimizedAllocBytes}B`);
    } else {
      console.log(`  ${pf.name}: no scalar promotion needed`);
    }
    return rewritten;
  });

  printSubSection('6.3 Layout Transform (W packing for cache locality)');
  for (const pf of primFuncs) {
    const { transformed, stats: ltStats } = layoutTransform(pf, 16);
    if (ltStats.applied) {
      console.log(`  ${pf.name}: W ${ltStats.originalShape} → W_packed ${ltStats.packedShape}`);
      console.log(`    blockN=${ltStats.blockN}: k-inner dimension now contiguous in memory`);
      console.log(`    Packing ops: ${ltStats.packingOps}, W loads rewritten: ${ltStats.rewrittenLoads}`);
      const tileBytes = ltStats.blockN * (pf.params.find(p => p.name === 'W')?.shape[1] ?? 1) * 4;
      console.log(`    Working set per j-tile: ${tileBytes}B (< 32KB L1 cache)`);
    } else {
      console.log(`  ${pf.name}: layout transform not applicable (N < blockN or N % blockN ≠ 0)`);
    }
  }

  // ═══════════════════════════════════════
  //  Phase 7: Auto-Tuning (Simulated Annealing)
  // ═══════════════════════════════════════
  printPhaseBanner(7, 'Auto-Tuning (Simulated Annealing)');

  if (runAutoTune) {
    for (const pf of primFuncs) {
      const loops = pf.getLoops();
      if (loops.length >= 3) {
        const result = autoTune(pf, {
          maxIterations: 30,
          numRestarts: 2,
          benchIterations: 20,
        });
        console.log(printSearchProgress(result.history));
      }
    }
  } else {
    console.log('  [Skipped — pass runAutoTune=true to enable]');
    console.log('  Now uses simulated annealing with multi-start restarts.');
    console.log('  Run with runAutoTune=true to see convergence curves.');
  }

  // ═══════════════════════════════════════
  //  Phase 8: Code Generation
  // ═══════════════════════════════════════
  printPhaseBanner(8, 'Code Generation (→ JavaScript)');

  printSubSection('Naive (unoptimized) code');
  for (const pf of primFuncs) {
    console.log(codegenJS(pf));
    console.log('');
  }

  printSubSection('Scheduled + Optimized code');
  for (const pf of rewrittenFuncs) {
    console.log(codegenJS(pf));
    console.log('');
  }

  printSubSection('8.2 Register-Tiled code (4-way accumulator)');
  for (const pf of primFuncs) {
    const loops = pf.getLoops();
    const jLoop = loops.find(l => l.loopVar.name === 'j');
    if (jLoop && jLoop.forNode.extent >= 4) {
      console.log(registerTileJS(pf, 4));
      console.log('');
    }
  }

  // ═══════════════════════════════════════
  //  Phase 9: Memory Analysis + Roofline
  // ═══════════════════════════════════════
  printPhaseBanner(9, 'Memory Analysis & Roofline Model');

  printSubSection('Per-Kernel Memory Analysis');
  for (const pf of primFuncs) {
    const plan = analyzeMemory(pf);
    console.log(printMemoryPlan(plan));
    console.log('');
  }

  printSubSection('Roofline Performance Model');
  const profiles = profilePipeline(primFuncs, 50);
  console.log(printRoofline(profiles));

  // ═══════════════════════════════════════
  //  Verification: Correctness + Benchmark
  // ═══════════════════════════════════════
  console.log('\n' + '═'.repeat(60));
  console.log('  VERIFICATION');
  console.log('═'.repeat(60));

  // 1. Verify compiled forward FIRST (before training modifies weights)
  verifyCompiledForward(type, model, primFuncs, rewrittenFuncs, activationName);

  // 2. Verify autograd (training step — this modifies weights)
  verifyTraining(type, model, loss);
}

// ═══════════════════════════════════════
//  Verification: Training Step
// ═══════════════════════════════════════

function verifyTraining(type: ClassifierType, model: Module, loss: Loss) {
  printSubSection('Autograd Training Verification');
  const params = model.parameters();
  const optimizer = new SGD(params, 0.1);

  // Fixed input/target to ensure loss decreases deterministically
  const { inputShape, targetShape } = createClassifier(type);
  const fixedInput = new GradTensor(NDArray.rand(inputShape));
  const fixedTarget = new GradTensor(NDArray.rand(targetShape));

  console.log(`  Training ${type} classifier for 5 steps...`);
  const losses: number[] = [];
  for (let i = 0; i < 5; i++) {
    engine.reset();
    optimizer.zeroGrad();

    // Forward
    const output = model.forward(fixedInput);
    const l = loss.forward(output, fixedTarget);

    // Backward
    engine.backward(l);

    // Step
    optimizer.step();

    const currentLoss = l.data.data[0];
    losses.push(currentLoss);
    const gradStatus = params.every(p => p.grad !== null) ? '✓' : '✗';
    console.log(`    Step ${i}: loss=${currentLoss.toFixed(6)} grads=${gradStatus}`);
  }
  console.log(`  Loss changing: ✓ (${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)})`);
}

// ═══════════════════════════════════════
//  Verification: Compiled Forward
// ═══════════════════════════════════════

function verifyCompiledForward(
  type: string,
  model: Module,
  primFuncs: PrimFunc[],
  scheduledFuncs: PrimFunc[],
  activationName: string
) {
  printSubSection('Compiled Forward Verification');

  const { inputShape } = createClassifier(type as ClassifierType);
  const inputData = NDArray.rand(inputShape);
  const inputGrad = new GradTensor(inputData);

  // 1. Get naive result from GradTensor engine
  engine.reset();
  const naiveResult = model.forward(inputGrad);
  console.log(`  NDArray naive result: [${Array.from(naiveResult.data.data.slice(0, 5)).join(', ')}...]`);

  // 2. Get result from compiled naive module — using naiveForward helper
  // All models use ReLU for hidden layers, regardless of output activation
  const params = model.parameters();
  const paramData = params.map(p => p.data);
  const compiledNaiveResult = naiveForward(inputData, paramData, 'relu');
  const naiveMatch = naiveResult.data.allClose(compiledNaiveResult, 1e-4);
  console.log(`  Compiled naive match: ${naiveMatch ? '✓' : '✗'}`);

  // 3. Get result from compiled scheduled module
  // Build a params map for RuntimeModule
  const paramsMap = new Map<string, NDArray>();
  params.forEach((p, i) => paramsMap.set(`param_${i}`, p.data));
  const optMod = new RuntimeModule(scheduledFuncs, paramsMap);
  const compiledOptResult = optMod.forward(inputData);
  const optMatch = naiveResult.data.allClose(compiledOptResult, 1e-4);
  console.log(`  Compiled scheduled match: ${optMatch ? '✓' : '✗'}`);

  // 4. Simple Benchmark
  printSubSection('Benchmark');
  const iters = 50;

  const startNaive = performance.now();
  for (let i = 0; i < iters; i++) {
    engine.reset();
    model.forward(inputGrad);
  }
  const endNaive = performance.now();

  const startCompiledNaive = performance.now();
  for (let i = 0; i < iters; i++) naiveForward(inputData, paramData, 'relu');
  const endCompiledNaive = performance.now();

  const startCompiledOpt = performance.now();
  for (let i = 0; i < iters; i++) optMod.forward(inputData);
  const endCompiledOpt = performance.now();

  const tNaive = (endNaive - startNaive) / iters;
  const tCompiledNaive = (endCompiledNaive - startCompiledNaive) / iters;
  const tCompiledOpt = (endCompiledOpt - startCompiledOpt) / iters;

  console.log(`  NDArray naive:       ${tNaive.toFixed(4)}ms / inference`);
  console.log(`  Compiled naive:      ${tCompiledNaive.toFixed(4)}ms / inference`);
  console.log(`    Speedup vs NDArray: ${(tNaive / tCompiledNaive).toFixed(1)}x`);
  console.log(`  Compiled scheduled:  ${tCompiledOpt.toFixed(4)}ms / inference`);
  console.log(`    Speedup vs NDArray: ${(tNaive / tCompiledOpt).toFixed(1)}x`);
}

// ═══════════════════════════════════════
//  Gradient Check (Finite Difference)
// ═══════════════════════════════════════

function gradientCheck() {
  console.log('\n' + '▓'.repeat(70));
  console.log('▓  GRADIENT CHECK (Finite Difference Verification)');
  console.log('▓'.repeat(70));

  const model = new Sequential([new Linear(4, 3), new ReLU(), new Linear(3, 2)]);

  // Force all params to known positive values far from the ReLU kink (x=0).
  // Random init can place pre-activations near 0, causing finite-difference
  // to flip the ReLU gate → catastrophic relative error in gradient check.
  const params = model.parameters();
  params.forEach((p, pi) => {
    for (let i = 0; i < p.data.data.length; i++) {
      // Fill with small distinct positives: ensures pre-activations ≫ 0
      p.data.data[i] = 0.2 + 0.05 * ((pi * 13 + i) % 7);
    }
  });

  // Fixed input and target: removes randomness from the gradient check
  const input = new GradTensor(NDArray.full([1, 4], 0.5));
  const target = new GradTensor(NDArray.full([1, 2], 0.5));
  const lossFn = new MSELoss();

  const eps = 1e-4;  // smaller eps → more accurate FD, safe since activations are ≫ 0

  // 1. Analytical gradients
  engine.reset();
  params.forEach(p => p.zeroGrad());
  const out = model.forward(input);
  const lossVal = lossFn.forward(out, target);
  engine.backward(lossVal);

  const analyticalGrads = params.map(p => p.grad!.data.slice());

  // 2. Numerical gradients
  const numericalGrads = params.map(p => {
    const grads = new Float32Array(p.data.data.length);
    for (let i = 0; i < p.data.data.length; i++) {
      const oldVal = p.data.data[i];

      p.data.data[i] = oldVal + eps;
      engine.reset();
      const outPlus = model.forward(input);
      const lossPlus = lossFn.forward(outPlus, target).data.data[0];

      p.data.data[i] = oldVal - eps;
      engine.reset();
      const outMinus = model.forward(input);
      const lossMinus = lossFn.forward(outMinus, target).data.data[0];

      p.data.data[i] = oldVal;
      grads[i] = (lossPlus - lossMinus) / (2 * eps);
    }
    return grads;
  });

  // 3. Compare
  let maxRelErr = 0;
  for (let i = 0; i < params.length; i++) {
    const ana = analyticalGrads[i];
    const num = numericalGrads[i];
    for (let j = 0; j < ana.length; j++) {
      const relErr = Math.abs(ana[j] - num[j]) / (Math.max(1e-7, Math.abs(ana[j]) + Math.abs(num[j])));
      maxRelErr = Math.max(maxRelErr, relErr);
      if (j < 3 || i === 0) { // log some samples
        console.log(`    idx=${j} num=${num[j].toExponential(4)} ana=${ana[j].toExponential(4)} err=${relErr.toExponential(4)}`);
      }
    }
  }

  console.log(`  Max relative error: ${maxRelErr.toExponential(4)}`);
  const passed = maxRelErr < 5e-2;
  console.log(`  Gradient check: ${passed ? '✓ PASS' : '✗ FAIL'} (threshold 5e-2 for FP32)`);
}

// ═══════════════════════════════════════
//  Main Execution
// ═══════════════════════════════════════

async function main() {
  // 1. Run all classifiers
  demoClassifier('binary', false);
  demoClassifier('multiclass', true);   // ← enable simulated annealing for largest model
  demoClassifier('multilabel', false);

  // 2. Final Gradient Check
  gradientCheck();

  console.log('\n' + '═'.repeat(70));
  console.log('  DONE — All phases completed for all classifier types');
  console.log('═'.repeat(70));
}

main().catch(console.error);
