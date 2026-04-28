import { NDArray } from './tensor/ndarray.js';
import { GradTensor, engine } from './autograd/engine.js';
import { Linear, ReLU, Sequential, type Module } from './model/nn.js';
import { RuntimeModule } from './runtime/executor.js';
import { Tracer } from './trace/tracer.js';
import { buildIR } from './ir/high_level.js';
import { constantFold } from './transform/constant_fold.js';
import { fuseOps } from './transform/op_fusion.js';
import { deadCodeElimination } from './transform/dead_code_elimination.js';
import { cseModule } from './transform/cse.js';
import { lowerModule } from './lower/lowering.js';
import { arithmeticSimplify } from './transform/arithmetic_simplify.js';
import { storageRewrite } from './transform/storage_rewrite.js';
import { mlcCompile, compile } from './compile/decorator.js';

const BATCH = 4;
const IN    = 32;
const HID   = 64;
const OUT   = 8;

function sep(label: string) {
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`  ${label}`);
  console.log('─'.repeat(60));
}

function assertClose(a: NDArray, b: NDArray, thresh = 1e-4, label = ''): void {
  let maxDelta = 0;
  const aFlat = a.data;
  const bFlat = b.data;
  for (let i = 0; i < aFlat.length; i++) {
    maxDelta = Math.max(maxDelta, Math.abs(aFlat[i] - bFlat[i]));
  }
  const ok = maxDelta < thresh;
  console.log(`  ${ok ? '✓ PASS' : '✗ FAIL'} ${label} (max|Δ| = ${maxDelta.toExponential(3)})`);
  if (!ok) {
    console.error(`  [FAIL] max delta ${maxDelta.toExponential(3)} exceeds threshold ${thresh}`);
    process.exitCode = 1;
  }
}

function makeModel(): Sequential {
  return new Sequential([
    new Linear(IN, HID),
    new ReLU(),
    new Linear(HID, OUT),
  ]);
}

function referenceForward(model: Module, input: NDArray): NDArray {
  const tracer = new Tracer();
  const graph  = tracer.traceInference(model, input.shape);
  const irMod  = buildIR(graph);
  const folded = constantFold(irMod);
  const fused  = fuseOps(folded);
  const { module: dceMod } = deadCodeElimination(fused);
  const { module: cseMod } = cseModule(dceMod);
  const primFuncs = lowerModule(cseMod);
  const optimized = primFuncs.map(pf => storageRewrite(arithmeticSimplify(pf)).func);
  const params = model.parameters();
  const paramsMap = new Map<string, NDArray>();
  params.forEach((p, i) => paramsMap.set(`param_${i}`, p.data));
  const rt = new RuntimeModule(optimized, paramsMap, true);
  return rt.forward(input);
}

let testsPassed = 0;
let testsFailed = 0;

function test(name: string, fn: () => void): void {
  try {
    fn();
    testsPassed++;
  } catch (e: any) {
    console.error(`  [ERROR] ${name}: ${e.message}`);
    testsFailed++;
    process.exitCode = 1;
  }
}

console.log('═'.repeat(60));
console.log('  MLC @compile Decorator — API Flexibility Tests');
console.log('═'.repeat(60));

sep('Test 1: Lazy compile (no inputShape)');
test('lazy compile', () => {
  const model = makeModel();
  const compiled = mlcCompile(model);  // no inputShape → not compiled yet

  console.log(`  getStats() before forward: ${compiled.getStats() === null ? '✓ null (not compiled yet)' : '✗ should be null'}`);
  if (compiled.getStats() !== null) throw new Error('Should not compile before forward()');

  const input = NDArray.rand([BATCH, IN]);
  const out   = compiled.forward(input);  // triggers compilation here

  const stats = compiled.getStats();
  if (!stats) throw new Error('Stats should be set after forward()');

  console.log(`  Compiled after first forward(): ✓`);
  console.log(`  inputShape : [${stats.inputShape.join(', ')}]`);
  console.log(`  numKernels : ${stats.numKernels}`);
  console.log(`  kernelNames: ${stats.kernelNames.join(', ')}`);
  console.log(`  compileMs  : ${stats.compilationMs.toFixed(1)}ms`);
  console.log(`  output shape: [${out.shape.join(', ')}]`);

  if (out.shape[0] !== BATCH || out.shape[1] !== OUT)
    throw new Error(`Wrong output shape: [${out.shape}]`);

  // Second call should NOT recompile (same stats object)
  const stats2 = compiled.getStats();
  if (stats2!.compiledAt !== stats.compiledAt)
    throw new Error('Should not recompile on second forward()');
  console.log(`  Second forward() does NOT retrace: ✓`);
});

sep('Test 2: Eager compile (inputShape provided upfront)');
test('eager compile', () => {
  const model    = makeModel();
  const compiled = mlcCompile(model, { inputShape: [BATCH, IN] });

  const stats = compiled.getStats();
  if (!stats) throw new Error('Stats should exist immediately after eager mlcCompile()');

  console.log(`  getStats() immediately after wrap: ✓`);
  console.log(`  inputShape : [${stats.inputShape.join(', ')}]`);
  console.log(`  numKernels : ${stats.numKernels}  (${stats.kernelNames.join(', ')})`);
  console.log(`  compileMs  : ${stats.compilationMs.toFixed(1)}ms`);

  const input = NDArray.rand([BATCH, IN]);
  const out   = compiled.forward(input);
  console.log(`  output shape: [${out.shape.join(', ')}]`);

  if (out.shape[0] !== BATCH || out.shape[1] !== OUT)
    throw new Error(`Wrong output shape: [${out.shape}]`);
});

sep('Test 3: @compile class decorator');
test('class decorator', () => {
  @compile({ useRegTile: true })
  class Classifier extends Sequential {}

  const net = new Classifier([
    new Linear(IN, HID),
    new ReLU(),
    new Linear(HID, OUT),
  ]);

  // GradTensor input (typical user path)
  const inputNd = NDArray.rand([BATCH, IN]);
  const inputGt = new GradTensor(inputNd, false);

  const outGt = net.forward(inputGt);
  console.log(`  output type: GradTensor ✓`);
  console.log(`  output shape: [${outGt.data.shape.join(', ')}]`);

  // Access compile stats via getCompileStats()
  const stats = (net as any).getCompileStats();
  if (!stats) throw new Error('getCompileStats() should be set after forward()');
  console.log(`  compileMs  : ${stats.compilationMs.toFixed(1)}ms`);
  console.log(`  kernels    : ${stats.kernelNames.join(', ')}`);

  if (outGt.data.shape[0] !== BATCH || outGt.data.shape[1] !== OUT)
    throw new Error(`Wrong output shape`);
});

sep('Test 4: Custom non-Sequential model (class extends Module)');
test('custom model', () => {
  class TwoLayerNet extends Sequential {
    constructor(inF: number, hidF: number, outF: number) {
      super([new Linear(inF, hidF), new ReLU(), new Linear(hidF, outF)]);
    }
  }

  const net      = new TwoLayerNet(IN, HID, OUT);
  const compiled = mlcCompile(net, { inputShape: [BATCH, IN] });
  const input    = NDArray.rand([BATCH, IN]);
  const out      = compiled.forward(input);

  console.log(`  Custom model compiled: ✓`);
  console.log(`  output shape: [${out.shape.join(', ')}]`);
  if (out.shape[0] !== BATCH || out.shape[1] !== OUT)
    throw new Error(`Wrong output shape: [${out.shape}]`);

  const stats = compiled.getStats()!;
  console.log(`  kernels    : ${stats.kernelNames.join(', ')}`);
});

sep('Test 5: Verbose mode');
test('verbose mode', () => {
  const model    = makeModel();
  console.log('  (verbose output below)');
  const compiled = mlcCompile(model, { inputShape: [BATCH, IN], verbose: true });
  const input    = NDArray.rand([BATCH, IN]);
  compiled.forward(input);
  console.log(`  Verbose compile completed: ✓`);
});

sep('Test 6: Correctness — CompiledModule vs RuntimeModule reference');
test('correctness', () => {
  const model   = makeModel();
  const input   = NDArray.rand([BATCH, IN]);

  const ref = referenceForward(model, input);

  const compiled = mlcCompile(model, { inputShape: [BATCH, IN] });
  const result   = compiled.forward(input);

  assertClose(ref, result, 1e-4, 'CompiledModule vs manual RuntimeModule');
});

sep('Test 7: Benchmark — GradTensor engine vs CompiledModule');
test('benchmark', () => {
  const model    = makeModel();
  const input    = NDArray.rand([BATCH, IN]);
  const inputGt  = new GradTensor(input, false);
  const compiled = mlcCompile(model, { inputShape: [BATCH, IN] });

  const WARMUP = 100;
  const REPS   = 500;

  // Warmup GradTensor engine
  for (let i = 0; i < WARMUP; i++) { engine.reset(); model.forward(inputGt); }
  // Benchmark GradTensor engine
  const t0 = performance.now();
  for (let i = 0; i < REPS; i++) { engine.reset(); model.forward(inputGt); }
  const engineMs = (performance.now() - t0) / REPS;

  // Warmup compiled
  for (let i = 0; i < WARMUP; i++) compiled.forward(input);
  // Benchmark compiled
  const t1 = performance.now();
  for (let i = 0; i < REPS; i++) compiled.forward(input);
  const compiledMs = (performance.now() - t1) / REPS;

  const speedup = engineMs / compiledMs;
  console.log(`  GradTensor engine : ${engineMs.toFixed(4)}ms / call`);
  console.log(`  CompiledModule    : ${compiledMs.toFixed(4)}ms / call`);
  console.log(`  Speedup           : ${speedup.toFixed(2)}×  (${speedup >= 1 ? 'compiled faster' : 'engine faster'})`);

  if (compiledMs > engineMs * 5) {
    throw new Error(`Compiled path is unexpectedly slow (${speedup.toFixed(2)}×)`);
  }
});

sep('Test 8: recompile() — change batch size 4 → 8');
test('recompile', () => {
  const model    = makeModel();
  const compiled = mlcCompile(model, { inputShape: [BATCH, IN] });

  const stats4 = compiled.getStats()!;
  console.log(`  Initial compile: batch=${stats4.inputShape[0]} ✓`);

  // Change batch size to 8
  const NEW_BATCH = 8;
  compiled.recompile([NEW_BATCH, IN]);

  const stats8 = compiled.getStats()!;
  console.log(`  After recompile: batch=${stats8.inputShape[0]} ✓`);
  if (stats8.inputShape[0] !== NEW_BATCH)
    throw new Error(`inputShape not updated: got ${stats8.inputShape[0]}`);

  const input8 = NDArray.rand([NEW_BATCH, IN]);
  const out8   = compiled.forward(input8);
  console.log(`  Output shape: [${out8.shape.join(', ')}]`);
  if (out8.shape[0] !== NEW_BATCH || out8.shape[1] !== OUT)
    throw new Error(`Wrong output shape after recompile: [${out8.shape}]`);

  const ref8 = referenceForward(model, input8);
  assertClose(ref8, out8, 1e-4, 'recompiled output vs reference');
});

console.log('\n' + '═'.repeat(60));
console.log(`  Results: ${testsPassed} passed, ${testsFailed} failed`);
if (testsFailed === 0) {
  console.log('  ✓ All tests passed');
} else {
  console.log('  ✗ Some tests failed');
}
console.log('═'.repeat(60));

if (testsFailed > 0) process.exit(1);
