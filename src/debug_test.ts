import { NDArray } from './tensor/ndarray.js';
import { Linear, Sigmoid, Sequential } from './model/nn.js';
import { BCEWithLogitsLoss } from './loss/loss.js';
import { naiveForward, RuntimeModule } from './runtime/executor.js';
import { Tracer } from './trace/tracer.js';
import { buildIR } from './ir/high_level.js';
import { constantFold } from './transform/constant_fold.js';
import { fuseOps } from './transform/op_fusion.js';
import { deadCodeElimination } from './transform/dead_code_elimination.js';
import { lowerModule } from './lower/lowering.js';
import { Schedule } from './transform/schedule.js';
import { arithmeticSimplify } from './transform/arithmetic_simplify.js';
import { storageRewrite } from './transform/storage_rewrite.js';
import { codegenJS, compile } from './codegen/js_codegen.js';

const model = new Sequential([
  new Linear(4, 3),
  new Sigmoid(),
  new Linear(3, 2),
]);

const params = model.parameters();
const tracer = new Tracer();
const graph = tracer.traceTraining(model, new BCEWithLogitsLoss(), [1, 4], [1, 2]);

const irModule = buildIR(graph);
const folded = constantFold(irModule);
const fused = fuseOps(folded);
const { module: dceModule } = deadCodeElimination(fused);
const primFuncs = lowerModule(dceModule);

console.log('=== Naive code ===');
console.log(codegenJS(primFuncs[0]));

// Schedule
const pf = primFuncs[0];
const simplified = arithmeticSimplify(pf);
const { func: rewritten } = storageRewrite(simplified);

console.log('\n=== After arith simplify + storage rewrite ===');
console.log(codegenJS(rewritten));

// Try compile and run
const input = NDArray.rand([1, 4]);
const naiveFn = compile(primFuncs[0]);
const rewrittenFn = compile(rewritten);

const naiveOut = new Float32Array(3);
const rewrittenOut = new Float32Array(3);

naiveFn(input.data, params[0].data.data, params[1].data.data, naiveOut);
console.log('\nNaive output:', Array.from(naiveOut));

rewrittenFn(input.data, params[0].data.data, params[1].data.data, rewrittenOut);
console.log('Rewritten output:', Array.from(rewrittenOut));

console.log('Match:', Array.from(naiveOut).every((v, i) => Math.abs(v - rewrittenOut[i]) < 1e-5));
