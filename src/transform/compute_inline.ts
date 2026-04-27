// ═══════════════════════════════════════════════════════════════
//  compute_inline — Eliminate an intermediate buffer by inlining
//  a producer kernel's body directly into its consumer.
//
//  Why this matters (MLC concept):
//    When a producer writes to a buffer that the consumer reads,
//    we can substitute the producer's expression directly at each
//    load site — removing the intermediate buffer allocation,
//    the producer's loop nest, and the memory round-trip.
//
//    This is one of TVM's most impactful schedule primitives.
//    It turns two kernel passes (write-then-read) into one, which:
//      - Eliminates intermediate buffer memory traffic
//      - Enables the compiler to further fuse/optimize the merged loop
//      - Reduces launch overhead (one kernel instead of two)
//
//  Example (bias_add inlined into relu):
//
//    BEFORE:
//      @bias_add(A[M,N], B[N], Out[M,N]):
//        for i: for j: Out[i,j] = A[i,j] + B[0,j]
//
//      @relu(X[M,N], Out[M,N]):
//        for i: for j: Out[i,j] = max(X[i,j], 0)
//
//    AFTER inlining bias_add into relu:
//      @relu_inlined(A[M,N], B[N], Out[M,N]):
//        for i: for j: Out[i,j] = max(A[i,j] + B[0,j], 0)
//
//  The intermediate buffer and the bias_add loop are gone.
// ═══════════════════════════════════════════════════════════════

import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, AllocNode,
  BufferLoadExpr, BinOpExpr, MaxExpr, MinExpr, CallExprTIR,
  LoopVar, ConstExpr, VarRefExpr, BufferDecl,
  VarIndex, ConstIndex, BinOpIndex,
  type Stmt, type ValueExpr, type IndexExpr
} from '../ir/low_level.js';

// ─── Result type ───

export interface ComputeInlineResult {
  /** The merged consumer function with producer inlined */
  inlined: PrimFunc;
  /** Number of load sites that were substituted */
  substituted: number;
  /** Name of producer function that was eliminated */
  producerName: string;
  /** Name of consumer function (now the merged result) */
  consumerName: string;
}

// ─── Main entry point ───
//
// producer: the PrimFunc to inline (its output buffer is the inline target)
// consumer: the PrimFunc that reads from the producer's output buffer
// producerOutBuf: name of the producer's output buffer (e.g. 'Out', 'Bias_Out')
//
// Returns a new PrimFunc that is the consumer with all loads from producerOutBuf
// replaced by the producer's computation expression.

export function computeInline(
  producer: PrimFunc,
  consumer: PrimFunc,
  producerOutBufName: string = 'Out'
): ComputeInlineResult {
  // Step 1: Extract the producer's computation as a function of indices.
  // We assume the producer has a simple loop structure:
  //   for i: for j: Out[i,j] = expr(A, B, i, j)
  // We extract the rhs expression and the loop vars it depends on.
  const producerBody = extractProducerBody(producer, producerOutBufName);

  if (!producerBody) {
    // Cannot inline — producer structure too complex
    return {
      inlined: consumer,
      substituted: 0,
      producerName: producer.name,
      consumerName: consumer.name,
    };
  }

  const { loopVars, rhs, indexToLoopVar } = producerBody;

  // Step 2: Walk the consumer and replace every BufferLoad(producerOut, [i,j,...])
  // with the producer's rhs expression, substituting loop vars with the consumer's indices.
  let substituted = 0;

  function inlineValue(val: ValueExpr, consumerIndicesAtLoad?: IndexExpr[]): ValueExpr {
    if (val instanceof BufferLoadExpr) {
      if (val.buffer.name === producerOutBufName) {
        // Substitute: rhs with producer loop vars replaced by consumer's load indices
        const indexMap = new Map<string, IndexExpr>();
        for (let d = 0; d < loopVars.length; d++) {
          indexMap.set(loopVars[d].name, val.indices[d] ?? new ConstIndex(0));
        }
        substituted++;
        return substituteLoopVarsInValue(rhs, indexMap);
      }
      return val;
    }
    if (val instanceof BinOpExpr) {
      return new BinOpExpr(val.op, inlineValue(val.left), inlineValue(val.right));
    }
    if (val instanceof MaxExpr) {
      return new MaxExpr(inlineValue(val.left), inlineValue(val.right));
    }
    if (val instanceof MinExpr) {
      return new MinExpr(inlineValue(val.left), inlineValue(val.right));
    }
    if (val instanceof CallExprTIR) {
      return new CallExprTIR(val.funcName, val.args.map(a => inlineValue(a)));
    }
    return val; // ConstExpr, VarRefExpr
  }

  function inlineStmt(stmt: Stmt): Stmt {
    if (stmt instanceof BufferStoreNode) {
      return new BufferStoreNode(stmt.buffer, stmt.indices, inlineValue(stmt.value));
    }
    if (stmt instanceof SeqNode) {
      return new SeqNode(stmt.stmts.map(inlineStmt));
    }
    if (stmt instanceof ForNode) {
      return new ForNode(stmt.loopVar, stmt.min, stmt.extent, inlineStmt(stmt.body), stmt.annotation);
    }
    if (stmt instanceof AllocNode) {
      return new AllocNode(stmt.buffer, inlineStmt(stmt.body));
    }
    return stmt;
  }

  const newBody = inlineStmt(consumer.body);

  // Step 3: Build the merged PrimFunc.
  // Params: consumer's params but with producerOutBuf replaced by producer's input params
  // (remove the intermediate buffer, add producer's inputs that aren't already in consumer)
  const mergedName = `${producer.name}_inlined_into_${consumer.name}`;
  const intermediateBuffer = producer.params.find(p => p.name === producerOutBufName);

  // Collect params: keep consumer params (excluding intermediate), add producer params (excluding Out)
  const producerInputParams = producer.params.filter(p => p.name !== producerOutBufName);
  const consumerParamsWithoutIntermediate = consumer.params.filter(
    p => p.name !== producerOutBufName
  );
  // Deduplicate by name (some producer inputs may already be in consumer)
  const existingNames = new Set(consumerParamsWithoutIntermediate.map(p => p.name));
  const newProducerParams = producerInputParams.filter(p => !existingNames.has(p.name));

  const mergedParams = [...consumerParamsWithoutIntermediate, ...newProducerParams];

  const inlinedFunc = new PrimFunc(
    mergedName,
    mergedParams,
    newBody,
    [...consumer.allocations]
  );

  return {
    inlined: inlinedFunc,
    substituted,
    producerName: producer.name,
    consumerName: consumer.name,
  };
}

// ─── Extract the producer's computation body ───
// Returns the loop vars (in order) and the rhs expression from the innermost store.

interface ProducerBody {
  /** Ordered loop variables from outermost to innermost */
  loopVars: LoopVar[];
  /** The rhs value expression written to Out[loopVars[0], loopVars[1], ...] */
  rhs: ValueExpr;
  /** Mapping from dimension index → loop var name */
  indexToLoopVar: Map<number, string>;
}

function extractProducerBody(producer: PrimFunc, outBufName: string): ProducerBody | null {
  const loopVars: LoopVar[] = [];

  function findInnermostStore(stmt: Stmt): BufferStoreNode | null {
    if (stmt instanceof ForNode) {
      loopVars.push(stmt.loopVar);
      const result = findInnermostStore(stmt.body);
      if (!result) loopVars.pop();
      return result;
    }
    if (stmt instanceof SeqNode) {
      for (const s of stmt.stmts) {
        loopVars.length = 0; // reset for each statement branch
        const r = findInnermostStore(s);
        if (r) return r;
      }
      return null;
    }
    if (stmt instanceof AllocNode) {
      return findInnermostStore(stmt.body);
    }
    if (stmt instanceof BufferStoreNode && stmt.buffer.name === outBufName) {
      return stmt;
    }
    return null;
  }

  const store = findInnermostStore(producer.body);
  if (!store) return null;

  const indexToLoopVar = new Map<number, string>();
  for (let i = 0; i < loopVars.length; i++) {
    indexToLoopVar.set(i, loopVars[i].name);
  }

  return { loopVars: [...loopVars], rhs: store.value, indexToLoopVar };
}

// ─── Substitute loop vars in a value expression ───
// Replaces VarIndex references to old loop var names with the provided index expressions.

function substituteLoopVarsInValue(
  val: ValueExpr,
  indexMap: Map<string, IndexExpr>
): ValueExpr {
  if (val instanceof BufferLoadExpr) {
    const newIndices = val.indices.map(idx => substituteLoopVarsInIndex(idx, indexMap));
    return new BufferLoadExpr(val.buffer, newIndices);
  }
  if (val instanceof BinOpExpr) {
    return new BinOpExpr(
      val.op,
      substituteLoopVarsInValue(val.left, indexMap),
      substituteLoopVarsInValue(val.right, indexMap)
    );
  }
  if (val instanceof MaxExpr) {
    return new MaxExpr(
      substituteLoopVarsInValue(val.left, indexMap),
      substituteLoopVarsInValue(val.right, indexMap)
    );
  }
  if (val instanceof MinExpr) {
    return new MinExpr(
      substituteLoopVarsInValue(val.left, indexMap),
      substituteLoopVarsInValue(val.right, indexMap)
    );
  }
  if (val instanceof CallExprTIR) {
    return new CallExprTIR(
      val.funcName,
      val.args.map(a => substituteLoopVarsInValue(a, indexMap))
    );
  }
  if (val instanceof VarRefExpr) {
    // If this loop var is in the index map, replace with the index as a load-friendly form
    // VarRefExpr is for scalar loop var references; we keep as-is here
    return val;
  }
  return val; // ConstExpr
}

function substituteLoopVarsInIndex(
  idx: IndexExpr,
  indexMap: Map<string, IndexExpr>
): IndexExpr {
  if (idx instanceof VarIndex) {
    const replacement = indexMap.get(idx.loopVar.name);
    return replacement ?? idx;
  }
  if (idx instanceof BinOpIndex) {
    return new BinOpIndex(
      idx.op,
      substituteLoopVarsInIndex(idx.left, indexMap),
      substituteLoopVarsInIndex(idx.right, indexMap)
    );
  }
  return idx; // ConstIndex
}

// ─── Demonstration helper ───
// Builds two small PrimFuncs (bias_add + relu) as separate kernels,
// then inlines the bias_add into the relu, producing a merged kernel.
// Returns the result string for printing.

export function demoComputeInline(M: number, N: number): string {
  const lines: string[] = [];

  // Build bias_add: for i: for j: BiasOut[i,j] = A[i,j] + B[0,j]
  const iVar = new LoopVar('i', 'spatial');
  const jVar = new LoopVar('j', 'spatial');
  const A = new BufferDecl('A', [M, N]);
  const B = new BufferDecl('B', [1, N]);
  const BiasOut = new BufferDecl('BiasOut', [M, N]);

  const biasStore = new BufferStoreNode(
    BiasOut, [new VarIndex(iVar), new VarIndex(jVar)],
    new BinOpExpr('+',
      new BufferLoadExpr(A, [new VarIndex(iVar), new VarIndex(jVar)]),
      new BufferLoadExpr(B, [new ConstIndex(0), new VarIndex(jVar)])
    )
  );
  const biasBody = new ForNode(iVar, 0, M,
    new ForNode(jVar, 0, N, biasStore)
  );
  const biasAdd = new PrimFunc('bias_add', [A, B, BiasOut], biasBody);

  // Build relu: for i: for j: Out[i,j] = max(X[i,j], 0)
  const iVar2 = new LoopVar('i', 'spatial');
  const jVar2 = new LoopVar('j', 'spatial');
  const X = new BufferDecl('BiasOut', [M, N]); // reads from bias_add output
  const Out = new BufferDecl('Out', [M, N]);

  const reluStore = new BufferStoreNode(
    Out, [new VarIndex(iVar2), new VarIndex(jVar2)],
    new MaxExpr(
      new BufferLoadExpr(X, [new VarIndex(iVar2), new VarIndex(jVar2)]),
      new ConstExpr(0)
    )
  );
  const reluBody = new ForNode(iVar2, 0, M,
    new ForNode(jVar2, 0, N, reluStore)
  );
  const relu = new PrimFunc('relu', [X, Out], reluBody);

  // Import printer to format TIR
  // (we print manually instead to avoid circular import)
  const printSimpleTIR = (func: PrimFunc): string => {
    const stmtLines: string[] = [];
    function printStmt(stmt: Stmt, indent: string): void {
      if (stmt instanceof ForNode) {
        stmtLines.push(`${indent}for ${stmt.loopVar.name} in range(0, ${stmt.extent}):`);
        printStmt(stmt.body, indent + '  ');
      } else if (stmt instanceof SeqNode) {
        for (const s of stmt.stmts) printStmt(s, indent);
      } else if (stmt instanceof AllocNode) {
        stmtLines.push(`${indent}alloc ${stmt.buffer.name}: float32[${stmt.buffer.shape.join(',')}]`);
        printStmt(stmt.body, indent);
      } else if (stmt instanceof BufferStoreNode) {
        stmtLines.push(`${indent}${stmt.buffer.name}[${stmt.indices.map(printIdx).join(',')}] = ${printVal(stmt.value)}`);
      }
    }
    function printIdx(idx: IndexExpr): string {
      if (idx instanceof VarIndex) return idx.loopVar.name;
      if (idx instanceof ConstIndex) return `${idx.value}`;
      if (idx instanceof BinOpIndex) return `(${printIdx(idx.left)}${idx.op}${printIdx(idx.right)})`;
      return '?';
    }
    function printVal(val: ValueExpr): string {
      if (val instanceof BufferLoadExpr) return `${val.buffer.name}[${val.indices.map(printIdx).join(',')}]`;
      if (val instanceof BinOpExpr) return `(${printVal(val.left)}${val.op}${printVal(val.right)})`;
      if (val instanceof MaxExpr) return `max(${printVal(val.left)},${printVal(val.right)})`;
      if (val instanceof MinExpr) return `min(${printVal(val.left)},${printVal(val.right)})`;
      if (val instanceof ConstExpr) return `${val.value}`;
      if (val instanceof VarRefExpr) return val.loopVar.name;
      if (val instanceof CallExprTIR) return `${val.funcName}(${val.args.map(printVal).join(',')})`;
      return '?';
    }
    const paramStr = func.params.map(p => `${p.name}[${p.shape.join(',')}]`).join(', ');
    stmtLines.push(`@${func.name}(${paramStr}):`);
    printStmt(func.body, '  ');
    return stmtLines.join('\n');
  };

  lines.push(`  Producer (bias_add):  ${M}×${N} matrix, adds B[0,j] to each row`);
  lines.push(printSimpleTIR(biasAdd).split('\n').map(l => '    ' + l).join('\n'));
  lines.push('');
  lines.push(`  Consumer (relu):      max(BiasOut[i,j], 0)`);
  lines.push(printSimpleTIR(relu).split('\n').map(l => '    ' + l).join('\n'));
  lines.push('');

  // Perform inline
  const result = computeInline(biasAdd, relu, 'BiasOut');

  lines.push(`  After computeInline(bias_add → relu):`);
  lines.push(`    Substituted ${result.substituted} load site(s)`);
  lines.push(`    Intermediate buffer 'BiasOut[${M},${N}]' eliminated`);
  lines.push(`    Merged kernel '${result.inlined.name}':`);
  lines.push(printSimpleTIR(result.inlined).split('\n').map(l => '    ' + l).join('\n'));

  return lines.join('\n');
}
