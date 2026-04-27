// ═══════════════════════════════════════════════════════════════
//  Common Subexpression Elimination (CSE)
//  Detects and eliminates redundant computations in the IR.
//
//  Algorithm:
//    1. Walk the expression tree bottom-up
//    2. Hash each CallExpr by content: op name + arg hashes
//    3. On first occurrence → assign a LetExpr binding (_cse_N = expr)
//    4. On second occurrence (same object OR same content hash) → replace
//       with the previously bound VarExpr
//
//  This pass has two effects:
//    a) LINEARIZATION: converts implicit DAG sharing to explicit LetExpr
//       bindings (making the sharing visible to downstream passes)
//    b) TRUE CSE: eliminates structurally identical sub-expressions that
//       appear in separate branches (e.g., shared activations in backward)
//
//  Note: for simple 2-layer forward models, CSE finds 0 true duplicates
//  but always linearizes the expression into explicit let-bindings.
// ═══════════════════════════════════════════════════════════════

import {
  Expr, CallExpr, VarExpr, LetExpr,
  IRModule, IRFunction, TensorType
} from '../ir/high_level.js';

// ─── Statistics ───

export interface CSEStats {
  checked: number;    // CallExprs examined
  replaced: number;   // duplicates found and replaced
  bindings: number;   // new LetExpr bindings created (linearization)
}

// ─── Content hash for structural equality detection ───

function hashExpr(e: Expr): string {
  if (e.kind === 'var') return `var:${e.name}`;
  if (e.kind === 'constant') return `const:${e.name ?? '?'}:${e.data.shape.join(',')}`;
  if (e.kind === 'let') return `let:${e.varName.name}`;
  // CallExpr: hash recursively through args
  return `call:${e.op.name}:${e.args.map(a => hashExpr(a)).join('|')}`;
}

// ═══════════════════════════════════════
//  Main CSE Pass
// ═══════════════════════════════════════

export function cseModule(module: IRModule): { module: IRModule; stats: CSEStats } {
  const newModule = new IRModule();
  let totalChecked = 0;
  let totalReplaced = 0;
  let totalBindings = 0;

  for (const [name, func] of module.functions) {
    // Maps for deduplication
    const hashToVar = new Map<string, VarExpr>();  // content hash → first VarExpr
    const objToVar  = new Map<Expr, VarExpr>();    // object identity → VarExpr
    const letBindings: Array<{ varExpr: VarExpr; value: Expr }> = [];
    let varCounter = 0;

    function processExpr(e: Expr): Expr {
      // Leaves pass through unchanged
      if (e.kind === 'var' || e.kind === 'constant') return e;

      // LetExpr: recurse into value and body
      if (e.kind === 'let') {
        return new LetExpr(
          e.varName,
          processExpr(e.value),
          processExpr(e.body)
        );
      }

      // ─── CallExpr ───
      totalChecked++;

      // Case 1: this exact object was already processed (DAG sharing → make explicit)
      if (objToVar.has(e)) {
        totalReplaced++;
        return objToVar.get(e)!;
      }

      // Process args bottom-up first (so arg hashes reflect processed forms)
      const newArgs = e.args.map(a => processExpr(a));

      // Case 2: structurally identical expression seen before (true CSE)
      const argHashes = newArgs.map(a => hashExpr(a)).join('|');
      const hash = `call:${e.op.name}:${argHashes}`;

      if (hashToVar.has(hash)) {
        const varExpr = hashToVar.get(hash)!;
        objToVar.set(e, varExpr);
        totalReplaced++;
        return varExpr;
      }

      // New expression — create a LetExpr binding
      const outShape = (e.attrs.outputShape as number[]) ?? [1];
      const varExpr = new VarExpr(`_cse_${varCounter++}`, new TensorType(outShape));
      const newCall = new CallExpr(e.op, newArgs, e.attrs);

      hashToVar.set(hash, varExpr);
      objToVar.set(e, varExpr);
      letBindings.push({ varExpr, value: newCall });
      totalBindings++;

      return varExpr;
    }

    // Process the function body
    const finalBodyVar = processExpr(func.body);

    // Wrap result in LetExpr chain (innermost = last binding)
    let wrappedBody: Expr = finalBodyVar;
    for (let i = letBindings.length - 1; i >= 0; i--) {
      const { varExpr, value } = letBindings[i];
      wrappedBody = new LetExpr(varExpr, value, wrappedBody);
    }

    newModule.addFunction(new IRFunction(
      name,
      func.params,
      wrappedBody,
      func.retType
    ));
  }

  return {
    module: newModule,
    stats: { checked: totalChecked, replaced: totalReplaced, bindings: totalBindings },
  };
}
