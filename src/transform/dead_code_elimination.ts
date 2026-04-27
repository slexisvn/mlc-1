// ═══════════════════════════════════════════════════════════════
//  Dead Code Elimination (DCE)
//  Removes IR nodes whose results are never consumed.
//
//  Algorithm:
//    1. Linearize the expression tree into a list of nodes
//    2. Walk backwards from the output (root), marking all
//       nodes that are transitively needed
//    3. Rebuild IR keeping only marked nodes
//
//  This is the standard "liveness analysis" from compiler theory,
//  adapted for our functional IR where every CallExpr is a "def".
// ═══════════════════════════════════════════════════════════════

import {
  Expr, CallExpr, ConstantExpr, VarExpr, LetExpr,
  IRModule, IRFunction, TensorType, Op, OpPattern
} from '../ir/high_level.js';

// ─── Assign unique ids to each expression node ───

interface TaggedExpr {
  id: number;
  expr: Expr;
  deps: number[];   // ids of expressions this node depends on
}

function tagExprs(root: Expr): { tagged: TaggedExpr[]; rootId: number } {
  const tagged: TaggedExpr[] = [];
  const exprToId = new Map<Expr, number>();
  // Maps let-bound variable name → the id of the expression that defines it
  const varDefMap = new Map<string, number>();
  let nextId = 0;

  function visit(e: Expr): number {
    if (exprToId.has(e)) return exprToId.get(e)!;

    const id = nextId++;
    exprToId.set(e, id);

    if (e.kind === 'var') {
      // deps will be filled in post-processing via varDefMap
      tagged.push({ id, expr: e, deps: [] });
      return id;
    }

    if (e.kind === 'constant') {
      tagged.push({ id, expr: e, deps: [] });
      return id;
    }

    if (e.kind === 'let') {
      const valId = visit(e.value);
      const bodyId = visit(e.body);
      // FIX: LetExpr only structurally depends on its BODY.
      // The value is only live if the bound variable is actually used in the body.
      // We track the var→value mapping and fill in VarExpr deps below.
      tagged.push({ id, expr: e, deps: [bodyId] });
      varDefMap.set(e.varName.name, valId);
      return id;
    }

    // CallExpr
    const deps = e.args.map(arg => visit(arg));
    tagged.push({ id, expr: e, deps });
    return id;
  }

  const rootId = visit(root);

  // Post-process: for each VarExpr that references a let-bound name,
  // add the defining value's id as a dependency so markLive propagates correctly.
  for (const t of tagged) {
    if (t.expr.kind === 'var') {
      const defValId = varDefMap.get(t.expr.name);
      if (defValId !== undefined) {
        t.deps = [defValId];
      }
    }
  }

  return { tagged, rootId };
}

// ─── Mark live nodes by walking backwards from root ───

function markLive(tagged: TaggedExpr[], rootId: number): Set<number> {
  const live = new Set<number>();
  const worklist = [rootId];

  while (worklist.length > 0) {
    const id = worklist.pop()!;
    if (live.has(id)) continue;
    live.add(id);

    const node = tagged.find(t => t.id === id);
    if (node) {
      for (const dep of node.deps) {
        if (!live.has(dep)) worklist.push(dep);
      }
    }
  }

  return live;
}

// ─── Rebuild expression tree, removing dead LetExprs ───

function pruneExpr(expr: Expr, live: Set<number>, exprToId: Map<Expr, number>): Expr {
  if (expr.kind === 'var' || expr.kind === 'constant') return expr;

  if (expr.kind === 'let') {
    const valId = exprToId.get(expr.value);
    // If the let-bound value is dead, skip this let entirely
    if (valId !== undefined && !live.has(valId)) {
      return pruneExpr(expr.body, live, exprToId);
    }
    return new LetExpr(
      expr.varName,
      pruneExpr(expr.value, live, exprToId),
      pruneExpr(expr.body, live, exprToId)
    );
  }

  // CallExpr — always keep (it's marked live if we got here)
  const newArgs = expr.args.map(a => pruneExpr(a, live, exprToId));
  return new CallExpr(expr.op, newArgs, expr.attrs);
}

// ─── Count nodes for stats ───

function countNodes(expr: Expr): number {
  if (expr.kind === 'var' || expr.kind === 'constant') return 0;
  if (expr.kind === 'let') return countNodes(expr.value) + countNodes(expr.body);
  // call
  let count = 1;
  for (const arg of expr.args) count += countNodes(arg);
  return count;
}

// ═══════════════════════════════════════
//  Main DCE Pass
// ═══════════════════════════════════════

export interface DCEStats {
  totalBefore: number;
  totalAfter: number;
  eliminated: number;
}

export function deadCodeElimination(module: IRModule): { module: IRModule; stats: DCEStats } {
  const newModule = new IRModule();
  let totalBefore = 0;
  let totalAfter = 0;

  for (const [name, func] of module.functions) {
    const before = countNodes(func.body);
    totalBefore += before;

    const { tagged, rootId } = tagExprs(func.body);
    const live = markLive(tagged, rootId);

    // Build exprToId map for pruning
    const exprToId = new Map<Expr, number>();
    for (const t of tagged) {
      exprToId.set(t.expr, t.id);
    }

    const prunedBody = pruneExpr(func.body, live, exprToId);
    const after = countNodes(prunedBody);
    totalAfter += after;

    newModule.addFunction(new IRFunction(
      name,
      func.params,
      prunedBody,
      func.retType
    ));
  }

  return {
    module: newModule,
    stats: {
      totalBefore,
      totalAfter,
      eliminated: totalBefore - totalAfter,
    }
  };
}

// ═══════════════════════════════════════
//  Wrap module with a dead LetExpr to demonstrate DCE
//  Adds one dead binding representing training overhead not needed
//  for forward inference — DCE will correctly eliminate it.
// ═══════════════════════════════════════

export function wrapWithDeadCode(module: IRModule): IRModule {
  const newModule = new IRModule();
  for (const [name, func] of module.functions) {
    // A dead CallExpr with no args — represents discarded training context.
    // Its bound variable '_training_overhead' is never referenced in the forward body.
    const deadOp = new Op('nn.training_overhead', OpPattern.COMPLEX);
    const deadCall = new CallExpr(deadOp, [], {});
    const deadVar = new VarExpr('_training_overhead', new TensorType([1]));
    const newBody = new LetExpr(deadVar, deadCall, func.body);
    newModule.addFunction(new IRFunction(name, func.params, newBody, func.retType));
  }
  return newModule;
}
