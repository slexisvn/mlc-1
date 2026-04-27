// ═══════════════════════════════════════════════════════════════
//  Operator Fusion Pass
//  Fuses sequences of ops into single fused kernels.
//  Rules:
//    1. REDUCTION + trailing INJECTIVE(s) → fuse
//    2. INJECTIVE chain → fuse
//    3. COMPLEX ops like softmax+CE → special fused patterns
// ═══════════════════════════════════════════════════════════════

import {
  Expr, CallExpr, ConstantExpr, VarExpr, LetExpr,
  IRModule, IRFunction, Op, OpPattern, OP_REGISTRY, TensorType
} from '../ir/high_level.js';

// ─── Fusion Group ───

interface FusionGroup {
  ops: string[];              // ordered op names in this group
  fusedName: string;          // name of fused op
  pattern: OpPattern;         // dominant pattern
}

// ─── Known Fusion Patterns ───

const FUSION_PATTERNS: { match: string[]; fusedOp: string }[] = [
  // Forward fusions
  { match: ['nn.dense', 'nn.bias_add', 'nn.relu'],       fusedOp: 'fused.dense_bias_relu' },
  { match: ['nn.dense', 'nn.bias_add', 'nn.sigmoid'],    fusedOp: 'fused.dense_bias_sigmoid' },
  { match: ['nn.dense', 'nn.bias_add', 'nn.tanh'],       fusedOp: 'fused.dense_bias_tanh' },
  { match: ['nn.dense', 'nn.bias_add', 'nn.leaky_relu'], fusedOp: 'fused.dense_bias_leaky_relu' },
  { match: ['nn.dense', 'nn.bias_add'],                  fusedOp: 'fused.dense_bias' },
  // Loss fusions
  { match: ['nn.softmax', 'cross_entropy'],               fusedOp: 'fused.softmax_ce' },
  { match: ['nn.sigmoid', 'bce_with_logits'],             fusedOp: 'fused.sigmoid_bce' },
];

// ─── Flatten IR into linear sequence of Calls ───

interface LinearNode {
  id: number;
  expr: CallExpr;
  opName: string;
  inputIds: number[];
  outputUsedBy: number[];   // which nodes consume this output
}

function linearize(expr: Expr): { nodes: LinearNode[]; leafExprs: Map<number, Expr> } {
  const nodes: LinearNode[] = [];
  const leafExprs = new Map<number, Expr>();
  let nextId = 0;
  const exprToId = new Map<Expr, number>();

  function visit(e: Expr): number {
    if (exprToId.has(e)) return exprToId.get(e)!;

    if (e.kind === 'var' || e.kind === 'constant') {
      const id = nextId++;
      exprToId.set(e, id);
      leafExprs.set(id, e);
      return id;
    }

    if (e.kind === 'let') {
      visit(e.value);
      return visit(e.body);
    }

    // CallExpr
    const inputIds = e.args.map(arg => visit(arg));
    const id = nextId++;
    exprToId.set(e, id);
    nodes.push({
      id,
      expr: e,
      opName: e.op.name,
      inputIds,
      outputUsedBy: [],
    });
    return id;
  }

  visit(expr);

  // Build outputUsedBy
  for (const node of nodes) {
    for (const inId of node.inputIds) {
      const producer = nodes.find(n => n.id === inId);
      if (producer) producer.outputUsedBy.push(node.id);
    }
  }

  return { nodes, leafExprs };
}

// ─── Try to match fusion patterns ───

function tryFuse(nodes: LinearNode[]): { groups: FusionGroup[]; membership: Map<number, number> } {
  const groups: FusionGroup[] = [];
  const membership = new Map<number, number>(); // node id → group index
  const consumed = new Set<number>();

  for (let i = 0; i < nodes.length; i++) {
    if (consumed.has(nodes[i].id)) continue;

    // Try each fusion pattern
    let matched = false;
    for (const pattern of FUSION_PATTERNS) {
      if (nodes[i].opName !== pattern.match[0]) continue;

      // Try to match the rest of the pattern
      const chain: LinearNode[] = [nodes[i]];
      let current = nodes[i];
      let canFuse = true;

      for (let p = 1; p < pattern.match.length; p++) {
        // Current node must have exactly 1 consumer
        if (current.outputUsedBy.length !== 1) { canFuse = false; break; }

        const nextNode = nodes.find(n => n.id === current.outputUsedBy[0]);
        if (!nextNode || nextNode.opName !== pattern.match[p] || consumed.has(nextNode.id)) {
          canFuse = false;
          break;
        }
        chain.push(nextNode);
        current = nextNode;
      }

      if (canFuse && chain.length === pattern.match.length) {
        const groupIdx = groups.length;
        groups.push({
          ops: pattern.match,
          fusedName: pattern.fusedOp,
          pattern: OpPattern.REDUCTION,
        });
        for (const n of chain) {
          membership.set(n.id, groupIdx);
          consumed.add(n.id);
        }
        matched = true;
        break;
      }
    }

    // If not matched, standalone node
    if (!matched && !consumed.has(nodes[i].id)) {
      const groupIdx = groups.length;
      groups.push({
        ops: [nodes[i].opName],
        fusedName: nodes[i].opName,
        pattern: OP_REGISTRY[nodes[i].opName]?.pattern || OpPattern.INJECTIVE,
      });
      membership.set(nodes[i].id, groupIdx);
    }
  }

  return { groups, membership };
}

// ─── Apply fusion: rebuild IR with fused ops ───

function rebuildFused(
  nodes: LinearNode[],
  leafExprs: Map<number, Expr>,
  groups: FusionGroup[],
  membership: Map<number, number>
): Expr {
  const idToExpr = new Map<number, Expr>();

  // Copy leaf exprs
  for (const [id, expr] of leafExprs) {
    idToExpr.set(id, expr);
  }

  // Process groups in order
  const processedGroups = new Set<number>();

  for (const node of nodes) {
    const groupIdx = membership.get(node.id);
    if (groupIdx === undefined) continue;
    if (processedGroups.has(groupIdx)) continue;
    processedGroups.add(groupIdx);

    const group = groups[groupIdx];
    const groupNodes = nodes.filter(n => membership.get(n.id) === groupIdx);

    if (groupNodes.length === 1) {
      // No fusion, just rebuild
      const n = groupNodes[0];
      const args = n.inputIds.map(id => idToExpr.get(id)!).filter(Boolean);
      const call = new CallExpr(n.expr.op, args, n.expr.attrs);
      idToExpr.set(n.id, call);
    } else {
      // Fused group: collect all external inputs
      const internalIds = new Set(groupNodes.map(n => n.id));
      const externalInputs: Expr[] = [];
      const seenInputIds = new Set<number>();

      for (const n of groupNodes) {
        for (const inId of n.inputIds) {
          if (!internalIds.has(inId) && !seenInputIds.has(inId)) {
            seenInputIds.add(inId);
            const expr = idToExpr.get(inId);
            if (expr) externalInputs.push(expr);
          }
        }
      }

      // Create fused op
      const fusedOp = OP_REGISTRY[group.fusedName] || new Op(group.fusedName, group.pattern);
      const fusedCall = new CallExpr(fusedOp, externalInputs, {
        fusedOps: group.ops,
        outputShape: groupNodes[groupNodes.length - 1].expr.attrs.outputShape,
      });

      // Map the last node in the group to this fused call
      const lastNode = groupNodes[groupNodes.length - 1];
      idToExpr.set(lastNode.id, fusedCall);
    }
  }

  // Return the last expression
  const lastNode = nodes[nodes.length - 1];
  return idToExpr.get(lastNode.id)!;
}

// ─── Main Fusion Pass ───

export function fuseOps(module: IRModule): IRModule {
  const newModule = new IRModule();

  for (const [name, func] of module.functions) {
    const { nodes, leafExprs } = linearize(func.body);

    if (nodes.length === 0) {
      newModule.addFunction(func);
      continue;
    }

    const { groups, membership } = tryFuse(nodes);
    const fusedBody = rebuildFused(nodes, leafExprs, groups, membership);

    newModule.addFunction(new IRFunction(
      name,
      func.params,
      fusedBody,
      func.retType
    ));

    // Print fusion summary
    const fusedGroups = groups.filter(g => g.ops.length > 1);
    if (fusedGroups.length > 0) {
      console.log(`  Fusion applied in '${name}':`);
      for (const g of fusedGroups) {
        console.log(`    ${g.ops.join(' + ')} → ${g.fusedName}`);
      }
    }
  }

  return newModule;
}

// ─── Fusion Statistics ───

export function fusionStats(module: IRModule): {
  totalOps: number;
  fusedOps: number;
  fusedGroups: string[];
} {
  let totalOps = 0;
  let fusedOps = 0;
  const fusedGroups: string[] = [];

  for (const [, func] of module.functions) {
    countOps(func.body);
  }

  function countOps(expr: Expr): void {
    if (expr.kind === 'call') {
      totalOps++;
      if (expr.op.name.startsWith('fused.')) {
        fusedOps++;
        const ops = (expr.attrs.fusedOps as string[]) || [];
        fusedGroups.push(`${ops.join('+')} → ${expr.op.name}`);
      }
      for (const arg of expr.args) countOps(arg);
    } else if (expr.kind === 'let') {
      countOps(expr.value);
      countOps(expr.body);
    }
  }

  return { totalOps, fusedOps, fusedGroups };
}
