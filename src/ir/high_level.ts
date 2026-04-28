// ═══════════════════════════════════════════════════════════════
//  High-Level IR — Relay-like functional IR for forward graphs
// ═══════════════════════════════════════════════════════════════

import { Tensor } from '../tensor/tensor.js';
import { TraceGraph } from '../trace/tracer.js';

export type Expr = VarExpr | ConstantExpr | CallExpr | LetExpr;

export class VarExpr {
  readonly kind = 'var' as const;
  constructor(
    public name: string,
    public type: TensorType
  ) {}
}

export class ConstantExpr {
  readonly kind = 'constant' as const;
  constructor(
    public data: Tensor,
    public name: string = ''
  ) {}
}

export class CallExpr {
  readonly kind = 'call' as const;
  constructor(
    public op: Op,
    public args: Expr[],
    public attrs: Record<string, any> = {}
  ) {}
}

export class LetExpr {
  readonly kind = 'let' as const;
  constructor(
    public varName: VarExpr,
    public value: Expr,
    public body: Expr
  ) {}
}

export class TensorType {
  constructor(
    public shape: number[],
    public dtype: string = 'float32'
  ) {}
  toString(): string {
    return `Tensor[(${this.shape.join(', ')}), ${this.dtype}]`;
  }
}

export enum OpPattern {
  INJECTIVE = 'injective',
  REDUCTION = 'reduction',
  COMPLEX = 'complex',
}

export class Op {
  constructor(
    public name: string,
    public pattern: OpPattern,
    public inferShape?: (inputShapes: number[][], attrs: Record<string, any>) => number[]
  ) {}
}

export const OP_REGISTRY: Record<string, Op> = {
  'nn.dense': new Op('nn.dense', OpPattern.REDUCTION),
  'nn.bias_add': new Op('nn.bias_add', OpPattern.INJECTIVE),
  'nn.relu': new Op('nn.relu', OpPattern.INJECTIVE),
  'nn.sigmoid': new Op('nn.sigmoid', OpPattern.INJECTIVE),
  'nn.tanh': new Op('nn.tanh', OpPattern.INJECTIVE),
  'nn.leaky_relu': new Op('nn.leaky_relu', OpPattern.INJECTIVE),
  'nn.softmax': new Op('nn.softmax', OpPattern.COMPLEX),
  'nn.exp': new Op('nn.exp', OpPattern.INJECTIVE),
  'nn.log': new Op('nn.log', OpPattern.INJECTIVE),
  'nn.neg': new Op('nn.neg', OpPattern.INJECTIVE),
  'multiply': new Op('multiply', OpPattern.INJECTIVE),
  'add': new Op('add', OpPattern.INJECTIVE),
  'subtract': new Op('subtract', OpPattern.INJECTIVE),
  'reduce_sum': new Op('reduce_sum', OpPattern.REDUCTION),
  'reduce_max': new Op('reduce_max', OpPattern.REDUCTION),
  'reduce_mean': new Op('reduce_mean', OpPattern.REDUCTION),
  'cross_entropy': new Op('cross_entropy', OpPattern.COMPLEX),
  'bce_with_logits': new Op('bce_with_logits', OpPattern.COMPLEX),
  'mse': new Op('mse', OpPattern.COMPLEX),
  'fused.dense_bias_relu': new Op('fused.dense_bias_relu', OpPattern.REDUCTION),
  'fused.dense_bias_sigmoid': new Op('fused.dense_bias_sigmoid', OpPattern.REDUCTION),
  'fused.dense_bias_tanh': new Op('fused.dense_bias_tanh', OpPattern.REDUCTION),
  'fused.dense_bias': new Op('fused.dense_bias', OpPattern.REDUCTION),
  'fused.softmax_ce': new Op('fused.softmax_ce', OpPattern.COMPLEX),
  'fused.sigmoid_bce': new Op('fused.sigmoid_bce', OpPattern.COMPLEX),
  'optim.sgd_update': new Op('optim.sgd_update', OpPattern.INJECTIVE),
};

const TRACE_TO_IR_OP: Record<string, string> = {
  'matmul': 'nn.dense',
  'bias_add': 'nn.bias_add',
  'relu': 'nn.relu',
  'sigmoid': 'nn.sigmoid',
  'tanh': 'nn.tanh',
  'leaky_relu': 'nn.leaky_relu',
  'softmax': 'nn.softmax',
  'log': 'nn.log',
  'exp': 'nn.exp',
  'neg': 'nn.neg',
  'multiply': 'multiply',
  'add': 'add',
  'subtract': 'subtract',
  'sum': 'reduce_sum',
  'mean': 'reduce_mean',
  'cross_entropy': 'cross_entropy',
  'bce_with_logits': 'bce_with_logits',
  'mse': 'mse',
};

export class IRFunction {
  constructor(
    public name: string,
    public params: VarExpr[],
    public body: Expr,
    public retType: TensorType
  ) {}
}

export class IRModule {
  functions: Map<string, IRFunction> = new Map();

  addFunction(func: IRFunction): void {
    this.functions.set(func.name, func);
  }

  getFunction(name: string): IRFunction | undefined {
    return this.functions.get(name);
  }
}

export function buildIR(graph: TraceGraph): IRModule {
  const module = new IRModule();
  const inputVar = new VarExpr('x', new TensorType(graph.inputShape));
  const exprMap = new Map<number, Expr>();
  exprMap.set(graph.inputId, inputVar);

  for (const [name, info] of graph.params) {
    exprMap.set(info.tensor.id, new ConstantExpr(info.tensor, name));
  }

  for (const node of graph.nodes) {
    const irOpName = TRACE_TO_IR_OP[node.op] || node.op;
    const op = OP_REGISTRY[irOpName] ?? new Op(irOpName, OpPattern.INJECTIVE);
    const args = node.inputs
      .map(id => exprMap.get(id))
      .filter((e): e is Expr => e !== undefined);
    exprMap.set(node.id, new CallExpr(op, args, { ...node.attrs, outputShape: node.outputShape }));
  }

  const bodyExpr = exprMap.get(graph.outputId);
  if (!bodyExpr) throw new Error('Could not build IR: missing forward output expression');

  const liveNodeIds = new Set<number>();
  const nodeById = new Map<number, TraceGraph['nodes'][number]>();
  for (const node of graph.nodes) nodeById.set(node.id, node);

  const worklist = [graph.outputId];
  while (worklist.length > 0) {
    const id = worklist.pop()!;
    if (liveNodeIds.has(id)) continue;
    liveNodeIds.add(id);
    const node = nodeById.get(id);
    if (!node) continue;
    for (const inputId of node.inputs) {
      if (nodeById.has(inputId) && !liveNodeIds.has(inputId)) {
        worklist.push(inputId);
      }
    }
  }

  let wrappedBody: Expr = bodyExpr;
  let deadBindingIndex = 0;
  for (const node of graph.nodes) {
    if (liveNodeIds.has(node.id)) continue;
    const deadExpr = exprMap.get(node.id);
    if (!deadExpr) continue;
    const deadVar = new VarExpr(`_dead_${deadBindingIndex++}`, new TensorType(node.outputShape));
    wrappedBody = new LetExpr(deadVar, deadExpr, wrappedBody);
  }

  const retShape = bodyExpr.kind === 'call'
    ? ((bodyExpr.attrs.outputShape as number[] | undefined) ?? graph.outputShape)
    : graph.outputShape;

  module.addFunction(new IRFunction(
    'main',
    [inputVar],
    wrappedBody,
    new TensorType(retShape)
  ));

  return module;
}
