// ═══════════════════════════════════════════════════════════════
//  High-Level IR — Relay-like functional IR
//  Represents computation as a tree of expressions.
//  Each node is a high-level tensor operation.
// ═══════════════════════════════════════════════════════════════

import { NDArray } from '../tensor/ndarray.js';
import { TraceGraph } from '../trace/tracer.js';

// ─── Expression Types ───

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
    public data: NDArray,
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

// ─── Types ───

export class TensorType {
  constructor(
    public shape: number[],
    public dtype: string = 'float32'
  ) {}
  toString(): string {
    return `Tensor[(${this.shape.join(', ')}), ${this.dtype}]`;
  }
}

// ─── Op & Op Pattern ───

export enum OpPattern {
  INJECTIVE = 'injective',   // 1-to-1 element mapping (relu, sigmoid, add...)
  REDUCTION = 'reduction',   // many-to-1 (matmul, sum, conv...)
  COMPLEX = 'complex',       // multi-stage (softmax, batch_norm...)
}

export class Op {
  constructor(
    public name: string,
    public pattern: OpPattern,
    public inferShape?: (inputShapes: number[][], attrs: Record<string, any>) => number[]
  ) {}
}

// ─── Op Registry ───

export const OP_REGISTRY: Record<string, Op> = {
  // Forward ops
  'nn.dense':        new Op('nn.dense',        OpPattern.REDUCTION),
  'nn.bias_add':     new Op('nn.bias_add',     OpPattern.INJECTIVE),
  'nn.relu':         new Op('nn.relu',         OpPattern.INJECTIVE),
  'nn.sigmoid':      new Op('nn.sigmoid',      OpPattern.INJECTIVE),
  'nn.tanh':         new Op('nn.tanh',         OpPattern.INJECTIVE),
  'nn.leaky_relu':   new Op('nn.leaky_relu',   OpPattern.INJECTIVE),
  'nn.softmax':      new Op('nn.softmax',      OpPattern.COMPLEX),
  'nn.exp':          new Op('nn.exp',          OpPattern.INJECTIVE),
  'nn.log':          new Op('nn.log',          OpPattern.INJECTIVE),
  'nn.neg':          new Op('nn.neg',          OpPattern.INJECTIVE),
  'multiply':        new Op('multiply',        OpPattern.INJECTIVE),
  'add':             new Op('add',             OpPattern.INJECTIVE),
  'subtract':        new Op('subtract',        OpPattern.INJECTIVE),
  'reduce_sum':      new Op('reduce_sum',      OpPattern.REDUCTION),
  'reduce_max':      new Op('reduce_max',      OpPattern.REDUCTION),
  'reduce_mean':     new Op('reduce_mean',     OpPattern.REDUCTION),

  // Loss ops
  'cross_entropy':      new Op('cross_entropy',      OpPattern.COMPLEX),
  'bce_with_logits':    new Op('bce_with_logits',    OpPattern.COMPLEX),
  'mse':                new Op('mse',                OpPattern.COMPLEX),

  // Backward ops
  'nn.dense_grad_data':     new Op('nn.dense_grad_data',     OpPattern.REDUCTION),
  'nn.dense_grad_weight':   new Op('nn.dense_grad_weight',   OpPattern.REDUCTION),
  'nn.relu_grad':           new Op('nn.relu_grad',           OpPattern.INJECTIVE),
  'nn.sigmoid_grad':        new Op('nn.sigmoid_grad',        OpPattern.INJECTIVE),
  'nn.tanh_grad':           new Op('nn.tanh_grad',           OpPattern.INJECTIVE),
  'nn.leaky_relu_grad':     new Op('nn.leaky_relu_grad',     OpPattern.INJECTIVE),
  'nn.softmax_grad':        new Op('nn.softmax_grad',        OpPattern.COMPLEX),
  'nn.bias_add_grad':       new Op('nn.bias_add_grad',       OpPattern.REDUCTION),
  'nn.cross_entropy_grad':  new Op('nn.cross_entropy_grad',  OpPattern.INJECTIVE),
  'nn.bce_grad':            new Op('nn.bce_grad',            OpPattern.INJECTIVE),

  // Fused ops (created by fusion pass)
  'fused.dense_bias_relu':      new Op('fused.dense_bias_relu',      OpPattern.REDUCTION),
  'fused.dense_bias_sigmoid':   new Op('fused.dense_bias_sigmoid',   OpPattern.REDUCTION),
  'fused.dense_bias_tanh':      new Op('fused.dense_bias_tanh',      OpPattern.REDUCTION),
  'fused.dense_bias':           new Op('fused.dense_bias',           OpPattern.REDUCTION),
  'fused.softmax_ce':           new Op('fused.softmax_ce',           OpPattern.COMPLEX),
  'fused.sigmoid_bce':          new Op('fused.sigmoid_bce',          OpPattern.COMPLEX),

  // Optimizer
  'optim.sgd_update':  new Op('optim.sgd_update',  OpPattern.INJECTIVE),
};

// Map trace op names to IR op names
const TRACE_TO_IR_OP: Record<string, string> = {
  'matmul':          'nn.dense',
  'bias_add':        'nn.bias_add',
  'relu':            'nn.relu',
  'sigmoid':         'nn.sigmoid',
  'tanh':            'nn.tanh',
  'leaky_relu':      'nn.leaky_relu',
  'softmax':         'nn.softmax',
  'log':             'nn.log',
  'exp':             'nn.exp',
  'neg':             'nn.neg',
  'multiply':        'multiply',
  'add':             'add',
  'subtract':        'subtract',
  'sum':             'reduce_sum',
  'mean':            'reduce_mean',
  'cross_entropy':   'cross_entropy',
  'bce_with_logits': 'bce_with_logits',
  'mse':             'mse',
};

// ─── IR Module ───

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

// ─── Build IR from TraceGraph ───

export function buildIR(graph: TraceGraph): IRModule {
  const module = new IRModule();

  // Create input var
  const inputVar = new VarExpr('x', new TensorType(graph.inputShape));
  const exprMap = new Map<number, Expr>();
  exprMap.set(graph.inputId, inputVar);

  // Map param tensor ids to constants
  for (const [name, info] of graph.params) {
    exprMap.set(info.tensor.id, new ConstantExpr(info.tensor.data, name));
  }

  // Map target if exists
  if (graph.targetId !== undefined) {
    const targetVar = new VarExpr('target', new TensorType([1]));
    exprMap.set(graph.targetId, targetVar);
  }

  // Build expression tree from trace nodes
  for (const node of graph.nodes) {
    const irOpName = TRACE_TO_IR_OP[node.op] || node.op;
    const op = OP_REGISTRY[irOpName];
    if (!op) {
      // Unknown op, create a generic one
      const genericOp = new Op(irOpName, OpPattern.INJECTIVE);
      const args = node.inputs
        .map(id => exprMap.get(id))
        .filter((e): e is Expr => e !== undefined);
      const call = new CallExpr(genericOp, args, node.attrs);
      exprMap.set(node.id, call);
      continue;
    }

    const args = node.inputs
      .map(id => exprMap.get(id))
      .filter((e): e is Expr => e !== undefined);

    // For matmul that takes a transposed weight, we note this in attrs
    const call = new CallExpr(op, args, { ...node.attrs, outputShape: node.outputShape });
    exprMap.set(node.id, call);
  }

  // Get the final expression
  const lastNodeId = graph.nodes.length > 0
    ? graph.nodes[graph.nodes.length - 1].id
    : graph.outputId;
  const bodyExpr = exprMap.get(lastNodeId) || exprMap.get(graph.outputId);

  if (!bodyExpr) throw new Error('Could not build IR: missing output expression');

  // Create main function with Let bindings for readability
  const params = [inputVar];
  if (graph.targetId !== undefined) {
    params.push(new VarExpr('target', new TensorType([1])));
  }

  const retShape = graph.lossId ? [1] : graph.outputShape;
  const mainFunc = new IRFunction(
    'main',
    params,
    bodyExpr,
    new TensorType(retShape)
  );

  module.addFunction(mainFunc);
  return module;
}
