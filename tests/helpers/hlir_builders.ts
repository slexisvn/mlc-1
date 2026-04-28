// ─── High-Level IR fixture builders ───────────────────────────────

import { Tensor } from '../../src/tensor/tensor.js';
import {
  VarExpr, ConstantExpr, CallExpr, LetExpr,
  IRModule, IRFunction, TensorType, Op, OpPattern, OP_REGISTRY,
  type Expr,
} from '../../src/ir/high_level.js';

// ─── Leaf builders ───

export function mkVar(name: string, shape: number[]): VarExpr {
  return new VarExpr(name, new TensorType(shape));
}

export function mkConst(data: number[], shape: number[], name = 'c'): ConstantExpr {
  return new ConstantExpr(Tensor.fromArray(data, shape), name);
}

// ─── Call builder ───

export function mkCall(
  opName: string,
  args: Expr[],
  outputShape?: number[]
): CallExpr {
  const op = OP_REGISTRY[opName] ?? new Op(opName, OpPattern.INJECTIVE);
  const attrs: Record<string, any> = {};
  if (outputShape) attrs.outputShape = outputShape;
  return new CallExpr(op, args, attrs);
}

// ─── Module builder ───

export function mkModule(
  funcName: string,
  params: VarExpr[],
  body: Expr,
  retShape: number[]
): IRModule {
  const mod = new IRModule();
  mod.addFunction(
    new IRFunction(funcName, params, body, new TensorType(retShape))
  );
  return mod;
}

// ─── Pre-built fixture: dense → bias_add → relu chain ───
//
//  Creates a chain of:
//    let nn.dense(x, W) [M×K, N×K → M×N]
//    let nn.bias_add(dense, B) [M×N, 1×N → M×N]
//    nn.relu(biasAdd) [M×N → M×N]
//
//  Suitable for testing fuseOps (triggers dense_bias_relu fusion).
//  Shape attrs are set on each call for compatibility with shape-aware passes.

export function mkDenseBiasReluModule(
  M: number,
  N: number,
  K: number
): {
  mod: IRModule;
  denseCall: CallExpr;
  biasAddCall: CallExpr;
  reluCall: CallExpr;
} {
  const x = mkVar('x', [M, K]);
  const W = mkConst(new Array(N * K).fill(0.1), [N, K], 'W');
  const B = mkConst(new Array(N).fill(0.0), [1, N], 'B');

  const denseCall = mkCall('nn.dense', [x, W], [M, N]);
  const biasAddCall = mkCall('nn.bias_add', [denseCall, B], [M, N]);
  const reluCall = mkCall('nn.relu', [biasAddCall], [M, N]);

  const mod = mkModule('forward', [x], reluCall, [M, N]);
  return { mod, denseCall, biasAddCall, reluCall };
}

// ─── Pre-built fixture: dense → bias_add only (no activation) ───

export function mkDenseBiasModule(
  M: number,
  N: number,
  K: number
): {
  mod: IRModule;
  denseCall: CallExpr;
  biasAddCall: CallExpr;
} {
  const x = mkVar('x', [M, K]);
  const W = mkConst(new Array(N * K).fill(0.1), [N, K], 'W');
  const B = mkConst(new Array(N).fill(0.0), [1, N], 'B');

  const denseCall = mkCall('nn.dense', [x, W], [M, N]);
  const biasAddCall = mkCall('nn.bias_add', [denseCall, B], [M, N]);

  const mod = mkModule('forward', [x], biasAddCall, [M, N]);
  return { mod, denseCall, biasAddCall };
}

// ─── Pre-built fixture: dense with 2 bias_add consumers (multi-consumer guard) ───

export function mkMultiConsumerModule(
  M: number,
  N: number,
  K: number
): { mod: IRModule; denseCall: CallExpr } {
  const x = mkVar('x', [M, K]);
  const W = mkConst(new Array(N * K).fill(0.1), [N, K], 'W');
  const B = mkConst(new Array(N).fill(0.0), [1, N], 'B');

  // Same `denseCall` object used by both biasAdd1 and biasAdd2 → 2 consumers
  const denseCall = mkCall('nn.dense', [x, W], [M, N]);
  const biasAdd1 = mkCall('nn.bias_add', [denseCall, B], [M, N]);
  const biasAdd2 = mkCall('nn.bias_add', [denseCall, B], [M, N]);
  const addResult = mkCall('add', [biasAdd1, biasAdd2], [M, N]);

  const mod = mkModule('forward', [x], addResult, [M, N]);
  return { mod, denseCall };
}
