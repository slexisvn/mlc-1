// ═══════════════════════════════════════════════════════════════
//  Constant Folding Pass
//  Post-order DFS: if ALL inputs to a Call are Constants,
//  evaluate at compile time and replace with Constant node.
// ═══════════════════════════════════════════════════════════════

import { Tensor } from '../tensor/tensor.js';
import {
  Expr, CallExpr, ConstantExpr, VarExpr, LetExpr,
  IRModule, IRFunction, TensorType
} from '../ir/high_level.js';

// Evaluate a call node with all-constant inputs
function evaluateConstantCall(op: string, args: Tensor[]): Tensor | null {
  switch (op) {
    case 'add':        return args[0].add(args[1]);
    case 'subtract':   return args[0].sub(args[1]);
    case 'multiply':   return args[0].mul(args[1]);
    case 'nn.relu': {
      const out = Tensor.zeros(args[0].shape);
      for (let i = 0; i < args[0].size; i++)
        out.data[i] = Math.max(args[0].data[i], 0);
      return out;
    }
    case 'nn.sigmoid': {
      const out = Tensor.zeros(args[0].shape);
      for (let i = 0; i < args[0].size; i++)
        out.data[i] = 1 / (1 + Math.exp(-args[0].data[i]));
      return out;
    }
    case 'nn.neg':     return args[0].neg();
    case 'nn.exp':     return args[0].exp();
    case 'nn.log':     return args[0].log();
    case 'nn.dense':   return args[0].matmul(args[1].transpose());
    case 'nn.bias_add': return args[0].add(args[1]);
    default:           return null; // Can't fold this op
  }
}

function foldExpr(expr: Expr): Expr {
  switch (expr.kind) {
    case 'var':
    case 'constant':
      return expr;

    case 'let':
      return new LetExpr(
        expr.varName,
        foldExpr(expr.value),
        foldExpr(expr.body)
      );

    case 'call': {
      // First, recursively fold args
      const foldedArgs = expr.args.map(a => foldExpr(a));

      // Check if all args are constants
      const allConstant = foldedArgs.every(a => a.kind === 'constant');
      if (allConstant) {
        const constArgs = foldedArgs.map(a => (a as ConstantExpr).data);
        const result = evaluateConstantCall(expr.op.name, constArgs);
        if (result) {
          return new ConstantExpr(result, `folded_${expr.op.name}`);
        }
      }

      return new CallExpr(expr.op, foldedArgs, expr.attrs);
    }
  }
}

export function constantFold(module: IRModule): IRModule {
  const newModule = new IRModule();

  for (const [name, func] of module.functions) {
    const foldedBody = foldExpr(func.body);
    newModule.addFunction(new IRFunction(
      name,
      func.params,
      foldedBody,
      func.retType
    ));
  }

  return newModule;
}
