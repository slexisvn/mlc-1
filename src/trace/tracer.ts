// ═══════════════════════════════════════════════════════════════
//  Symbolic Tracer — Captures the forward graph from a model
//  using meta tensors so inference can be compiled later.
// ═══════════════════════════════════════════════════════════════

import { Tensor } from '../tensor/tensor.js';
import { Module } from '../model/nn.js';
import { engine as globalEngine } from '../autograd/engine.js';

export interface TraceNode {
  id: number;
  op: string;
  inputs: number[];
  outputShape: number[];
  attrs: Record<string, any>;
}

export interface TraceGraph {
  nodes: TraceNode[];
  inputId: number;
  inputShape: number[];
  outputId: number;
  outputShape: number[];
  params: Map<string, { tensor: Tensor; shape: number[] }>;
}

let _traceNextId = 0;

export class Tracer {
  traceInference(model: Module, inputShape: number[]): TraceGraph {
    _traceNextId = 0;
    globalEngine.reset();

    const inputTensor = Tensor.meta(inputShape, false);
    const inputId = inputTensor.id;
    const output = model.forward(inputTensor);

    const params = new Map<string, { tensor: Tensor; shape: number[] }>();
    const allParams = model.parameters();
    for (let i = 0; i < allParams.length; i++) {
      const p = allParams[i];
      params.set(`param_${i}`, { tensor: p, shape: [...p.shape] });
    }

    const nodes: TraceNode[] = globalEngine.tape.map((entry) => ({
      id: entry.output.id,
      op: entry.op,
      inputs: entry.inputs.map(inp => inp.id),
      outputShape: [...entry.output.shape],
      attrs: {},
    }));

    return {
      nodes,
      inputId,
      inputShape: [...inputShape],
      outputId: output.id,
      outputShape: [...output.shape],
      params,
    };
  }

  static printGraph(graph: TraceGraph): string {
    const lines: string[] = ['═══ Traced Computation Graph ═══'];
    lines.push(`Input: id=${graph.inputId} shape=[${graph.inputShape}]`);
    lines.push('');

    for (let i = 0; i < graph.nodes.length; i++) {
      const node = graph.nodes[i];
      if (i === 0) lines.push('── Forward ──');
      const inputStr = node.inputs.map(id => `%${id}`).join(', ');
      lines.push(`  %${node.id} = ${node.op}(${inputStr}) → [${node.outputShape}]`);
    }

    lines.push('');
    lines.push(`Output: id=${graph.outputId} shape=[${graph.outputShape}]`);
    lines.push(`Parameters: ${graph.params.size}`);
    for (const [name, info] of graph.params) {
      lines.push(`  ${name}: [${info.shape}]`);
    }

    return lines.join('\n');
  }
}
