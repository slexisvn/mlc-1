// ═══════════════════════════════════════════════════════════════
//  Symbolic Tracer — Captures computation graph from model
//  Instead of executing ops, it records them as TraceNodes.
//  Captures both forward and backward subgraphs.
// ═══════════════════════════════════════════════════════════════

import { NDArray } from '../tensor/ndarray.js';
import { Module } from '../model/nn.js';
import { Loss } from '../loss/loss.js';
import { GradTensor, AutogradEngine, engine as globalEngine } from '../autograd/engine.js';

export interface TraceNode {
  id: number;
  op: string;
  inputs: number[];      // ids of input trace tensors
  outputShape: number[];
  attrs: Record<string, any>;
}

export interface TraceGraph {
  nodes: TraceNode[];
  inputId: number;
  inputShape: number[];
  outputId: number;
  outputShape: number[];
  targetId?: number;
  targetShape?: number[];
  // Forward vs backward boundary
  forwardEndIdx: number;      // nodes[0..forwardEndIdx-1] = forward
  backwardStartIdx: number;   // nodes[backwardStartIdx..end] = backward
  // Parameter info
  params: Map<string, { tensor: GradTensor; shape: number[] }>;
  // Loss value node
  lossId?: number;
  backwardSeedId?: number;
  gradientIds: Map<number, number>;
}

let _traceNextId = 0;

export class Tracer {
  // Trace forward-only (inference)
  traceInference(model: Module, inputShape: number[]): TraceGraph {
    _traceNextId = 0;
    globalEngine.reset();

    const inputData = NDArray.rand(inputShape);
    const inputTensor = new GradTensor(inputData, false);
    const inputId = inputTensor.id;

    const output = model.forward(inputTensor);

    // Collect params
    const params = new Map<string, { tensor: GradTensor; shape: number[] }>();
    const allParams = model.parameters();
    for (let i = 0; i < allParams.length; i++) {
      const p = allParams[i];
      const name = `param_${i}`;
      params.set(name, { tensor: p, shape: [...p.data.shape] });
    }

    // Convert tape entries to trace nodes
    const nodes: TraceNode[] = globalEngine.tape.map((entry) => ({
      id: entry.output.id,
      op: entry.op,
      inputs: entry.inputs.map(inp => inp.id),
      outputShape: [...entry.output.data.shape],
      attrs: {},
    }));

    return {
      nodes,
      inputId,
      inputShape: [...inputShape],
      outputId: output.id,
      outputShape: [...output.data.shape],
      forwardEndIdx: nodes.length,
      backwardStartIdx: nodes.length,
      params,
      gradientIds: new Map(),
    };
  }

  // Trace full training step (forward + loss + backward)
  traceTraining(
    model: Module,
    lossFunc: Loss,
    inputShape: number[],
    targetShape: number[]
  ): TraceGraph {
    _traceNextId = 0;
    globalEngine.reset();

    // Forward
    const inputData = NDArray.rand(inputShape);
    const inputTensor = new GradTensor(inputData, false);
    const inputId = inputTensor.id;

    const targetData = NDArray.zeros(targetShape);
    // Put a valid class index for cross entropy
    if (targetShape.length === 1) {
      targetData.data[0] = Math.floor(Math.random() * 10);
    }
    const targetTensor = new GradTensor(targetData, false);
    const targetId = targetTensor.id;

    const output = model.forward(inputTensor);
    const forwardEndIdx = globalEngine.tape.length;

    // Loss
    const loss = lossFunc.forward(output, targetTensor);
    const lossIdx = globalEngine.tape.length;

    // Backward
    globalEngine.backward(loss);
    const totalNodes = globalEngine.tape.length;

    // Collect params
    const params = new Map<string, { tensor: GradTensor; shape: number[] }>();
    const allParams = model.parameters();
    for (let i = 0; i < allParams.length; i++) {
      const p = allParams[i];
      params.set(`param_${i}`, { tensor: p, shape: [...p.data.shape] });
    }

    const forwardAndLossNodes: TraceNode[] = globalEngine.tape.map((entry) => ({
      id: entry.output.id,
      op: entry.op,
      inputs: entry.inputs.map(inp => inp.id),
      outputShape: [...entry.output.data.shape],
      attrs: {},
    }));

    const backwardNodes: TraceNode[] = globalEngine.backwardTrace.map((node) => ({
      id: node.id,
      op: node.op,
      inputs: [...node.inputs],
      outputShape: [...node.outputShape],
      attrs: { ...node.attrs },
    }));

    const nodes = [...forwardAndLossNodes, ...backwardNodes];

    return {
      nodes,
      inputId,
      inputShape: [...inputShape],
      outputId: output.id,
      outputShape: [...output.data.shape],
      targetId,
      targetShape: [...targetShape],
      forwardEndIdx,
      backwardStartIdx: lossIdx,
      lossId: loss.id,
      backwardSeedId: globalEngine.backwardSeedId,
      gradientIds: new Map(globalEngine.gradientTensorIds),
      params,
    };
  }

  // Pretty-print the traced graph
  static printGraph(graph: TraceGraph): string {
    const lines: string[] = ['═══ Traced Computation Graph ═══'];
    lines.push(`Input: id=${graph.inputId} shape=[${graph.inputShape}]`);
    if (graph.targetId !== undefined) {
      lines.push(`Target: id=${graph.targetId}`);
    }
    lines.push('');

    for (let i = 0; i < graph.nodes.length; i++) {
      const node = graph.nodes[i];
      let section = '';
      if (i === 0) section = '── Forward ──';
      if (i === graph.forwardEndIdx) section = '── Loss ──';
      if (i === graph.backwardStartIdx && i !== graph.forwardEndIdx) section = '── Backward ──';
      if (section) lines.push(section);

      const inputStr = node.inputs.map(id => `%${id}`).join(', ');
      lines.push(`  %${node.id} = ${node.op}(${inputStr}) → [${node.outputShape}]`);
    }

    lines.push('');
    lines.push(`Output: id=${graph.outputId} shape=[${graph.outputShape}]`);
    if (graph.lossId !== undefined) lines.push(`Loss: id=${graph.lossId}`);
    lines.push(`Parameters: ${graph.params.size}`);
    for (const [name, info] of graph.params) {
      lines.push(`  ${name}: [${info.shape}]`);
    }

    return lines.join('\n');
  }
}
