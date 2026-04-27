// ═══════════════════════════════════════════════════════════════
//  JavaScript Code Generation
//  Converts TensorIR (PrimFunc) into a JavaScript function string
//  that can be eval()'d and executed.
// ═══════════════════════════════════════════════════════════════

import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, AllocNode,
  BufferLoadExpr, BinOpExpr, ConstExpr, VarRefExpr, MaxExpr,
  MinExpr, CallExprTIR,
  VarIndex, ConstIndex, BinOpIndex,
  type Stmt, type ValueExpr, type IndexExpr, BufferDecl
} from '../ir/low_level.js';
import {
  ScalarStoreNode, ScalarLoadExpr, ScalarDeclNode
} from '../transform/storage_rewrite.js';

export function codegenJS(func: PrimFunc): string {
  const gen = new JSCodeGenerator(func);
  return gen.generate();
}

class JSCodeGenerator {
  private func: PrimFunc;
  private indent = 0;
  private lines: string[] = [];
  private bufferStrides: Map<string, number[]> = new Map();

  constructor(func: PrimFunc) {
    this.func = func;
    // Precompute strides for all buffers
    for (const param of func.params) {
      this.bufferStrides.set(param.name, this.computeStrides(param.shape));
    }
    for (const alloc of func.allocations) {
      this.bufferStrides.set(alloc.name, this.computeStrides(alloc.shape));
    }
  }

  private computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  generate(): string {
    const paramNames = this.func.params.map(p => p.name);
    this.lines = [];
    this.indent = 0;

    // Function header
    this.emit(`function ${this.func.name}(${paramNames.join(', ')}) {`);
    this.indent++;

    // Comments: buffer shapes
    this.emit(`// Buffer shapes:`);
    for (const p of this.func.params) {
      this.emit(`// ${p.name}: [${p.shape.join(', ')}] (${p.scope})`);
    }
    this.emit('');

    // Local allocations (arrays)
    for (const alloc of this.func.allocations) {
      const size = alloc.shape.reduce((a, b) => a * b, 1);
      this.emit(`const ${alloc.name} = new Float32Array(${size}); // local`);
    }

    // Scalar declarations (from storage rewrite pass)
    const scalarNames = this.collectScalarNames(this.func.body);
    for (const name of scalarNames) {
      this.emit(`let ${name} = 0; // scalar (promoted from buffer)`);
    }
    if (this.func.allocations.length > 0 || scalarNames.size > 0) this.emit('');

    // Generate body
    this.genStmt(this.func.body);

    this.indent--;
    this.emit('}');

    return this.lines.join('\n');
  }

  private genStmt(stmt: Stmt): void {
    if (stmt instanceof ForNode) {
      this.genFor(stmt);
    } else if (stmt instanceof SeqNode) {
      for (const s of stmt.stmts) this.genStmt(s);
    } else if (stmt instanceof BufferStoreNode) {
      this.genStore(stmt);
    } else if (stmt instanceof AllocNode) {
      const size = stmt.buffer.shape.reduce((a, b) => a * b, 1);
      this.emit(`const ${stmt.buffer.name} = new Float32Array(${size});`);
      this.genStmt(stmt.body);
    } else if ((stmt as any) instanceof ScalarStoreNode) {
      const s = stmt as any as ScalarStoreNode;
      this.emit(`${s.scalarName} = ${this.genValue(s.value)};`);
    } else if ((stmt as any) instanceof ScalarDeclNode) {
      const s = stmt as any as ScalarDeclNode;
      this.emit(`let ${s.scalarName} = ${s.initValue};`);
      this.genStmt(s.body as Stmt);
    }
  }

  private genFor(node: ForNode): void {
    const v = node.loopVar.name;
    const annotation = node.annotation !== 'none' ? ` /* ${node.annotation} */` : '';

    if (node.annotation === 'unroll' && node.extent <= 16) {
      // Fully unroll
      this.emit(`// unrolled loop ${v} (${node.extent} iterations)`);
      for (let val = node.min; val < node.min + node.extent; val++) {
        this.emit(`{ const ${v} = ${val};`);
        this.indent++;
        this.genStmt(node.body);
        this.indent--;
        this.emit('}');
      }
      return;
    }

    this.emit(`for (let ${v} = ${node.min}; ${v} < ${node.min + node.extent}; ${v}++) {${annotation}`);
    this.indent++;
    this.genStmt(node.body);
    this.indent--;
    this.emit('}');
  }

  private genStore(node: BufferStoreNode): void {
    const idx = this.genFlatIndex(node.buffer.name, node.indices);
    const val = this.genValue(node.value);
    this.emit(`${node.buffer.name}[${idx}] = ${val};`);
  }

  private genFlatIndex(bufferName: string, indices: IndexExpr[]): string {
    const strides = this.bufferStrides.get(bufferName);
    if (!strides || indices.length === 0) return '0';

    if (indices.length === 1) {
      return this.genIndex(indices[0]);
    }

    const terms: string[] = [];
    for (let i = 0; i < indices.length; i++) {
      const idx = this.genIndex(indices[i]);
      const stride = strides[i];
      if (stride === 1) {
        terms.push(idx);
      } else if (stride === 0) {
        // skip
      } else {
        terms.push(`${idx} * ${stride}`);
      }
    }
    return terms.length > 0 ? terms.join(' + ') : '0';
  }

  private genIndex(idx: IndexExpr): string {
    if (idx instanceof VarIndex) return idx.loopVar.name;
    if (idx instanceof ConstIndex) return `${idx.value}`;
    if (idx instanceof BinOpIndex) {
      const l = this.genIndex(idx.left);
      const r = this.genIndex(idx.right);
      // Use Math.floor for '/' to ensure integer division in JS
      if (idx.op === '/') return `Math.floor(${l} / ${r})`;
      return `(${l} ${idx.op} ${r})`;
    }
    return '0';
  }

  private genValue(val: ValueExpr): string {
    if (val.kind === 'const') return `${(val as ConstExpr).value}`;
    if (val.kind === 'varref') return (val as VarRefExpr).loopVar.name;
    if (val.kind === 'load') {
      const v = val as BufferLoadExpr;
      const idx = this.genFlatIndex(v.buffer.name, v.indices);
      return `${v.buffer.name}[${idx}]`;
    }
    if (val.kind === 'binop') {
      const v = val as BinOpExpr;
      const l = this.genValue(v.left);
      const r = this.genValue(v.right);
      return `(${l} ${v.op} ${r})`;
    }
    if (val.kind === 'max') {
      const v = val as MaxExpr;
      return `Math.max(${this.genValue(v.left)}, ${this.genValue(v.right)})`;
    }
    if (val.kind === 'min') {
      const v = val as MinExpr;
      return `Math.min(${this.genValue(v.left)}, ${this.genValue(v.right)})`;
    }
    if (val.kind === 'call') {
      const v = val as CallExprTIR;
      const args = v.args.map(a => this.genValue(a)).join(', ');
      return `${v.funcName}(${args})`;
    }
    if ((val as any).kind === 'scalar_load') {
      return (val as any as ScalarLoadExpr).scalarName;
    }
    return '0';
  }

  private collectScalarNames(stmt: Stmt | any): Set<string> {
    const names = new Set<string>();
    const visit = (s: any): void => {
      if (s instanceof ScalarStoreNode || s?.nodeType === 'scalar_store') {
        names.add(s.scalarName);
      }
      if (s instanceof ScalarDeclNode || s?.nodeType === 'scalar_decl') {
        names.add(s.scalarName);
        visit(s.body);
      }
      if (s instanceof ForNode) visit(s.body);
      if (s instanceof SeqNode) s.stmts.forEach((st: any) => visit(st));
      if (s instanceof AllocNode) visit(s.body);
    };
    visit(stmt);
    return names;
  }

  private emit(line: string): void {
    const pad = '  '.repeat(this.indent);
    this.lines.push(pad + line);
  }
}

// ─── Compile: codegen + eval → executable function ───
export function compile(func: PrimFunc): Function {
  const code = codegenJS(func);
  // Use Function constructor (safer than eval)
  const paramNames = func.params.map(p => p.name);
  const body = code
    .split('\n')
    .slice(1, -1) // remove function header and closing brace
    .map(l => l.replace(/^  /, '')) // remove one level of indent
    .join('\n');

  return new Function(...paramNames, body);
}

// ═══════════════════════════════════════
//  Register Tiling Codegen
//  Generates JS with explicit 4-way accumulator unrolling:
//    - Loads A[k] once per k iteration
//    - Accumulates into acc0..acc3 for 4 output elements simultaneously
//    - Reduces loop overhead 4× and enables instruction-level parallelism
// ═══════════════════════════════════════

export function registerTileJS(func: PrimFunc, tileSize = 4): string {
  // Find the spatial j loop (output features) and reduction k loop
  const loops = func.getLoops();
  const jLoop = loops.find(l => l.loopVar.name === 'j');
  const kLoop = loops.find(l => l.loopVar.name === 'k');

  if (!jLoop || !kLoop || jLoop.forNode.extent % tileSize !== 0) {
    return `// register tiling not applicable to ${func.name}\n` + codegenJS(func);
  }

  const M = 1; // batch size
  const N = jLoop.forNode.extent;
  const K = kLoop.forNode.extent;
  const wBuf = func.params.find(p => p.name === 'W');
  const aBuf = func.params.find(p => p.name === 'A');
  const bBuf = func.params.find(p => p.name === 'B');
  const outBuf = func.params.find(p => p.name === 'Out');

  if (!wBuf || !aBuf || !outBuf) {
    return `// register tiling: missing A/W/Out buffers in ${func.name}\n` + codegenJS(func);
  }

  const hasAct = func.name.includes('relu');
  const accNames = Array.from({ length: tileSize }, (_, t) => `acc${t}`);
  const lines: string[] = [];

  lines.push(`function ${func.name}_reg${tileSize}(${func.params.map(p => p.name).join(', ')}) {`);
  lines.push(`  // Register-tiled: ${tileSize}-way accumulator unrolling`);
  lines.push(`  // Loads A[k] once, reuses for ${tileSize} output elements → ${tileSize}× fewer A reads`);
  lines.push(`  for (let i = 0; i < ${M}; i++) {`);
  lines.push(`    for (let j = 0; j < ${N}; j += ${tileSize}) {`);
  lines.push(`      let ${accNames.join(' = ')} = 0;`);
  lines.push(`      for (let k = 0; k < ${K}; k++) {`);
  lines.push(`        const a = A[i * ${K} + k]; // load once, reuse ${tileSize}x`);
  for (let t = 0; t < tileSize; t++) {
    lines.push(`        acc${t} += a * W[(j + ${t}) * ${K} + k];`);
  }
  lines.push(`      }`);
  if (bBuf) {
    for (let t = 0; t < tileSize; t++) {
      lines.push(`      acc${t} += B[j + ${t}];`);
    }
  }
  for (let t = 0; t < tileSize; t++) {
    const store = hasAct ? `Math.max(acc${t}, 0)` : `acc${t}`;
    lines.push(`      Out[i * ${N} + j + ${t}] = ${store};`);
  }
  lines.push(`    }`);
  lines.push(`  }`);
  lines.push(`}`);

  return lines.join('\n');
}
