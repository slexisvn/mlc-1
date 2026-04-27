// ═══════════════════════════════════════════════════════════════
//  F10: WebAssembly WAT Codegen
//
//  What this does (MLC concept):
//    WebAssembly Text format (WAT) is the human-readable form of
//    WebAssembly (WASM). It compiles to a compact binary .wasm
//    that runs in browsers and Node.js at near-native speed.
//
//    In a real MLC stack, targeting WASM means:
//    1. Generating .wat from a TensorIR PrimFunc
//    2. Assembling .wat → .wasm (via wat2wasm / Binaryen)
//    3. Loading the .wasm module in JS and calling exports
//
//    WASM advantages for ML inference:
//    - Portable: same binary runs on all platforms
//    - Predictable performance: no JIT warm-up variance
//    - WASM SIMD (v128): 4× float32 operations per instruction
//    - sandboxed: safe to run untrusted ML models
//
//    WAT instruction mapping from TensorIR:
//      ForNode → block/loop/br_if pattern
//      BufferStoreNode → f32.store
//      BufferLoadExpr → f32.load
//      BinOpExpr (+) → f32.add
//      BinOpExpr (*) → f32.mul
//      BinOpExpr (-) → f32.sub
//      BinOpExpr (/) → f32.div
//      MaxExpr → f32.max (WASM has a native f32.max instruction)
//      ConstExpr → f32.const
//
//    Memory model in WASM:
//    - All float32 buffers are laid out sequentially in linear memory
//    - Buffer base address = sum of all preceding buffer sizes × 4
//    - Indices use i32 arithmetic: base_bytes + (idx * 4)
//
//  This codegen produces valid WAT for any PrimFunc with:
//    - ForNode loop nests (arbitrary depth)
//    - BufferStore/Load with flat f32 access
//    - Arithmetic expressions (BinOp, Const, Max, Min)
//
//  Note: The generated WAT uses WASM's stack machine model where:
//    - Values are pushed/popped from the operand stack
//    - Locals are declared at the function top level
//    - Blocks use 'block'/'loop'/'br_if' for structured control flow
// ═══════════════════════════════════════════════════════════════

import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, AllocNode,
  BufferLoadExpr, BinOpExpr, ConstExpr, VarRefExpr, MaxExpr, MinExpr,
  CallExprTIR, VarIndex, ConstIndex, BinOpIndex, BufferDecl,
  type Stmt, type ValueExpr, type IndexExpr
} from '../ir/low_level.js';

export interface WATModule {
  /** Full WAT text (can be passed to wat2wasm) */
  text: string;
  /** Buffer layout: name → byte offset in linear memory */
  bufferOffsets: Map<string, number>;
  /** Total memory size in bytes required */
  totalBytes: number;
  /** Names of exported functions */
  exports: string[];
}

// ─── Main entry point ─────────────────────────────────────────

/**
 * Generate a WAT module from a list of PrimFuncs.
 *
 * Each PrimFunc becomes one exported WASM function.
 * All buffer parameters share one linear memory (Memory 0).
 * The caller must import the memory or use the exported one.
 *
 * Usage:
 *   const wat = codegenWAT([primFunc]);
 *   console.log(wat.text);   // paste into wat2wasm online
 */
export function codegenWAT(funcs: PrimFunc[]): WATModule {
  const exports: string[] = funcs.map(f => f.name);

  // Build global buffer offset map across all functions
  // Each func uses its own set of params, but we lay them out
  // in one shared address space (simplification — in practice,
  // runtime passes pointers, but for demo we use fixed offsets).
  const bufferOffsets = new Map<string, number>();
  let offset = 0;
  for (const func of funcs) {
    for (const param of func.params) {
      bufferOffsets.set(param.name, offset);
      offset += param.shape.reduce((a, b) => a * b, 1) * 4;
    }
  }
  const totalBytes = Math.max(offset, 65536); // at least 1 page

  const lines: string[] = [];
  lines.push('(module');
  // Memory declaration: enough pages for all buffers
  const pages = Math.ceil(totalBytes / 65536);
  lines.push(`  ;; Linear memory: ${pages} page(s) = ${pages * 64}KB`);
  lines.push(`  (memory (export "memory") ${pages})`);
  lines.push('');

  for (const func of funcs) {
    const gen = new WATFuncGenerator(func, bufferOffsets);
    lines.push(gen.generate());
    lines.push('');
  }

  lines.push(')');

  return {
    text: lines.join('\n'),
    bufferOffsets,
    totalBytes,
    exports,
  };
}

// ─── Per-function WAT generator ───────────────────────────────

class WATFuncGenerator {
  private func: PrimFunc;
  private bufferOffsets: Map<string, number>;
  private lines: string[] = [];
  private indent = 2;
  /** i32 locals: loop variable names */
  private locals = new Set<string>();
  /** f32 locals: local scalar buffer names (alloc'd size=1 buffers) */
  private localScalarBuffers = new Set<string>();
  private labelCounter = 0;

  constructor(func: PrimFunc, bufferOffsets: Map<string, number>) {
    this.func = func;
    this.bufferOffsets = bufferOffsets;
  }

  generate(): string {
    this.lines = [];
    this.locals = new Set();
    this.localScalarBuffers = new Set();

    // Collect local scalar buffers from func.allocations
    for (const alloc of this.func.allocations) {
      const totalSize = alloc.shape.reduce((a, b) => a * b, 1);
      if (totalSize === 1 && alloc.scope === 'local') {
        this.localScalarBuffers.add(alloc.name);
      }
    }

    // First pass: collect all loop variable names for locals declaration
    this.collectLocals(this.func.body);

    // Function signature
    // Parameters: all buffer params passed as i32 base pointers
    const paramDefs = this.func.params.map(p => `(param $${p.name} i32)`).join(' ');
    this.emit(`  ;; ─── ${this.func.name} ───`);
    this.emit(`  ;; Inputs: ${this.func.params.map(p => `${p.name}[${p.shape.join(',')}]`).join(', ')}`);
    this.emit(`  (func $${this.func.name} (export "${this.func.name}") ${paramDefs}`);
    this.indent = 4;

    // Declare all loop variable locals as i32
    for (const loc of this.locals) {
      this.emit(`(local $${loc} i32)`);
    }
    // Declare local scalar buffer slots as f32
    for (const name of this.localScalarBuffers) {
      this.emit(`(local $${name} f32)`);
    }
    if (this.locals.size > 0 || this.localScalarBuffers.size > 0) this.emit('');

    // Generate body
    this.genStmt(this.func.body);

    this.indent = 2;
    this.emit(`  )`);

    return this.lines.join('\n');
  }

  private emit(line: string) {
    this.lines.push(' '.repeat(this.indent) + line);
  }

  private newLabel(): string {
    return `$L${this.labelCounter++}`;
  }

  private collectLocals(stmt: Stmt): void {
    if (stmt instanceof ForNode) {
      this.locals.add(stmt.loopVar.name);
      this.collectLocals(stmt.body);
    } else if (stmt instanceof SeqNode) {
      for (const s of stmt.stmts) this.collectLocals(s);
    } else if (stmt instanceof AllocNode) {
      this.collectLocals(stmt.body);
    }
  }

  // ─── Statements ─────────────────────────────────────────────

  private genStmt(stmt: Stmt): void {
    if (stmt instanceof ForNode) {
      this.genFor(stmt);
    } else if (stmt instanceof SeqNode) {
      for (const s of stmt.stmts) this.genStmt(s);
    } else if (stmt instanceof BufferStoreNode) {
      this.genStore(stmt);
    } else if (stmt instanceof AllocNode) {
      // Allocations are handled via linear memory — just generate body
      // In a real impl, we'd track alloc offsets; here we treat them as param-like
      this.genStmt(stmt.body);
    }
  }

  /**
   * WAT loop pattern (structured control flow):
   *
   *   ;; for v = 0; v < extent; v++
   *   i32.const 0
   *   local.set $v
   *   block $outer_L  ;; break target
   *     loop $inner_L  ;; continue target
   *       ;; check: if v >= extent, break
   *       local.get $v
   *       i32.const extent
   *       i32.ge_s
   *       br_if $outer_L
   *       ;; body
   *       ...
   *       ;; increment v
   *       local.get $v
   *       i32.const 1
   *       i32.add
   *       local.set $v
   *       ;; loop back
   *       br $inner_L
   *     end
   *   end
   */
  private genFor(node: ForNode): void {
    const v = node.loopVar.name;
    const breakLabel = this.newLabel();
    const contLabel = this.newLabel();

    this.emit(`;; for ${v} in [${node.min}, ${node.min + node.extent})`);
    // Init v = min
    this.emit(`i32.const ${node.min}`);
    this.emit(`local.set $${v}`);
    // Outer block (break target)
    this.emit(`block ${breakLabel}`);
    this.indent += 2;
    // Inner loop (continue target)
    this.emit(`loop ${contLabel}`);
    this.indent += 2;
    // Condition: v >= extent → break
    this.emit(`local.get $${v}`);
    this.emit(`i32.const ${node.min + node.extent}`);
    this.emit(`i32.ge_s`);
    this.emit(`br_if ${breakLabel}`);
    // Body
    this.genStmt(node.body);
    // Increment: v = v + 1
    this.emit(`local.get $${v}`);
    this.emit(`i32.const 1`);
    this.emit(`i32.add`);
    this.emit(`local.set $${v}`);
    // Loop back
    this.emit(`br ${contLabel}`);
    this.indent -= 2;
    this.emit(`end ;; loop ${contLabel}`);
    this.indent -= 2;
    this.emit(`end ;; block ${breakLabel}`);
  }

  /**
   * Generate f32.store for BufferStoreNode.
   * For local scalar buffers (size=1): use local.set $name.
   * For global/param buffers: WASM f32.store at byte_address.
   */
  private genStore(node: BufferStoreNode): void {
    if (this.localScalarBuffers.has(node.buffer.name)) {
      // Local scalar: value → local.set $name
      this.emit(`;; scalar $${node.buffer.name} = ...`);
      this.genValue(node.value);
      this.emit(`local.set $${node.buffer.name}`);
      return;
    }
    this.emit(`;; store ${node.buffer.name}[...]`);
    // Address: param_base + flat_index * 4
    this.genByteAddress(node.buffer.name, node.indices);
    // Value
    this.genValue(node.value);
    this.emit(`f32.store`);
  }

  /**
   * Generate byte address computation:
   *   param_base + flat_index * 4
   * Result is an i32 on the stack.
   */
  private genByteAddress(bufName: string, indices: IndexExpr[]): void {
    // Start with buffer base pointer (parameter)
    this.emit(`local.get $${bufName}`);
    // Compute flat index as i32
    this.genFlatIndex(bufName, indices);
    // Multiply by 4 (sizeof float32)
    this.emit(`i32.const 4`);
    this.emit(`i32.mul`);
    // Add to base
    this.emit(`i32.add`);
  }

  private genFlatIndex(bufName: string, indices: IndexExpr[]): void {
    if (indices.length === 0) {
      this.emit('i32.const 0');
      return;
    }

    // Get buffer shape for stride computation
    const param = this.func.params.find(p => p.name === bufName);
    const shape = param?.shape ?? [1];
    const strides = computeStrides(shape);

    if (indices.length === 1) {
      this.genIndex(indices[0]);
      return;
    }

    // Flat = sum(idx[i] * stride[i])
    // Always emit explicit stride multiplication even when stride=1,
    // so the generated WAT is unambiguous for any buffer shape.
    let first = true;
    for (let i = 0; i < indices.length; i++) {
      const stride = strides[i];
      if (stride === 0) continue;
      this.genIndex(indices[i]);
      // Always emit stride multiplication — makes addressing explicit and
      // safe against future shape changes (stride=1 case is now documented).
      this.emit(`i32.const ${stride}`);
      this.emit(`i32.mul`);
      if (!first) {
        this.emit(`i32.add`);
      }
      first = false;
    }
    if (first) this.emit('i32.const 0');
  }

  private genIndex(idx: IndexExpr): void {
    if (idx instanceof VarIndex) {
      this.emit(`local.get $${idx.loopVar.name}`);
    } else if (idx instanceof ConstIndex) {
      this.emit(`i32.const ${idx.value}`);
    } else if (idx instanceof BinOpIndex) {
      this.genIndex(idx.left);
      this.genIndex(idx.right);
      switch (idx.op) {
        case '+': this.emit('i32.add'); break;
        case '-': this.emit('i32.sub'); break;
        case '*': this.emit('i32.mul'); break;
        case '/': this.emit('i32.div_s'); break;
      }
    }
  }

  /**
   * Generate a float32 value expression.
   * Result is an f32 on the WASM operand stack.
   */
  private genValue(val: ValueExpr): void {
    if (val instanceof ConstExpr) {
      // WAT requires hex float or explicit decimal for f32.const
      this.emit(`f32.const ${val.value}`);
    } else if (val instanceof VarRefExpr) {
      // Loop variable used as value — convert i32 to f32
      this.emit(`local.get $${val.loopVar.name}`);
      this.emit(`f32.convert_i32_s`);
    } else if (val instanceof BufferLoadExpr) {
      if (this.localScalarBuffers.has(val.buffer.name)) {
        // Local scalar: just get the f32 local
        this.emit(`;; scalar load $${val.buffer.name}`);
        this.emit(`local.get $${val.buffer.name}`);
      } else {
        this.emit(`;; load ${val.buffer.name}[...]`);
        this.genByteAddress(val.buffer.name, val.indices);
        this.emit(`f32.load`);
      }
    } else if (val instanceof BinOpExpr) {
      this.genValue(val.left);
      this.genValue(val.right);
      switch (val.op) {
        case '+': this.emit('f32.add'); break;
        case '-': this.emit('f32.sub'); break;
        case '*': this.emit('f32.mul'); break;
        case '/': this.emit('f32.div'); break;
      }
    } else if (val instanceof MaxExpr) {
      // WASM has native f32.max (returns NaN-propagating max)
      this.genValue(val.left);
      this.genValue(val.right);
      this.emit('f32.max');
    } else if (val instanceof MinExpr) {
      this.genValue(val.left);
      this.genValue(val.right);
      this.emit('f32.min');
    } else {
      // Fallback: emit a 0 constant with a comment
      this.emit(`f32.const 0.0 ;; unsupported expr type`);
    }
  }
}

// ─── Helper ───────────────────────────────────────────────────

function computeStrides(shape: number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}
