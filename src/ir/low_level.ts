// ═══════════════════════════════════════════════════════════════
//  Low-Level IR — TensorIR
//  Loop-level representation of tensor computation.
//  Each PrimFunc is a series of nested loops + buffer operations.
// ═══════════════════════════════════════════════════════════════

// ─── Loop Variable ───

export class LoopVar {
  constructor(
    public name: string,
    public kind: 'spatial' | 'reduction' = 'spatial'
  ) {}
  toString(): string { return this.name; }
}

// ─── Loop Annotation ───

export type LoopAnnotation = 'none' | 'parallel' | 'vectorize' | 'unroll';

// ─── Statements ───

export type Stmt = ForNode | SeqNode | BufferStoreNode | AllocNode;

export class ForNode {
  constructor(
    public loopVar: LoopVar,
    public min: number,
    public extent: number,
    public body: Stmt,
    public annotation: LoopAnnotation = 'none'
  ) {}
}

export class SeqNode {
  constructor(public stmts: Stmt[]) {}
}

export class BufferStoreNode {
  constructor(
    public buffer: BufferDecl,
    public indices: IndexExpr[],
    public value: ValueExpr
  ) {}
}

export class AllocNode {
  constructor(
    public buffer: BufferDecl,
    public body: Stmt
  ) {}
}

// ─── Value Expressions ───

export type ValueExpr =
  | BufferLoadExpr
  | BinOpExpr
  | ConstExpr
  | VarRefExpr
  | MaxExpr
  | MinExpr
  | CallExprTIR;

export class BufferLoadExpr {
  readonly kind = 'load' as const;
  constructor(
    public buffer: BufferDecl,
    public indices: IndexExpr[]
  ) {}
}

export class BinOpExpr {
  readonly kind = 'binop' as const;
  constructor(
    public op: '+' | '-' | '*' | '/',
    public left: ValueExpr,
    public right: ValueExpr
  ) {}
}

export class ConstExpr {
  readonly kind = 'const' as const;
  constructor(public value: number) {}
}

export class VarRefExpr {
  readonly kind = 'varref' as const;
  constructor(public loopVar: LoopVar) {}
}

export class MaxExpr {
  readonly kind = 'max' as const;
  constructor(
    public left: ValueExpr,
    public right: ValueExpr
  ) {}
}

export class MinExpr {
  readonly kind = 'min' as const;
  constructor(
    public left: ValueExpr,
    public right: ValueExpr
  ) {}
}

export class CallExprTIR {
  readonly kind = 'call' as const;
  constructor(
    public funcName: string,        // e.g. 'Math.exp', 'Math.tanh', 'Math.abs'
    public args: ValueExpr[]
  ) {}
}

// ─── Index Expressions ───

export type IndexExpr = VarIndex | ConstIndex | BinOpIndex;

export class VarIndex {
  readonly kind = 'var' as const;
  constructor(public loopVar: LoopVar) {}
}

export class ConstIndex {
  readonly kind = 'const' as const;
  constructor(public value: number) {}
}

export class BinOpIndex {
  readonly kind = 'binop' as const;
  constructor(
    public op: '+' | '*' | '-' | '/' | '%',
    public left: IndexExpr,
    public right: IndexExpr
  ) {}
}

// ─── Buffer Declaration ───

export class BufferDecl {
  constructor(
    public name: string,
    public shape: number[],
    public scope: 'global' | 'local' = 'global'
  ) {}

  get flatSize(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }
}

// ─── PrimFunc — A primitive function (one kernel) ───

export class PrimFunc {
  constructor(
    public name: string,
    public params: BufferDecl[],       // input + output buffers (global scope)
    public body: Stmt,
    public allocations: BufferDecl[] = []  // local allocations
  ) {}

  // Collect all loops
  getLoops(): { loopVar: LoopVar; forNode: ForNode }[] {
    const loops: { loopVar: LoopVar; forNode: ForNode }[] = [];
    function visit(stmt: Stmt): void {
      if (stmt instanceof ForNode) {
        loops.push({ loopVar: stmt.loopVar, forNode: stmt });
        visit(stmt.body);
      } else if (stmt instanceof SeqNode) {
        for (const s of stmt.stmts) visit(s);
      } else if (stmt instanceof AllocNode) {
        visit(stmt.body);
      }
    }
    visit(this.body);
    return loops;
  }

  // Deep clone
  clone(): PrimFunc {
    return new PrimFunc(
      this.name,
      this.params.map(p => new BufferDecl(p.name, [...p.shape], p.scope)),
      cloneStmt(this.body),
      this.allocations.map(a => new BufferDecl(a.name, [...a.shape], a.scope))
    );
  }
}

// ─── Clone utilities ───

export function cloneStmt(stmt: Stmt): Stmt {
  if (stmt instanceof ForNode) {
    return new ForNode(
      new LoopVar(stmt.loopVar.name, stmt.loopVar.kind),
      stmt.min,
      stmt.extent,
      cloneStmt(stmt.body),
      stmt.annotation
    );
  }
  if (stmt instanceof SeqNode) {
    return new SeqNode(stmt.stmts.map(s => cloneStmt(s)));
  }
  if (stmt instanceof BufferStoreNode) {
    return new BufferStoreNode(stmt.buffer, stmt.indices, stmt.value);
  }
  if (stmt instanceof AllocNode) {
    return new AllocNode(stmt.buffer, cloneStmt(stmt.body));
  }
  return stmt;
}
