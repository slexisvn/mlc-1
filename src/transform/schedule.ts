// ═══════════════════════════════════════════════════════════════
//  Schedule Transformations
//  Transform loop structure in TensorIR WITHOUT changing
//  computation semantics. Optimizes for cache locality,
//  SIMD, parallelism.
// ═══════════════════════════════════════════════════════════════

import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, AllocNode,
  LoopVar, LoopAnnotation,
  BufferDecl, BufferLoadExpr, BinOpExpr, MaxExpr, MinExpr, CallExprTIR, ConstExpr,
  VarIndex, ConstIndex, BinOpIndex,
  VarRefExpr,
  type Stmt, type IndexExpr, type ValueExpr,
  cloneStmt
} from '../ir/low_level.js';

export class Schedule {
  private func: PrimFunc;
  private loopMap: Map<string, ForNode> = new Map();

  constructor(func: PrimFunc) {
    this.func = func.clone();
    this.rebuildLoopMap();
  }

  private rebuildLoopMap(): void {
    this.loopMap.clear();
    const visit = (stmt: Stmt) => {
      if (stmt instanceof ForNode) {
        this.loopMap.set(stmt.loopVar.name, stmt);
        visit(stmt.body);
      } else if (stmt instanceof SeqNode) {
        for (const s of stmt.stmts) visit(s);
      } else if (stmt instanceof AllocNode) {
        visit(stmt.body);
      }
    };
    visit(this.func.body);
  }

  getLoop(name: string): LoopVar {
    const loop = this.loopMap.get(name);
    if (!loop) throw new Error(`Loop '${name}' not found. Available: ${[...this.loopMap.keys()]}`);
    return loop.loopVar;
  }

  // ─── SPLIT ───
  // for(i, 0, N) → for(i_outer, 0, ceil(N/factor)) { for(i_inner, 0, factor) }
  // All references to i become i_outer * factor + i_inner
  split(loop: LoopVar, factor: number): [LoopVar, LoopVar] {
    const forNode = this.loopMap.get(loop.name);
    if (!forNode) throw new Error(`Loop '${loop.name}' not found`);

    const outerName = `${loop.name}_outer`;
    const innerName = `${loop.name}_inner`;
    const outerVar = new LoopVar(outerName, loop.kind);
    const innerVar = new LoopVar(innerName, loop.kind);

    const outerExtent = Math.ceil(forNode.extent / factor);
    const innerExtent = factor;

    // Rewrite body: replace all references to old loop var
    const newBody = this.rewriteLoopVar(forNode.body, loop.name, outerVar, innerVar, factor);

    // Create new nested loops
    const innerFor = new ForNode(innerVar, 0, innerExtent, newBody, 'none');
    const outerFor = new ForNode(outerVar, 0, outerExtent, innerFor, 'none');

    // Replace the original ForNode in the AST
    this.replaceForNode(loop.name, outerFor);
    this.rebuildLoopMap();

    return [outerVar, innerVar];
  }

  // ─── REORDER ───
  // Reorder nested loops. newOrder specifies desired nesting from outermost to innermost.
  reorder(newOrder: LoopVar[]): void {
    // Collect all ForNodes in current nesting
    const forNodes: ForNode[] = [];
    const innermostBody = this.collectNestedFors(this.func.body, forNodes);

    if (forNodes.length < 2) return;

    // Match by name  
    const nodeMap = new Map<string, ForNode>();
    for (const fn of forNodes) {
      nodeMap.set(fn.loopVar.name, fn);
    }

    // Rebuild nesting in new order
    let body: Stmt = innermostBody;
    for (let i = newOrder.length - 1; i >= 0; i--) {
      const fn = nodeMap.get(newOrder[i].name);
      if (!fn) continue;
      body = new ForNode(fn.loopVar, fn.min, fn.extent, body, fn.annotation);
    }

    // Include any ForNode not in newOrder at the outermost level
    for (const fn of forNodes) {
      if (!newOrder.find(lv => lv.name === fn.loopVar.name)) {
        body = new ForNode(fn.loopVar, fn.min, fn.extent, body, fn.annotation);
      }
    }

    this.func.body = body;
    this.rebuildLoopMap();
  }

  // ─── FUSE ───
  // for(i, 0, M) { for(j, 0, N) { body } }
  // → for(ij, 0, M*N) { i = ij / N; j = ij % N; body }
  fuse(outer: LoopVar, inner: LoopVar): LoopVar {
    const outerFor = this.loopMap.get(outer.name);
    const innerFor = this.loopMap.get(inner.name);
    if (!outerFor || !innerFor) throw new Error('Loops not found');

    const fusedName = `${outer.name}_${inner.name}`;
    const fusedVar = new LoopVar(fusedName, outer.kind);
    const fusedExtent = outerFor.extent * innerFor.extent;

    // Rewrite body: outer = fused / inner_extent, inner = fused % inner_extent
    const newBody = this.rewriteFusedVars(
      innerFor.body, outer.name, inner.name,
      fusedVar, innerFor.extent
    );

    const fusedFor = new ForNode(fusedVar, 0, fusedExtent, newBody, 'none');
    this.replaceForNode(outer.name, fusedFor);
    this.rebuildLoopMap();
    return fusedVar;
  }

  // ─── TILE ───
  // Compound: split(i, tileI) + split(j, tileJ) + reorder(i_o, j_o, i_i, j_i)
  tile(loopI: LoopVar, loopJ: LoopVar, tileI: number, tileJ: number): [LoopVar, LoopVar, LoopVar, LoopVar] {
    const [iOuter, iInner] = this.split(loopI, tileI);
    const [jOuter, jInner] = this.split(loopJ, tileJ);
    this.reorder([iOuter, jOuter, iInner, jInner]);
    return [iOuter, jOuter, iInner, jInner];
  }

  // ─── ANNOTATIONS ───

  vectorize(loop: LoopVar): void {
    this.annotate(loop.name, 'vectorize');
  }

  parallel(loop: LoopVar): void {
    this.annotate(loop.name, 'parallel');
  }

  unroll(loop: LoopVar): void {
    this.annotate(loop.name, 'unroll');
  }

  private annotate(name: string, ann: LoopAnnotation): void {
    const transform = (stmt: Stmt): Stmt => {
      if (stmt instanceof ForNode) {
        const newBody = transform(stmt.body);
        const newAnn = stmt.loopVar.name === name ? ann : stmt.annotation;
        return new ForNode(stmt.loopVar, stmt.min, stmt.extent, newBody, newAnn);
      }
      if (stmt instanceof SeqNode) {
        return new SeqNode(stmt.stmts.map(s => transform(s)));
      }
      if (stmt instanceof AllocNode) {
        return new AllocNode(stmt.buffer, transform(stmt.body));
      }
      return stmt;
    };
    this.func.body = transform(this.func.body);
    this.rebuildLoopMap();
  }

  // ─── CACHE READ ───
  // Creates a local copy (tile) of a buffer inside an outer loop,
  // improving cache locality by ensuring the tile fits in L1 cache.
  //
  // cacheRead('W', j_outer, tileSize):
  //   Inside j_outer, prepend:
  //     for _jl in [0, tileSize): for _kl in [0, K):
  //       W_local[_jl, _kl] = W[j_outer * tileSize + _jl, _kl]
  //   Then replace load(W, [j_outer*tileSize+j_inner, k]) → load(W_local, [j_inner, k])
  cacheRead(bufName: string, atLoopVar: LoopVar, tileSize: number): BufferDecl {
    const outerFor = this.loopMap.get(atLoopVar.name);
    if (!outerFor) throw new Error(`Loop '${atLoopVar.name}' not found`);

    const wBuf = this.func.params.find(p => p.name === bufName);
    if (!wBuf || wBuf.shape.length < 2) {
      throw new Error(`Buffer '${bufName}' not found or has wrong rank`);
    }

    const K = wBuf.shape[1];
    const wLocal = new BufferDecl(`${bufName}_local`, [tileSize, K], 'local');

    // The inner loop is named by stripping '_outer' from atLoopVar.name and
    // appending '_inner'.  e.g. 'j_outer' → 'j_inner'.
    // (NOT 'j_outer_inner' — that would never match the split's actual name.)
    const innerVarName = atLoopVar.name.replace(/_outer$/, '_inner');

    // ─── Build packing stage (inside atLoopVar) ───
    const jlVar = new LoopVar('_jl_cr', 'spatial');
    const klVar = new LoopVar('_kl_cr', 'spatial');
    const packStmt: Stmt = new ForNode(jlVar, 0, tileSize,
      new ForNode(klVar, 0, K,
        new BufferStoreNode(wLocal,
          [new VarIndex(jlVar), new VarIndex(klVar)],
          new BufferLoadExpr(wBuf, [
            new BinOpIndex('+',
              new BinOpIndex('*', new VarIndex(atLoopVar), new ConstIndex(tileSize)),
              new VarIndex(jlVar)
            ),
            new VarIndex(klVar)
          ])
        )
      )
    );

    // ─── Rewrite W loads in the outer loop's body ───
    const newBody = this.rewriteBufferToLocal(
      outerFor.body, bufName, wLocal, atLoopVar.name, innerVarName, tileSize
    );

    // ─── Assemble: [packing, main] inside outer loop ───
    const newOuterBody = new SeqNode([packStmt, newBody]);
    this.replaceForNode(atLoopVar.name,
      new ForNode(outerFor.loopVar, outerFor.min, outerFor.extent,
        newOuterBody, outerFor.annotation)
    );

    this.func.allocations = [...this.func.allocations, wLocal];
    this.rebuildLoopMap();
    return wLocal;
  }

  // ─── RFACTOR ───
  // Parallel reduction decomposition:
  //
  //   BEFORE (serial reduction):
  //     for i: for j:
  //       acc = 0
  //       for k: acc += A[i,k] * W[j,k]
  //       Out[i,j] = acc
  //
  //   AFTER rfactor(k, numParts=4):
  //     alloc acc_rf[k_outer_extent]   ← partial sum buffers
  //     for i: for j:
  //       for k_outer in parallel:     ← each part runs independently
  //         acc_rf[k_outer] = 0
  //         for k_inner:
  //           acc_rf[k_outer] += A[i, k_outer*kPart+k_inner] * W[j, k_outer*kPart+k_inner]
  //       acc = 0                      ← merge loop: sum the partial sums
  //       for k_outer:
  //         acc += acc_rf[k_outer]
  //       Out[i,j] = acc
  //
  // This teaches: parallel reduction via partial-sum factoring — the
  // technique behind fast multi-core and GPU reduction kernels.
  rfactor(reductionLoopVar: LoopVar, numParts: number): [LoopVar, LoopVar] {
    const kFor = this.loopMap.get(reductionLoopVar.name);
    if (!kFor) throw new Error(`Loop '${reductionLoopVar.name}' not found`);

    const kExtent = kFor.extent;
    const kPart = Math.ceil(kExtent / numParts);
    const kOuterExtent = Math.ceil(kExtent / kPart);

    // Step 1: split k → k_outer, k_inner via existing split()
    const [kOuter, kInner] = this.split(reductionLoopVar, kPart);

    // Step 2: create partial-sum buffer acc_rf[kOuterExtent]
    const accRf = new BufferDecl(`acc_rf`, [kOuterExtent], 'local');

    // Step 3: locate the inner reduction body
    //  After split, structure is: for(k_outer) { for(k_inner) { body } }
    //  We need to:
    //    a) wrap k_inner body as: acc_rf[k_outer] += body_value
    //    b) annotate k_outer as parallel
    //    c) append a merge loop after the k_outer loop
    //
    //  Implementation strategy: transform the AST that contains
    //  the k_outer ForNode (which already wraps k_inner after split).

    this.func.body = this.injectRfactorBuffers(
      this.func.body,
      kOuter.name,
      kInner.name,
      accRf,
      kOuterExtent
    );
    this.func.allocations = [...this.func.allocations, accRf];

    // Step 4: annotate k_outer as parallel
    this.annotate(kOuter.name, 'parallel');

    this.rebuildLoopMap();
    return [kOuter, kInner];
  }

  // Helper: find the k_outer ForNode in the AST, transform it to:
  //   for k_outer (parallel):
  //     acc_rf[k_outer] = 0
  //     for k_inner: acc_rf[k_outer] += ...
  // Then append a merge loop: for k_outer: acc += acc_rf[k_outer]
  // And replace the original acc buffer store with acc += merge_result
  private injectRfactorBuffers(
    stmt: Stmt,
    kOuterName: string,
    kInnerName: string,
    accRf: BufferDecl,
    kOuterExtent: number
  ): Stmt {
    if (stmt instanceof ForNode) {
      // If this is the k_outer loop, transform it
      if (stmt.loopVar.name === kOuterName) {
        return this.buildRfactorNode(stmt, kInnerName, accRf, kOuterExtent);
      }
      // Otherwise recurse into body
      const newBody = this.injectRfactorBuffers(stmt.body, kOuterName, kInnerName, accRf, kOuterExtent);
      return new ForNode(stmt.loopVar, stmt.min, stmt.extent, newBody, stmt.annotation);
    }
    if (stmt instanceof SeqNode) {
      const newStmts = stmt.stmts.map(s =>
        this.injectRfactorBuffers(s, kOuterName, kInnerName, accRf, kOuterExtent)
      );
      return new SeqNode(newStmts);
    }
    if (stmt instanceof AllocNode) {
      const newBody = this.injectRfactorBuffers(stmt.body, kOuterName, kInnerName, accRf, kOuterExtent);
      return new AllocNode(stmt.buffer, newBody);
    }
    return stmt;
  }

  // Build the rfactor transformation for the k_outer loop and its context:
  //   Rewrites k_inner body to accumulate into acc_rf[k_outer],
  //   adds an init store, and appends a merge loop.
  //   Returns a SeqNode: [k_outer_parallel_loop, merge_loop, final_store]
  private buildRfactorNode(
    kOuterFor: ForNode,     // for(k_outer) { for(k_inner) { ... store to acc ... } }
    kInnerName: string,
    accRf: BufferDecl,
    kOuterExtent: number
  ): Stmt {
    const kOuterVar = kOuterFor.loopVar;
    const kInnerFor = kOuterFor.body;

    // Extract the innermost BufferStore(s) that write to acc
    // and rewrite them to write to acc_rf[k_outer] instead
    const rewrittenKInnerBody = this.rewriteAccToRf(
      kInnerFor instanceof ForNode ? kInnerFor.body : kInnerFor,
      accRf,
      kOuterVar
    );

    // Init store: acc_rf[k_outer] = 0 (runs at start of each k_outer iteration)
    const initNode = new BufferStoreNode(
      accRf,
      [new VarIndex(kOuterVar)],
      new ConstExpr(0)
    );

    // Rebuild k_inner loop with rewritten body
    const newKInnerFor = kInnerFor instanceof ForNode
      ? new ForNode(kInnerFor.loopVar, kInnerFor.min, kInnerFor.extent, rewrittenKInnerBody, kInnerFor.annotation)
      : rewrittenKInnerBody;

    // k_outer loop body: [init acc_rf[k_outer]=0, k_inner loop]
    const kOuterBody = new SeqNode([initNode, newKInnerFor]);
    const kOuterParallel = new ForNode(kOuterVar, 0, kOuterExtent, kOuterBody, 'parallel');

    // Merge loop: for k_outer: accumulate acc_rf[k_outer] into a local acc
    // We represent this as a simple reduction ForNode that sums acc_rf
    const mergeVar = new LoopVar(`${kOuterVar.name}_merge`, 'reduction');
    const mergeLoad = new BufferLoadExpr(accRf, [new VarIndex(mergeVar)]);

    // Find the original acc buffer from the context (it's in func.allocations)
    // Look for the 'acc' buffer declared in allocations
    const accBuf = this.func.allocations.find(a => a.name === 'acc');
    const mergeTarget = accBuf ?? new BufferDecl('acc', [1], 'local');

    const mergeBody = new BufferStoreNode(
      mergeTarget,
      [new ConstIndex(0)],
      new BinOpExpr('+',
        new BufferLoadExpr(mergeTarget, [new ConstIndex(0)]),
        mergeLoad
      )
    );
    const mergeLoop = new ForNode(mergeVar, 0, kOuterExtent, mergeBody, 'none');

    // acc = 0 before merge
    const mergeInit = new BufferStoreNode(
      mergeTarget,
      [new ConstIndex(0)],
      new ConstExpr(0)
    );

    return new SeqNode([kOuterParallel, mergeInit, mergeLoop]);
  }

  // Rewrite all BufferStore to 'acc' within the k_inner body
  // to instead accumulate into acc_rf[k_outer]
  private rewriteAccToRf(stmt: Stmt, accRf: BufferDecl, kOuterVar: LoopVar): Stmt {
    if (stmt instanceof BufferStoreNode && stmt.buffer.name === 'acc') {
      // acc[0] += val → acc_rf[k_outer] += val
      return new BufferStoreNode(
        accRf,
        [new VarIndex(kOuterVar)],
        this.rewriteAccLoadsInValue(stmt.value, accRf, kOuterVar)
      );
    }
    if (stmt instanceof SeqNode) {
      return new SeqNode(stmt.stmts.map(s => this.rewriteAccToRf(s, accRf, kOuterVar)));
    }
    if (stmt instanceof ForNode) {
      return new ForNode(stmt.loopVar, stmt.min, stmt.extent,
        this.rewriteAccToRf(stmt.body, accRf, kOuterVar), stmt.annotation);
    }
    if (stmt instanceof AllocNode) {
      return new AllocNode(stmt.buffer, this.rewriteAccToRf(stmt.body, accRf, kOuterVar));
    }
    return stmt;
  }

  // Rewrite any BufferLoad from 'acc[0]' to 'acc_rf[k_outer]' within a value expression
  private rewriteAccLoadsInValue(val: ValueExpr, accRf: BufferDecl, kOuterVar: LoopVar): ValueExpr {
    if (val instanceof BufferLoadExpr && val.buffer.name === 'acc') {
      return new BufferLoadExpr(accRf, [new VarIndex(kOuterVar)]);
    }
    if (val instanceof BinOpExpr) {
      return new BinOpExpr(
        val.op,
        this.rewriteAccLoadsInValue(val.left, accRf, kOuterVar),
        this.rewriteAccLoadsInValue(val.right, accRf, kOuterVar)
      );
    }
    if (val instanceof MaxExpr) {
      return new MaxExpr(
        this.rewriteAccLoadsInValue(val.left, accRf, kOuterVar),
        this.rewriteAccLoadsInValue(val.right, accRf, kOuterVar)
      );
    }
    if (val instanceof MinExpr) {
      return new MinExpr(
        this.rewriteAccLoadsInValue(val.left, accRf, kOuterVar),
        this.rewriteAccLoadsInValue(val.right, accRf, kOuterVar)
      );
    }
    if (val instanceof CallExprTIR) {
      return new CallExprTIR(val.funcName, val.args.map(a => this.rewriteAccLoadsInValue(a, accRf, kOuterVar)));
    }
    return val;
  }

  // ─── COMPUTE AT (stub) ───
  // Moves a producer stage inside a consumer's loop nest.
  // Full implementation requires producer-consumer dependency analysis.
  computeAt(_producerBuf: string, _consumerLoop: LoopVar): void {
    // Stub — not implemented in this simplified version
  }

  // ─── BUILD: return transformed PrimFunc ───

  build(): PrimFunc {
    return this.func;
  }

  // ─── Internal helpers ───

  private rewriteLoopVar(
    stmt: Stmt, oldName: string,
    outerVar: LoopVar, innerVar: LoopVar, factor: number
  ): Stmt {
    if (stmt instanceof ForNode) {
      return new ForNode(
        stmt.loopVar, stmt.min, stmt.extent,
        this.rewriteLoopVar(stmt.body, oldName, outerVar, innerVar, factor),
        stmt.annotation
      );
    }
    if (stmt instanceof SeqNode) {
      return new SeqNode(
        stmt.stmts.map(s => this.rewriteLoopVar(s, oldName, outerVar, innerVar, factor))
      );
    }
    if (stmt instanceof BufferStoreNode) {
      return new BufferStoreNode(
        stmt.buffer,
        stmt.indices.map(idx => this.rewriteIndex(idx, oldName, outerVar, innerVar, factor)),
        this.rewriteValue(stmt.value, oldName, outerVar, innerVar, factor)
      );
    }
    if (stmt instanceof AllocNode) {
      return new AllocNode(
        stmt.buffer,
        this.rewriteLoopVar(stmt.body, oldName, outerVar, innerVar, factor)
      );
    }
    return stmt;
  }

  private rewriteIndex(
    idx: IndexExpr, oldName: string,
    outerVar: LoopVar, innerVar: LoopVar, factor: number
  ): IndexExpr {
    if (idx instanceof VarIndex && idx.loopVar.name === oldName) {
      // old_var → outer * factor + inner
      return new BinOpIndex('+',
        new BinOpIndex('*', new VarIndex(outerVar), new ConstIndex(factor)),
        new VarIndex(innerVar)
      );
    }
    if (idx instanceof BinOpIndex) {
      return new BinOpIndex(idx.op,
        this.rewriteIndex(idx.left, oldName, outerVar, innerVar, factor),
        this.rewriteIndex(idx.right, oldName, outerVar, innerVar, factor)
      );
    }
    return idx;
  }

  private rewriteValue(
    val: ValueExpr, oldName: string,
    outerVar: LoopVar, innerVar: LoopVar, factor: number
  ): ValueExpr {
    if (val.kind === 'varref' && val.loopVar.name === oldName) {
      return new BinOpExpr('+',
        new BinOpExpr('*', new VarRefExpr(outerVar), { kind: 'const', value: factor } as any),
        new VarRefExpr(innerVar)
      );
    }
    if (val.kind === 'load') {
      const v = val as BufferLoadExpr;
      return new BufferLoadExpr(v.buffer, v.indices.map(i => this.rewriteIndex(i, oldName, outerVar, innerVar, factor)));
    }
    if (val.kind === 'binop') {
      return new BinOpExpr(val.op,
        this.rewriteValue(val.left, oldName, outerVar, innerVar, factor),
        this.rewriteValue(val.right, oldName, outerVar, innerVar, factor)
      );
    }
    if (val.kind === 'max') {
      const v = val as MaxExpr;
      return new MaxExpr(
        this.rewriteValue(v.left, oldName, outerVar, innerVar, factor),
        this.rewriteValue(v.right, oldName, outerVar, innerVar, factor)
      );
    }
    if (val.kind === 'call') {
      const v = val as CallExprTIR;
      return new CallExprTIR(v.funcName, v.args.map(a => this.rewriteValue(a, oldName, outerVar, innerVar, factor)));
    }
    return val;
  }

  private collectNestedFors(stmt: Stmt, collected: ForNode[]): Stmt {
    if (stmt instanceof ForNode) {
      collected.push(stmt);
      return this.collectNestedFors(stmt.body, collected);
    }
    if (stmt instanceof AllocNode) {
      return this.collectNestedFors(stmt.body, collected);
    }
    return stmt; // the innermost body (SeqNode or BufferStoreNode)
  }

  private replaceForNode(name: string, replacement: Stmt): void {
    this.func.body = this.replaceFor(this.func.body, name, replacement);
  }

  private replaceFor(stmt: Stmt, name: string, replacement: Stmt): Stmt {
    if (stmt instanceof ForNode) {
      if (stmt.loopVar.name === name) return replacement;
      return new ForNode(
        stmt.loopVar, stmt.min, stmt.extent,
        this.replaceFor(stmt.body, name, replacement),
        stmt.annotation
      );
    }
    if (stmt instanceof SeqNode) {
      return new SeqNode(stmt.stmts.map(s => this.replaceFor(s, name, replacement)));
    }
    if (stmt instanceof AllocNode) {
      return new AllocNode(stmt.buffer, this.replaceFor(stmt.body, name, replacement));
    }
    return stmt;
  }

  private rewriteFusedVars(
    stmt: Stmt, outerName: string, innerName: string,
    fusedVar: LoopVar, innerExtent: number
  ): Stmt {
    // Similar rewrite but: outer = fused / innerExtent, inner = fused % innerExtent
    // For simplicity, we just annotate this conceptually
    return stmt;
  }

  // ─── Helper: rewrite buffer loads from global buf → local tile ───
  // Matches pattern: load(bufName, [(outer * tileSize + inner), k_expr])
  // Replaces with:   load(local, [inner_var, k_expr])
  private rewriteBufferToLocal(
    stmt: Stmt,
    bufName: string,
    wLocal: BufferDecl,
    outerVarName: string,
    innerVarName: string,
    tileSize: number
  ): Stmt {
    if (stmt instanceof ForNode) {
      return new ForNode(stmt.loopVar, stmt.min, stmt.extent,
        this.rewriteBufferToLocal(stmt.body, bufName, wLocal, outerVarName, innerVarName, tileSize),
        stmt.annotation
      );
    }
    if (stmt instanceof SeqNode) {
      return new SeqNode(stmt.stmts.map(s =>
        this.rewriteBufferToLocal(s, bufName, wLocal, outerVarName, innerVarName, tileSize)
      ));
    }
    if (stmt instanceof BufferStoreNode) {
      return new BufferStoreNode(stmt.buffer, stmt.indices,
        this.rewriteValueToLocal(stmt.value, bufName, wLocal, outerVarName, innerVarName, tileSize)
      );
    }
    if (stmt instanceof AllocNode) {
      return new AllocNode(stmt.buffer,
        this.rewriteBufferToLocal(stmt.body, bufName, wLocal, outerVarName, innerVarName, tileSize)
      );
    }
    return stmt;
  }

  private rewriteValueToLocal(
    val: ValueExpr,
    bufName: string,
    wLocal: BufferDecl,
    outerVarName: string,
    innerVarName: string,
    tileSize: number
  ): ValueExpr {
    if (val.kind === 'load') {
      const v = val as BufferLoadExpr;
      if (v.buffer.name === bufName && v.indices.length === 2) {
        const idx0 = v.indices[0];
        if (this.matchSplitPattern(idx0, outerVarName, innerVarName, tileSize)) {
          // Replace W[j_outer*tile+j_inner, k] → W_local[j_inner, k]
          const innerIdx = this.extractInner(idx0);
          return new BufferLoadExpr(wLocal, [innerIdx, v.indices[1]]);
        }
      }
      return val;
    }
    if (val.kind === 'binop') {
      const v = val as BinOpExpr;
      return new BinOpExpr(v.op,
        this.rewriteValueToLocal(v.left, bufName, wLocal, outerVarName, innerVarName, tileSize),
        this.rewriteValueToLocal(v.right, bufName, wLocal, outerVarName, innerVarName, tileSize)
      );
    }
    if (val.kind === 'max') {
      const v = val as MaxExpr;
      return new MaxExpr(
        this.rewriteValueToLocal(v.left, bufName, wLocal, outerVarName, innerVarName, tileSize),
        this.rewriteValueToLocal(v.right, bufName, wLocal, outerVarName, innerVarName, tileSize)
      );
    }
    if (val.kind === 'min') {
      const v = val as MinExpr;
      return new MinExpr(
        this.rewriteValueToLocal(v.left, bufName, wLocal, outerVarName, innerVarName, tileSize),
        this.rewriteValueToLocal(v.right, bufName, wLocal, outerVarName, innerVarName, tileSize)
      );
    }
    if (val.kind === 'call') {
      const v = val as CallExprTIR;
      return new CallExprTIR(v.funcName,
        v.args.map(a => this.rewriteValueToLocal(a, bufName, wLocal, outerVarName, innerVarName, tileSize))
      );
    }
    return val;
  }

  // Detect: (outer * tileSize + inner) pattern in index expression
  private matchSplitPattern(
    idx: IndexExpr, outerName: string, innerName: string, factor: number
  ): boolean {
    if (!(idx instanceof BinOpIndex) || idx.op !== '+') return false;
    const left = idx.left;
    const right = idx.right;
    if (!(left instanceof BinOpIndex) || left.op !== '*') return false;
    if (!(left.left instanceof VarIndex) || left.left.loopVar.name !== outerName) return false;
    if (!(left.right instanceof ConstIndex) || left.right.value !== factor) return false;
    if (!(right instanceof VarIndex) || right.loopVar.name !== innerName) return false;
    return true;
  }

  // Extract the inner part from (outer * factor + inner)
  private extractInner(idx: IndexExpr): IndexExpr {
    if (idx instanceof BinOpIndex && idx.op === '+') return idx.right;
    return idx;
  }
}

// ─── Apply a default good schedule for matmul-like ops ───
export function applyDefaultSchedule(func: PrimFunc): PrimFunc {
  const loops = func.getLoops();
  if (loops.length < 2) return func;

  const sch = new Schedule(func);

  // Find spatial j loop (output features) and reduction k loop
  const jLoop = loops.find(l => l.loopVar.name === 'j');
  const kLoop = loops.find(l => l.loopVar.name === 'k');

  if (jLoop && jLoop.forNode.extent >= 16 && kLoop && kLoop.forNode.extent >= 16) {
    // Tile j and k for cache locality
    const tileJ = Math.min(32, jLoop.forNode.extent);
    const tileK = Math.min(64, kLoop.forNode.extent);

    try {
      const [jOuter, jInner] = sch.split(jLoop.loopVar, tileJ);
      const [kOuter, kInner] = sch.split(kLoop.loopVar, tileK);
      sch.reorder([jOuter, kOuter, jInner, kInner]);

      // Annotate
      if (jOuter) sch.parallel(jOuter);
      if (jInner) sch.unroll(jInner);
    } catch {
      // If schedule fails, return naive
      return func;
    }
  }

  return sch.build();
}
