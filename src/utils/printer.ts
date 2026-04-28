import { Expr, IRModule } from "../ir/high_level.js";
import {
  PrimFunc,
  ForNode,
  SeqNode,
  BufferStoreNode,
  AllocNode,
  BufferLoadExpr,
  BinOpExpr,
  ConstExpr as TIRConst,
  VarRefExpr,
  MaxExpr,
  MinExpr,
  CallExprTIR,
  VarIndex,
  ConstIndex,
  BinOpIndex,
  type Stmt,
  type ValueExpr,
  type IndexExpr,
} from "../ir/low_level.js";

export function printHighLevelIR(module: IRModule): string {
  const lines: string[] = [];

  for (const [name, func] of module.functions) {
    const params = func.params.map((p) => `%${p.name}: ${p.type}`).join(", ");
    lines.push(`@${name}(${params}) → ${func.retType} {`);

    const { text } = printExpr(func.body, 1);
    lines.push(text);
    lines.push("}");
    lines.push("");
  }

  return lines.join("\n");
}

function printExpr(
  expr: Expr,
  indent: number,
): { text: string; varCount: number } {
  const pad = "  ".repeat(indent);
  let counter = 0;

  function visit(e: Expr): string {
    switch (e.kind) {
      case "var":
        return `%${e.name}`;

      case "constant":
        return e.name ? `meta[${e.name}]` : `const(shape=[${e.data.shape}])`;

      case "let":
        return `let %${e.varName.name} = ${visit(e.value)};\n${pad}${visit(e.body)}`;

      case "call": {
        const args = e.args.map((a) => visit(a)).join(", ");
        const id = counter++;
        const shape = e.attrs.outputShape
          ? `  // [${e.attrs.outputShape}]`
          : "";
        const fusedInfo = e.attrs.fusedOps
          ? `  /* fused: ${(e.attrs.fusedOps as string[]).join(" → ")} */`
          : "";
        return `%${id} = ${e.op.name}(${args})${shape}${fusedInfo}`;
      }
    }
  }

  const assignments: string[] = [];

  function linearize(e: Expr): string {
    if (e.kind !== "call") return visit(e);

    const argNames: string[] = [];
    for (const arg of e.args) {
      if (arg.kind === "call") {
        const name = linearize(arg);
        argNames.push(name);
      } else {
        argNames.push(visit(arg));
      }
    }

    const id = counter++;
    const shape = e.attrs.outputShape ? `  // [${e.attrs.outputShape}]` : "";
    const fusedInfo = e.attrs.fusedOps
      ? `  /* fused: ${(e.attrs.fusedOps as string[]).join(" → ")} */`
      : "";
    const line = `${pad}%${id} = ${e.op.name}(${argNames.join(", ")})${shape}${fusedInfo}`;
    assignments.push(line);
    return `%${id}`;
  }

  linearize(expr);
  return { text: assignments.join("\n"), varCount: counter };
}

export function printTIR(func: PrimFunc): string {
  const lines: string[] = [];

  // Header
  const params = func.params
    .map(
      (p) =>
        `${p.name}: Buffer[${p.shape.join(",")}]${p.scope !== "global" ? ` (${p.scope})` : ""}`,
    )
    .join(", ");
  lines.push(`@${func.name}(${params}) {`);

  // Allocations
  for (const alloc of func.allocations) {
    lines.push(
      `  alloc ${alloc.name}: float32[${alloc.shape.join(",")}]  scope=${alloc.scope}`,
    );
  }

  // Body
  printStmt(func.body, 1, lines);

  lines.push("}");
  return lines.join("\n");
}

function printStmt(stmt: Stmt, indent: number, lines: string[]): void {
  const pad = "  ".repeat(indent);

  if (stmt instanceof ForNode) {
    const ann = stmt.annotation !== "none" ? `  // ${stmt.annotation}` : "";
    const kind = stmt.loopVar.kind === "reduction" ? "  // reduction" : "";
    lines.push(
      `${pad}for ${stmt.loopVar.name} in range(${stmt.min}, ${stmt.min + stmt.extent}):${ann}${kind}`,
    );
    printStmt(stmt.body, indent + 1, lines);
  } else if (stmt instanceof SeqNode) {
    for (const s of stmt.stmts) {
      printStmt(s, indent, lines);
    }
  } else if (stmt instanceof BufferStoreNode) {
    const idx = printIndices(stmt.indices);
    const val = printValue(stmt.value);
    lines.push(`${pad}${stmt.buffer.name}[${idx}] = ${val}`);
  } else if (stmt instanceof AllocNode) {
    lines.push(
      `${pad}alloc ${stmt.buffer.name}[${stmt.buffer.shape.join(",")}] (${stmt.buffer.scope})`,
    );
    printStmt(stmt.body, indent, lines);
  }
}

function printIndices(indices: IndexExpr[]): string {
  return indices.map(printIndex).join(", ");
}

function printIndex(idx: IndexExpr): string {
  if (idx instanceof VarIndex) return idx.loopVar.name;
  if (idx instanceof ConstIndex) return `${idx.value}`;
  if (idx instanceof BinOpIndex)
    return `(${printIndex(idx.left)} ${idx.op} ${printIndex(idx.right)})`;
  return "?";
}

function printValue(val: ValueExpr): string {
  if (val.kind === "const") return `${(val as TIRConst).value}`;
  if (val.kind === "varref") return (val as VarRefExpr).loopVar.name;
  if (val.kind === "load") {
    const v = val as BufferLoadExpr;
    return `${v.buffer.name}[${printIndices(v.indices)}]`;
  }
  if (val.kind === "binop") {
    const v = val as BinOpExpr;
    return `(${printValue(v.left)} ${v.op} ${printValue(v.right)})`;
  }
  if (val.kind === "max") {
    const v = val as MaxExpr;
    return `max(${printValue(v.left)}, ${printValue(v.right)})`;
  }
  if (val.kind === "min") {
    const v = val as MinExpr;
    return `min(${printValue(v.left)}, ${printValue(v.right)})`;
  }
  if (val.kind === "call") {
    const v = val as CallExprTIR;
    const args = v.args.map((a) => printValue(a)).join(", ");
    return `${v.funcName}(${args})`;
  }
  return "?";
}

export function printPhaseBanner(phase: number, title: string): void {
  const width = 60;
  console.log("");
  console.log("═".repeat(width));
  console.log(`  Phase ${phase}: ${title}`);
  console.log("═".repeat(width));
}

export function printSubSection(title: string): void {
  console.log(`\n── ${title} ──`);
}
