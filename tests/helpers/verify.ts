// ─── Verifier wrappers ─────────────────────────────────────────────

import { IRModule } from '../../src/ir/high_level.js';
import { PrimFunc } from '../../src/ir/low_level.js';
import {
  verifyHighLevelIR,
  verifyLowLevelIR,
} from '../../src/transform/verifier.js';

export function assertHLIRValid(mod: IRModule, label = ''): void {
  const result = verifyHighLevelIR(mod);
  if (!result.ok) {
    const tag = label ? `[${label}] ` : '';
    throw new Error(`${tag}verifyHighLevelIR failed:\n  ${result.errors.join('\n  ')}`);
  }
}

export function assertTIRValid(funcs: PrimFunc[], label = ''): void {
  const result = verifyLowLevelIR(funcs);
  if (!result.ok) {
    const tag = label ? `[${label}] ` : '';
    throw new Error(`${tag}verifyLowLevelIR failed:\n  ${result.errors.join('\n  ')}`);
  }
}
