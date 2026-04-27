// ═══════════════════════════════════════════════════════════════
//  Op Profiler & Roofline Model Analysis
//
//  Benchmarks each compiled kernel and plots results against a
//  "Roofline Model" — the theoretical performance ceiling
//  determined by:
//    1. Peak Compute (GFLOPS the CPU can deliver)
//    2. Peak Memory Bandwidth (GB/s the system can sustain)
//
//  Kernels below the roofline have room for optimization.
//  Kernels on the memory roof need better data reuse (tiling).
//  Kernels on the compute roof need fewer FLOPs (algorithmic).
//
//  The roofline "ridge point" = peak_compute / peak_bandwidth
//  is the arithmetic intensity where the bottleneck transitions
//  from memory to compute.
// ═══════════════════════════════════════════════════════════════

import { PrimFunc } from '../ir/low_level.js';
import { compile } from '../codegen/js_codegen.js';
import { analyzeMemory, type MemoryPlan } from './memory_planner.js';

// ─── Hardware Profile (estimated for V8/Node.js on x86-64) ───

export interface HardwareProfile {
  peakGFLOPS: number;      // theoretical peak compute
  peakBandwidthGBs: number; // peak memory bandwidth
  ridgePoint: number;       // FLOP/byte where compute meets memory
  name: string;
}

export const DEFAULT_PROFILE: HardwareProfile = {
  peakGFLOPS: 10,           // ~10 GFLOPS for single-threaded V8
  peakBandwidthGBs: 20,     // ~20 GB/s DDR4 bandwidth
  ridgePoint: 0.5,          // 10 / 20 = 0.5
  name: 'V8/Node.js (estimated)',
};

// ═══════════════════════════════════════
//  Benchmark Infrastructure
// ═══════════════════════════════════════

function benchmarkKernel(func: PrimFunc, iterations: number): number {
  // Create test buffers
  const buffers = func.params.map(p => {
    const size = p.shape.reduce((a, b) => a * b, 1);
    const buf = new Float32Array(size);
    for (let i = 0; i < size; i++) buf[i] = Math.random() * 0.1;
    return buf;
  });

  // Compile
  const fn = compile(func);

  // Warmup — enough iterations so V8 JIT fully compiles the hot loop
  // before we take measurements (10 is insufficient for small kernels).
  for (let i = 0; i < 100; i++) fn(...buffers);

  // Measure
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn(...buffers);
    const end = performance.now();
    times.push(end - start);
  }

  // Return median in milliseconds
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

// ═══════════════════════════════════════
//  Per-Kernel Roofline Analysis
// ═══════════════════════════════════════

export interface KernelProfile {
  name: string;
  medianTimeMs: number;
  achievedGFLOPS: number;
  arithmeticIntensity: number;
  achievedBandwidthGBs: number;
  percentOfPeak: number;
  bottleneck: 'compute' | 'memory';
  memoryPlan: MemoryPlan;
  rooflineBar: string;     // ASCII visualization
}

function createRooflineBar(percentOfPeak: number): string {
  const width = 20;
  const filled = Math.round(percentOfPeak / 100 * width);
  const bar = '█'.repeat(Math.min(filled, width)) + '░'.repeat(Math.max(0, width - filled));
  return `[${bar}] ${percentOfPeak.toFixed(1)}%`;
}

export function profileKernel(
  func: PrimFunc,
  iterations: number = 100,
  hw: HardwareProfile = DEFAULT_PROFILE
): KernelProfile {
  // Static analysis
  const memPlan = analyzeMemory(func);

  // Dynamic benchmark
  const medianMs = benchmarkKernel(func, iterations);

  // Compute achieved performance
  const medianSeconds = medianMs / 1000;
  const achievedGFLOPS = medianSeconds > 0
    ? memPlan.flopCount / medianSeconds / 1e9
    : 0;

  const achievedBW = medianSeconds > 0
    ? memPlan.effectiveTrafficBytes / medianSeconds / 1e9
    : 0;

  // Use cache-aware effectiveAI for roofline classification and efficiency %
  // (it reflects actual data reuse from tiling), but store the intrinsic
  // arithmeticIntensity in the profile so the table AI column is stable
  // across classifiers regardless of which tile the tuner happened to pick.
  const effectiveAI = memPlan.effectiveAI;
  const intrinsicAI = memPlan.arithmeticIntensity;

  // Theoretical max for this kernel's arithmetic intensity
  const memoryRoof = hw.peakBandwidthGBs * effectiveAI;
  const computeRoof = hw.peakGFLOPS;
  const theoreticalPeak = Math.min(memoryRoof, computeRoof);
  const percentOfPeak = theoreticalPeak > 0
    ? (achievedGFLOPS / theoreticalPeak) * 100
    : 0;

  const bottleneck: 'compute' | 'memory' =
    effectiveAI >= hw.ridgePoint ? 'compute' : 'memory';

  return {
    name: func.name,
    medianTimeMs: medianMs,
    achievedGFLOPS,
    arithmeticIntensity: intrinsicAI,   // intrinsic (naive) — stable across schedules
    achievedBandwidthGBs: achievedBW,
    percentOfPeak: Math.min(percentOfPeak, 100),
    bottleneck,
    memoryPlan: memPlan,
    rooflineBar: createRooflineBar(Math.min(percentOfPeak, 100)),
  };
}

// ═══════════════════════════════════════
//  Full Pipeline Profiler
// ═══════════════════════════════════════

export function profilePipeline(
  funcs: PrimFunc[],
  iterations: number = 100,
  hw: HardwareProfile = DEFAULT_PROFILE
): KernelProfile[] {
  return funcs.map(f => profileKernel(f, iterations, hw));
}

// ═══════════════════════════════════════
//  Roofline Visualization (Text-based)
// ═══════════════════════════════════════

export function printRoofline(
  profiles: KernelProfile[],
  hw: HardwareProfile = DEFAULT_PROFILE
): string {
  const lines: string[] = [];

  lines.push('╔══════════════════════════════════════════════════════════════╗');
  lines.push('║              ROOFLINE MODEL ANALYSIS                       ║');
  lines.push('╚══════════════════════════════════════════════════════════════╝');
  lines.push('');
  lines.push(`  Hardware: ${hw.name}`);
  lines.push(`  Peak compute:    ${hw.peakGFLOPS} GFLOPS`);
  lines.push(`  Peak bandwidth:  ${hw.peakBandwidthGBs} GB/s`);
  lines.push(`  Ridge point:     ${hw.ridgePoint.toFixed(2)} FLOP/byte`);
  lines.push('');
  lines.push('  ⚠  Timing: tuner-selected scheduled TIR (via compile()).');
  lines.push('     Main benchmark uses register-tiled path (compileRegTile) — see VERIFICATION.');
  lines.push('');

  // ─── Roofline plot (ASCII art) ───
  lines.push('  GFLOPS');
  lines.push(`  ${hw.peakGFLOPS.toFixed(1)} ┤─────────────── compute roof ─────────────────`);

  // Show where each kernel falls
  const sortedProfiles = [...profiles].sort((a, b) => b.achievedGFLOPS - a.achievedGFLOPS);

  // Scale for display
  const maxGFLOPS = hw.peakGFLOPS;
  const scaleY = (gflops: number) => {
    const levels = 8;
    return Math.round(gflops / maxGFLOPS * levels);
  };

  for (let level = 7; level >= 0; level--) {
    const gflopVal = (level / 8 * maxGFLOPS).toFixed(1);
    const padding = ' '.repeat(Math.max(0, 5 - gflopVal.length));
    let line = `  ${padding}${gflopVal} ┤ `;

    // Plot kernels at this level
    for (const p of sortedProfiles) {
      if (scaleY(p.achievedGFLOPS) === level) {
        line += ` ★ ${p.name} `;
      }
    }

    // Memory roof line (diagonal)
    if (level <= 4) {
      const aiAtLevel = level / 8 * maxGFLOPS / hw.peakBandwidthGBs;
      line += `   / (AI=${aiAtLevel.toFixed(2)})`;
    }

    lines.push(line);
  }

  lines.push('        └────────────────────────────────────────────');
  lines.push('         0.01  0.1   0.5   1.0   5.0   10.0');
  lines.push('                 Arithmetic Intensity (FLOP/byte)');
  lines.push('');

  // ─── Per-kernel table ───
  lines.push('  ┌──────────────────────────┬──────────┬──────────┬──────────┬─────────────────────────────┐');
  lines.push('  │ Kernel                   │ Time(ms) │  GFLOPS  │ AI(F/B)* │ Efficiency                  │');
  lines.push('  ├──────────────────────────┼──────────┼──────────┼──────────┼─────────────────────────────┤');

  for (const p of profiles) {
    const name = p.name.padEnd(24).slice(0, 24);
    const time = p.medianTimeMs.toFixed(4).padStart(8);
    const gflops = p.achievedGFLOPS.toFixed(3).padStart(8);
    const ai = p.arithmeticIntensity.toFixed(2).padStart(8);
    const effAI = p.memoryPlan.effectiveAI.toFixed(2);
    lines.push(`  │ ${name} │ ${time} │ ${gflops} │ ${ai} │ ${p.rooflineBar} │`);
    lines.push(`  │ ${''.padEnd(24)} │          │          │ eff:${effAI.padStart(4)} │                             │`);
  }

  lines.push('  └──────────────────────────┴──────────┴──────────┴──────────┴─────────────────────────────┘');
  lines.push('  * AI(F/B) = intrinsic (naive) arithmetic intensity — algorithm-level, schedule-independent.');
  lines.push('');

  // ─── Bottleneck summary ───
  const memBound = profiles.filter(p => p.bottleneck === 'memory');
  const compBound = profiles.filter(p => p.bottleneck === 'compute');

  if (memBound.length > 0) {
    lines.push(`  🧊 Memory-bound kernels (${memBound.length}): ${memBound.map(p => p.name).join(', ')}`);
    // Show which optimizations are already applied vs still suggested
    for (const p of memBound) {
      const sched = p.memoryPlan.scheduleInfo;
      const done = sched.applied.length > 0 ? `applied: [${sched.applied.join(', ')}]` : '';
      const remaining: string[] = [];
      if (!sched.tiled) remaining.push('increase tiling');
      // Suppress cache_read hint when: (a) kernel has only one output column,
      // or (b) tileJ=1 meaning j-tiling is trivial — cache_read needs tileJ>1.
      if (!sched.hasCacheRead && sched.jExtent > 1 && sched.tileJ > 1) remaining.push('cache_read');
      if (remaining.length > 0) {
        const hint = done ? `${done}  → still consider: ${remaining.join(', ')}` : `→ consider: ${remaining.join(', ')}`;
        lines.push(`     ${p.name}: ${hint}`);
      } else {
        lines.push(`     ${p.name}: ${done}  → model too small to reach ridge point (normal for tiny NN inference)`);
      }
    }
  }
  if (compBound.length > 0) {
    lines.push(`  🔥 Compute-bound kernels (${compBound.length}): ${compBound.map(p => p.name).join(', ')}`);
    lines.push('     → Recommendation: vectorize inner loops, reduce FLOPs (algorithmic)');
  }

  return lines.join('\n');
}
