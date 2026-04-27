// ═══════════════════════════════════════════════════════════════
//  Auto-Tuner v2 — Simulated Annealing Search
//
//  Upgrade from brute-force grid search to simulated annealing:
//
//  1. Start with a random schedule configuration
//  2. In each iteration, make a small random perturbation
//  3. If the new config is faster → always accept
//  4. If the new config is slower → accept with probability
//     P = exp(-delta/temperature), where temperature decreases
//     over time (cooling schedule)
//  5. This allows escaping local minima early on (high temp),
//     then converging to a good solution (low temp)
//
//  Real MLC frameworks (TVM/Ansor) use evolutionary search +
//  cost models (XGBoost). We demonstrate the core principle
//  with simulated annealing which is simpler but effective.
//
//  Also includes:
//  - Multiple independent restart ("multi-start")
//  - Performance history tracking for visualization
//  - Comparison with brute-force best
// ═══════════════════════════════════════════════════════════════

import { PrimFunc } from '../ir/low_level.js';
import { Schedule } from '../transform/schedule.js';
import { codegenJS, compile } from '../codegen/js_codegen.js';

// ═══════════════════════════════════════
//  Configuration Space
// ═══════════════════════════════════════

export interface TuneConfig {
  tileJ: number;
  tileK: number;
  unrollInner: boolean;
  parallelOuter: boolean;
}

export interface TuneResult {
  config: TuneConfig;
  code: string;
  medianTime: number;
}

// Valid tile sizes (must divide the loop extent or use findDivisor)
function getValidTileSizes(extent: number): number[] {
  const candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
  return candidates.filter(t => t <= extent && extent % t === 0);
}

// Random element from array
function randomChoice<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

// ═══════════════════════════════════════
//  Schedule Application
// ═══════════════════════════════════════

function applyConfig(func: PrimFunc, config: TuneConfig): PrimFunc | null {
  try {
    const sch = new Schedule(func);
    const loops = func.getLoops();
    const jLoop = loops.find(l => l.loopVar.name === 'j');
    const kLoop = loops.find(l => l.loopVar.name === 'k');

    if (!jLoop || !kLoop) return null;
    if (config.tileJ <= 1 && config.tileK <= 1) return func; // no tiling

    const [jOuter, jInner] = sch.split(jLoop.loopVar, config.tileJ);
    const [kOuter, kInner] = sch.split(kLoop.loopVar, config.tileK);
    sch.reorder([jOuter, kOuter, jInner, kInner]);

    if (config.unrollInner && config.tileJ <= 16) {
      sch.unroll(jInner);
    }
    if (config.parallelOuter) {
      sch.parallel(jOuter);
    }

    return sch.build();
  } catch {
    return null;
  }
}

// ═══════════════════════════════════════
//  Benchmark
// ═══════════════════════════════════════

function benchmark(fn: Function, buffers: Float32Array[], iterations: number): number {
  // Warmup
  for (let i = 0; i < 5; i++) fn(...buffers);

  // Measure
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn(...buffers);
    const end = performance.now();
    times.push(end - start);
  }

  // Return median
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

// ═══════════════════════════════════════
//  Simulated Annealing Core
// ═══════════════════════════════════════

export interface AnnealingOptions {
  maxIterations: number;       // total iterations
  initialTemp: number;         // starting temperature
  coolingRate: number;         // temperature *= coolingRate each step
  minTemp: number;             // stop when temp drops below this
  numRestarts: number;         // number of independent starts
  benchIterations: number;     // iterations per benchmark
}

export const DEFAULT_ANNEALING: AnnealingOptions = {
  maxIterations: 50,
  initialTemp: 1.0,
  coolingRate: 0.92,
  minTemp: 0.01,
  numRestarts: 3,
  benchIterations: 30,
};

export interface AnnealingHistory {
  iteration: number;
  config: TuneConfig;
  time: number;
  bestTime: number;
  temperature: number;
  accepted: boolean;
}

function perturbConfig(
  config: TuneConfig,
  validJ: number[],
  validK: number[]
): TuneConfig {
  // Make a small random change to the config
  const dimension = Math.floor(Math.random() * 3);

  switch (dimension) {
    case 0: {
      // Perturb tileJ
      const idx = validJ.indexOf(config.tileJ);
      const newIdx = Math.max(0, Math.min(validJ.length - 1,
        idx + (Math.random() < 0.5 ? -1 : 1)));
      return { ...config, tileJ: validJ[newIdx] };
    }
    case 1: {
      // Perturb tileK
      const idx = validK.indexOf(config.tileK);
      const newIdx = Math.max(0, Math.min(validK.length - 1,
        idx + (Math.random() < 0.5 ? -1 : 1)));
      return { ...config, tileK: validK[newIdx] };
    }
    case 2:
      // Toggle unroll
      return { ...config, unrollInner: !config.unrollInner };
    default:
      return config;
  }
}

function evaluateConfig(
  func: PrimFunc,
  config: TuneConfig,
  testBuffers: Float32Array[],
  benchIters: number
): number | null {
  const scheduled = applyConfig(func, config);
  if (!scheduled) return null;

  try {
    const fn = compile(scheduled);
    return benchmark(fn, testBuffers, benchIters);
  } catch {
    return null;
  }
}

function simulatedAnnealing(
  func: PrimFunc,
  validJ: number[],
  validK: number[],
  testBuffers: Float32Array[],
  options: AnnealingOptions
): { best: TuneConfig; bestTime: number; history: AnnealingHistory[] } {
  const history: AnnealingHistory[] = [];

  // Random initial config
  let current: TuneConfig = {
    tileJ: randomChoice(validJ),
    tileK: randomChoice(validK),
    unrollInner: Math.random() < 0.3,
    parallelOuter: false,
  };

  let currentTime = evaluateConfig(func, current, testBuffers, options.benchIterations);
  if (currentTime === null) {
    // Fallback to safe config
    current = { tileJ: validJ[0], tileK: validK[0], unrollInner: false, parallelOuter: false };
    currentTime = evaluateConfig(func, current, testBuffers, options.benchIterations) || Infinity;
  }

  let best = { ...current };
  let bestTime = currentTime;
  let temp = options.initialTemp;

  for (let iter = 0; iter < options.maxIterations && temp > options.minTemp; iter++) {
    // Perturb
    const candidate = perturbConfig(current, validJ, validK);
    const candidateTime = evaluateConfig(func, candidate, testBuffers, options.benchIterations);

    if (candidateTime === null) {
      temp *= options.coolingRate;
      continue;
    }

    // Accept or reject
    const delta = candidateTime - currentTime;
    const acceptProb = delta < 0 ? 1.0 : Math.exp(-delta / (temp * currentTime));
    const accepted = Math.random() < acceptProb;

    history.push({
      iteration: iter,
      config: candidate,
      time: candidateTime,
      bestTime,
      temperature: temp,
      accepted,
    });

    if (accepted) {
      current = candidate;
      currentTime = candidateTime;

      if (candidateTime < bestTime) {
        best = { ...candidate };
        bestTime = candidateTime;
      }
    }

    temp *= options.coolingRate;
  }

  return { best, bestTime, history };
}

// ═══════════════════════════════════════
//  Main Auto-Tune Function (v2)
// ═══════════════════════════════════════

export interface AutoTuneResult {
  best: TuneResult;
  naiveTime: number;
  speedup: number;
  totalConfigs: number;
  history: AnnealingHistory[];
  allRestarts: { bestTime: number; iterations: number }[];
}

export function autoTune(
  func: PrimFunc,
  options: Partial<AnnealingOptions> = {}
): AutoTuneResult {
  const opts = { ...DEFAULT_ANNEALING, ...options };

  // Find valid tile sizes
  const loops = func.getLoops();
  const jLoop = loops.find(l => l.loopVar.name === 'j');
  const kLoop = loops.find(l => l.loopVar.name === 'k');

  if (!jLoop || !kLoop) {
    const code = codegenJS(func);
    return {
      best: { config: { tileJ: 0, tileK: 0, unrollInner: false, parallelOuter: false }, code, medianTime: 0 },
      naiveTime: 0,
      speedup: 1,
      totalConfigs: 0,
      history: [],
      allRestarts: [],
    };
  }

  const validJ = getValidTileSizes(jLoop.forNode.extent);
  const validK = getValidTileSizes(kLoop.forNode.extent);

  console.log(`  Auto-tuning ${func.name} (simulated annealing)...`);
  console.log(`    Search space: ${validJ.length} × ${validK.length} × 2 = ${validJ.length * validK.length * 2} configs`);
  console.log(`    Strategy: ${opts.numRestarts} restarts × ${opts.maxIterations} iterations`);
  console.log(`    Cooling: T₀=${opts.initialTemp}, rate=${opts.coolingRate}`);

  // Create test data
  const testBuffers = func.params.map(p => {
    const size = p.shape.reduce((a, b) => a * b, 1);
    const buf = new Float32Array(size);
    for (let i = 0; i < size; i++) buf[i] = Math.random() * 0.1;
    return buf;
  });

  // Measure naive baseline
  const naiveFn = compile(func);
  const naiveTime = benchmark(naiveFn, testBuffers, opts.benchIterations);
  console.log(`    Naive baseline: ${naiveTime.toFixed(4)}ms`);

  // Run multiple restarts
  let globalBest: TuneConfig = { tileJ: validJ[0], tileK: validK[0], unrollInner: false, parallelOuter: false };
  let globalBestTime = naiveTime;
  const allHistory: AnnealingHistory[] = [];
  const restartResults: { bestTime: number; iterations: number }[] = [];

  for (let r = 0; r < opts.numRestarts; r++) {
    const { best, bestTime, history } = simulatedAnnealing(
      func, validJ, validK, testBuffers, opts
    );

    const tag = `tile(${best.tileJ},${best.tileK})${best.unrollInner ? '+unroll' : ''}`;
    console.log(`    Restart ${r + 1}: best=${bestTime.toFixed(4)}ms [${tag}]`);

    restartResults.push({ bestTime, iterations: history.length });
    allHistory.push(...history.map(h => ({ ...h, iteration: h.iteration + r * opts.maxIterations })));

    if (bestTime < globalBestTime) {
      globalBest = best;
      globalBestTime = bestTime;
    }
  }

  const speedup = naiveTime / globalBestTime;
  const bestTag = `tile(${globalBest.tileJ},${globalBest.tileK})${globalBest.unrollInner ? '+unroll' : ''}`;
  console.log(`    ───────────────────────────────`);
  console.log(`    Best overall: ${globalBestTime.toFixed(4)}ms [${bestTag}]`);
  console.log(`    Speedup vs naive: ${speedup.toFixed(2)}x`);

  // Get the code for the best config
  const bestScheduled = applyConfig(func, globalBest)!;
  const bestCode = codegenJS(bestScheduled);

  return {
    best: { config: globalBest, code: bestCode, medianTime: globalBestTime },
    naiveTime,
    speedup,
    totalConfigs: allHistory.length,
    history: allHistory,
    allRestarts: restartResults,
  };
}

// ═══════════════════════════════════════
//  Search Progress Visualization
// ═══════════════════════════════════════

export function printSearchProgress(history: AnnealingHistory[]): string {
  const lines: string[] = [];
  lines.push('Search Progress:');

  // Show convergence curve (best time over iterations)
  const bucketSize = Math.max(1, Math.floor(history.length / 20));
  let prevBest = Infinity;

  for (let i = 0; i < history.length; i += bucketSize) {
    const bucket = history.slice(i, i + bucketSize);
    const bestInBucket = Math.min(...bucket.map(h => h.bestTime));
    const temp = bucket[0]?.temperature || 0;
    const accepted = bucket.filter(h => h.accepted).length;

    if (bestInBucket < prevBest || i === 0) {
      prevBest = bestInBucket;
    }

    const barWidth = 30;
    const maxTime = history[0]?.time || 1;
    const filled = Math.round((1 - prevBest / maxTime) * barWidth);
    const bar = '▓'.repeat(Math.max(0, filled)) + '░'.repeat(Math.max(0, barWidth - filled));

    lines.push(`  [${String(i).padStart(3)}] ${bar} ${prevBest.toFixed(4)}ms  T=${temp.toFixed(3)}  acc=${accepted}/${bucket.length}`);
  }

  return lines.join('\n');
}
