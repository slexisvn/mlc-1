import { expect, test } from 'vitest';
import { mkDenseWithAccFunc, mkElemwiseFunc } from '../helpers/tir_builders.js';
import { lowerOp } from '../../src/lower/lowering.js';
import {
  applyConfig,
  autoTune,
  DEFAULT_ANNEALING,
  printSearchProgress,
  type AnnealingHistory,
  type TuneConfig,
} from '../../src/tune/auto_tune.js';

test('applyConfig returns original func for no-op config', () => {
  const func = mkDenseWithAccFunc(2, 8, 8);
  const cfg: TuneConfig = {
    tileI: 1,
    tileJ: 1,
    tileK: 1,
    unrollInner: false,
    parallelOuter: false,
    useCacheRead: false,
  };

  const result = applyConfig(func, cfg);
  expect(result).toBe(func);
});

test('applyConfig tiles loops and can add W_local cache buffer', () => {
  const func = mkDenseWithAccFunc(4, 16, 16);
  const cfg: TuneConfig = {
    tileI: 2,
    tileJ: 8,
    tileK: 8,
    unrollInner: true,
    parallelOuter: true,
    useCacheRead: true,
  };

  const result = applyConfig(func, cfg);
  expect(result !== null).toBe(true);
  const loopNames = result!.getLoops().map(l => l.loopVar.name);
  expect(loopNames.includes('j_outer')).toBe(true);
  expect(loopNames.includes('k_outer')).toBe(true);
  expect(result!.allocations.some(a => a.name === 'W_local')).toBe(true);
});

test('applyConfig returns null when required j/k loops are absent', () => {
  const func = mkElemwiseFunc(4, 8);
  const cfg: TuneConfig = {
    tileI: 1,
    tileJ: 2,
    tileK: 2,
    unrollInner: false,
    parallelOuter: false,
    useCacheRead: false,
  };

  const result = applyConfig(func, cfg);
  expect(result).toBeNull();
});

test('autoTune returns trivial result when j/k loops are missing', () => {
  const result = autoTune(mkElemwiseFunc(4, 8), { maxIterations: 2, numRestarts: 1, benchIterations: 1 });
  expect(result.totalConfigs).toBe(0);
  expect(result.speedup).toBe(1);
  expect(result.history.length).toBe(0);
});

test('autoTune searches a dense kernel and returns code/history', () => {
  const dense = lowerOp('nn.dense', [[2, 8], [8, 8]]);
  if (!dense) throw new Error('expected lowerOp(nn.dense) to produce a PrimFunc');

  const result = autoTune(dense, {
    maxIterations: 3,
    numRestarts: 1,
    benchIterations: 1,
    initialTemp: DEFAULT_ANNEALING.initialTemp,
    coolingRate: DEFAULT_ANNEALING.coolingRate,
    minTemp: DEFAULT_ANNEALING.minTemp,
  });

  expect(result.best.code.length).toBeGreaterThan(0);
  expect(result.naiveTime).toBeGreaterThan(0);
  expect(result.history.length <= 3).toBe(true);
  expect(result.allRestarts.length).toBe(1);
});

test('printSearchProgress formats annealing history', () => {
  const history: AnnealingHistory[] = [
    {
      iteration: 0,
      config: { tileI: 1, tileJ: 4, tileK: 4, unrollInner: false, parallelOuter: false, useCacheRead: false },
      time: 1.0,
      bestTime: 1.0,
      temperature: 1.0,
      accepted: true,
    },
    {
      iteration: 1,
      config: { tileI: 1, tileJ: 8, tileK: 4, unrollInner: true, parallelOuter: false, useCacheRead: true },
      time: 0.8,
      bestTime: 0.8,
      temperature: 0.92,
      accepted: true,
    },
  ];

  const printed = printSearchProgress(history);
  expect(printed.includes('Search Progress:')).toBe(true);
  expect(printed.includes('0.8000ms')).toBe(true);
});

