import { expect, test } from 'vitest';
import { Tensor } from '../../src/tensor/tensor.js';

test('static constructors create expected shapes and fills', () => {
  const zeros = Tensor.zeros([2, 3]);
  const ones = Tensor.ones([2, 2]);
  const full = Tensor.full([1, 4], 7);
  const from = Tensor.fromArray([1, 2, 3, 4], [2, 2]);

  expect(zeros.shape).toEqual([2, 3]);
  expect(Array.from(zeros.data)).toEqual([0, 0, 0, 0, 0, 0]);
  expect(Array.from(ones.data)).toEqual([1, 1, 1, 1]);
  expect(Array.from(full.data)).toEqual([7, 7, 7, 7]);
  expect(from.shape).toEqual([2, 2]);
});

test('computeStrides and ndim/size reflect shape metadata', () => {
  const arr = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
  expect(Tensor.computeStrides([2, 3, 4])).toEqual([12, 4, 1]);
  expect(arr.ndim).toBe(2);
  expect(arr.size).toBe(6);
});

test('rand and randn produce correct shape and finite values', () => {
  const rand = Tensor.rand([2, 2]);
  const randn = Tensor.randn([3, 1]);
  expect(rand.shape).toEqual([2, 2]);
  expect(randn.shape).toEqual([3, 1]);
  expect(Array.from(rand.data).every(Number.isFinite)).toBe(true);
  expect(Array.from(randn.data).every(Number.isFinite)).toBe(true);
});

test('get and set use flat indexing correctly', () => {
  const arr = Tensor.zeros([2, 2]);
  arr.set([1, 0], 5);
  expect(arr.get([1, 0])).toBe(5);
});

test('reshape copies data and transpose swaps axes for 2D', () => {
  const arr = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
  const reshaped = arr.reshape([4, 1]);
  const transposed = arr.transpose();

  expect(reshaped.shape).toEqual([4, 1]);
  expect(Array.from(reshaped.data)).toEqual([1, 2, 3, 4]);
  expect(transposed.shape).toEqual([2, 2]);
  expect(Array.from(transposed.data)).toEqual([1, 3, 2, 4]);
});

test('reshape/transpose reject invalid shapes and ranks', () => {
  const arr = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
  expect(() => arr.reshape([3, 3])).toThrow('Cannot reshape');
  expect(() => Tensor.fromArray([1, 2, 3], [3]).transpose()).toThrow('transpose() only for 2D');
});

test('scalar arithmetic and neg operate elementwise', () => {
  const arr = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
  expect(Array.from(arr.add(1).data)).toEqual([2, 3, 4, 5]);
  expect(Array.from(arr.sub(1).data)).toEqual([0, 1, 2, 3]);
  expect(Array.from(arr.mul(2).data)).toEqual([2, 4, 6, 8]);
  expect(Array.from(arr.div(2).data)).toEqual([0.5, 1, 1.5, 2]);
  expect(Array.from(arr.neg().data)).toEqual([-1, -2, -3, -4]);
});

test('broadcast binary ops work and incompatible shapes throw', () => {
  const a = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
  const b = Tensor.fromArray([10, 20], [2]);
  const c = Tensor.fromArray([2], [1]);

  expect(Array.from(a.add(b).data)).toEqual([11, 22, 13, 24]);
  expect(Array.from(a.sub(b).data)).toEqual([-9, -18, -7, -16]);
  expect(Array.from(a.mul(c).data)).toEqual([2, 4, 6, 8]);
  expect(Array.from(a.div(c).data)).toEqual([0.5, 1, 1.5, 2]);
  expect(() => a.add(Tensor.fromArray([1, 2, 3], [3]))).toThrow('Cannot broadcast shapes');
});

test('sum/max/mean support full reduction, axis, negative axis, and keepDim', () => {
  const arr = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);

  expect(Array.from(arr.sum().data)).toEqual([21]);
  expect(arr.sum(1, true).shape).toEqual([2, 1]);
  expect(Array.from(arr.sum(1, true).data)).toEqual([6, 15]);
  expect(Array.from(arr.max(0).data)).toEqual([4, 5, 6]);
  expect(Array.from(arr.mean(-1).data)).toEqual([2, 5]);
});

test('exp/log/abs/gt return expected transformed tensors', () => {
  const arr = Tensor.fromArray([-1, 0, 1], [3]);
  const exp = arr.exp();
  const log = Tensor.fromArray([1, Math.E], [2]).log();
  const abs = arr.abs();
  const gt = arr.gt(0);

  expect(Number(exp.data[0].toFixed(3))).toBe(0.368);
  expect(Number(log.data[1].toFixed(3))).toBe(1);
  expect(Array.from(abs.data)).toEqual([1, 0, 1]);
  expect(Array.from(gt.data)).toEqual([0, 0, 1]);
});

test('matmul computes product and validates inputs', () => {
  const a = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
  const b = Tensor.fromArray([5, 6, 7, 8], [2, 2]);
  const result = a.matmul(b);

  expect(Array.from(result.data)).toEqual([19, 22, 43, 50]);
  expect(() => Tensor.fromArray([1, 2], [2]).matmul(b)).toThrow('matmul requires 2D arrays');
  expect(() => a.matmul(Tensor.fromArray([1, 2, 3, 4, 5, 6], [3, 2]))).toThrow('matmul shape mismatch');
});

test('clone, allClose, and toString cover utility paths', () => {
  const arr = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
  const clone = arr.clone();
  clone.data[0] = 99;

  expect(arr.data[0]).toBe(1);
  expect(arr.allClose(Tensor.fromArray([1.000001, 2, 3, 4], [2, 2]))).toBe(true);
  expect(arr.allClose(Tensor.fromArray([9, 2, 3, 4], [2, 2]))).toBe(false);
  expect(Tensor.fromArray([1, 2], [2]).toString().startsWith('Tensor(') || Tensor.fromArray([1, 2], [2]).toString().startsWith('[')).toBe(true);
  expect(arr.toString().includes('Tensor(shape=[')).toBe(true);
  expect(Tensor.zeros([2, 2, 2]).toString().includes('size=8')).toBe(true);
});
