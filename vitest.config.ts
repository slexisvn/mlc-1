import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'text-summary'],
      exclude: [
        'tests/**',
        'src/main.ts',
        'src/test_wat.ts',
        'src/test_decorator_compile.ts',
      ],
    },
  },
});