# Numba Notes

## @jit decorator
- Basic Usage: `@jit(nopython=True, nogil=True, cache=True, parallel=True)`
  - nopython - enables JIT compilation
  - nogil - removes GIL, can introduce pitfalls of multi-threaded programs (race conditions, syncing, etc)
  - cache - saves function to FILE cache
  - parallel - tries automatic parallelization, requires nopython=True
  - Can be given input and output types for ahead of time compilation. `int64(int64, int64)`

## @vectorize decorator
- For creating Numpy ufuncs that operate on an array elementwise
- Basic Usage: `@vectorize(type specifiers, target="target", other @jit args)`
- Supported targets at `cpu`, `parallel`, and `cuda`
- Signatures should be passed for ahead of time compilation. Lower precision
  should be first then higher precision or dispatching might not work right. The
  syntax is as follows
    ```
    @vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)])
    ```
- Use `@guvectorize` for more complex tasks


## @jitclass decorator
- Still experimental, if I need this check back


## Automatic Parallelization
- Using `parallel=True` argument in the jit decorator enables a bunch of parallel optimizations
- use `prange` instead of `range` to indicate a loop that can be parallelized
- reductions (like `+=`) can be done in the loop. See the numba docs for details
- using `parallel_diagnostics()` to see what was done and maybe how to improve it

## @stencil decorator
- apply the same operation to every element in an array
- requires relative indexing. I.e. the current element being operated on is `a[0]' even if it's actually the nth element and the next element in is `a[1]`
- contains arguments for border (ghost cell) handling. Currently it only supports setting a constant