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