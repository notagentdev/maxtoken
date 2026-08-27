# maxtoken-kernel-cache

`maxtoken-kernel-cache` is a companion wheel for `maxtoken`. It ships prebuilt
TVM FFI kernel shared libraries so common runtime paths can load `.so` files
directly instead of compiling with nvcc on first use.

## Build Both Wheels

Use the top-level helper to build the runtime wheel and the matching kernel-cache
wheel in one command:

```bash
scripts/build-release-wheels.sh
```

By default, artifacts are written to `dist/`:

```text
dist/maxtoken-<version>-cp312-cp312-linux_x86_64.whl
dist/maxtoken_kernel_cache-<version>+cu130-py3-none-linux_x86_64.whl
```

Useful knobs:

```bash
MAXTOKEN_BUILD_OUT_DIR=/tmp/maxtoken-dist \
MAXTOKEN_BUILD_PYTHON=.venv/bin/python \
scripts/build-release-wheels.sh
```

To build only a subset of the cache for quick checks:

```bash
MAXTOKEN_KERNEL_CACHE_SPECS=maxtoken__store_1024_128_1_false \
scripts/build-release-wheels.sh
```

## Wheel Contents

The cache wheel contains one directory per kernel:

```text
maxtoken_kernel_cache/
  jit_cache/
    maxtoken__store_1024_128_1_false/
      maxtoken__store_1024_128_1_false.so
```

At runtime, `maxtoken.kernel.utils.load_jit()` and `load_aot()` look for
`maxtoken_kernel_cache.get_jit_cache_dir()` and load
`<jit_cache>/<kernel_name>/<kernel_name>.so` before falling back to JIT.

## Install

`install.sh` installs both wheels. Pass both explicitly:

```bash
MAXTOKEN_WHEEL=dist/maxtoken-0.1.1-cp312-cp312-linux_x86_64.whl \
MAXTOKEN_KERNEL_CACHE_WHEEL=dist/maxtoken_kernel_cache-0.1.1+cu130-py3-none-linux_x86_64.whl \
bash install.sh
```

If the cache wheel is next to the runtime wheel, `install.sh` can auto-detect the
sibling `maxtoken_kernel_cache-*.whl` when only `MAXTOKEN_WHEEL` is set.

For release validation, disable runtime JIT:

```bash
MAXTOKEN_DISABLE_JIT=1 mt serve --model <path>
```

With this flag, any cache miss fails immediately instead of compiling at runtime.
