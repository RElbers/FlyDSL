# Offline autotune configs

FlyDSL can serve a previously tuned config without benchmarking. This extends
the direct-JIT autotuner with one opt-in argument:

```python
@flyc.jit
def launch(x, out, N: fx.Constexpr[int], BLOCK: fx.Constexpr[int]):
    ...


tuned = autotune(
    configs=[Config(BLOCK=128), Config(BLOCK=256)],
    key=["N"],
    default=lambda x, out, N: Config(BLOCK=256),
    artifact_name="my_kernel",
)(launch)
```

`artifact_name` enables lookup when `FLYDSL_AUTOTUNE_CONFIG_DIR` is set. Normal
calls use the first available source:

```text
searched winner cache -> matching artifact -> default -> search
```

`FLYDSL_AUTOTUNE=1` bypasses those serving decisions, searches the existing
configs, updates the scratch winner cache, and atomically writes an artifact.
A normal fallback search updates only the scratch cache.
While artifact lookup is active, scratch winners use the same device descriptor
so a same-architecture product cannot shadow the matching artifact.

## Identity and compatibility

Artifact identity is the stable `artifact_name`, the declared `key` values,
and the call device's product name, target architecture, and compute-unit
count. Use a globally unique name for each kernel/config schema. The JSON is
self-describing; its filename is an identity digest.

The declared `key` owns the portable tuning axes. Include every
shape, dtype, layout, or mode that can change the winner. Keep structural knobs
as JIT `Constexpr` parameters on the existing entry point; offline tuning does
not need a build factory or a second key callback.

Artifacts intentionally do not include a compiler or kernel-source fingerprint.
Treat them as reviewed deployment inputs, and retune after a compiler, kernel,
compile-hint, or search-space change that can affect the winner.

The scratch winner cache has the same blind spot: it fingerprints the device,
toolchain, environment and compile hints, but not the adopter's kernel source or
search space. An adopter that needs stale scratch winners invalidated should
declare an integer schema parameter on its entry point and list it in `key`, then
bump it with any change that can move the winner. Softmax does this with
`tuning_schema`.

## Candidate correctness gate

`validate_hook(sig_args)` returns a context manager around one untimed candidate
launch. Code before `yield` can poison outputs; code after `yield` validates the
result, so skipped or partial stores cannot inherit a previous candidate's data.
The launch uses the same stream, compile hints, reset/restore policy and arguments
as timing, but validation work never affects ranking. `sig_args` maps every kernel
parameter name to its value, including positional tensors. Softmax uses this hook
to fill its output with NaNs before launch and check numerics afterward.

Raising from the hook rejects that candidate. If every candidate is rejected the
search raises `RuntimeError("All autotune configs failed")` with the last failure
chained, so a numerical rejection stays distinguishable from a compile failure.
Use it wherever a candidate could launch successfully and still compute the wrong
answer, and hold every candidate to the same tolerance as the default.

## Device timing contract

The shared `do_bench` timer queues a GPU-side backlog before batched event
windows. This is required for sub-100 µs kernels: a fresh event pair on an empty
stream can time the host enqueue gap instead of the kernel. Each window averages
several launches, and the reported value is the median across windows. The
callable must enqueue asynchronous work on the current stream and must not
synchronize internally.

For Softmax results within 2% of the measured minimum, selection prefers the
compatibility default, then a config without an explicit occupancy override,
then the candidate packing more rows per block. This prevents event granularity
from turning equivalent 6--10 µs candidates into unstable deployment artifacts;
an improvement outside the band still wins normally. Softmax uses 10 warmup and
100 measured launches, split into five backlogged event windows; the larger
sample stabilized bandwidth-scale rows that moved by more than the tie band with
the generic 25-launch default.

## Adopters

| Kernel | Module | `artifact_name` | Tuned axes |
|---|---|---|---|
| RMSNorm | `kernels/norm/rmsnorm_autotune.py` | `rmsnorm` | `BLOCK_THREADS`, `waves_per_eu` |
| Softmax forward | `kernels/norm/softmax_autotune.py` | `softmax_fwd` | full-row threads, `waves_per_eu`, threads/rows per block for short rows |

Softmax backward is not an adopter yet; its existing kernel and dispatch are
unchanged by `softmax_fwd` artifacts.

## Failure behavior

FlyDSL ignores missing, unreadable, mismatched, or structurally invalid
artifacts and continues normal lookup to the default or search path. Artifact
config values cannot overwrite arguments that the caller supplies or declared
key axes. Values must preserve their types when encoded as JSON. `Config.pre_hook`
is process-local code, so it blocks forced artifact generation.

Once a matching artifact has been accepted, its compile, launch, and runtime
errors propagate normally; FlyDSL does not hide them by retrying the default.
If forced generation cannot establish a device identity or write its artifact,
it fails clearly without caching the winner.

Generate deployment artifacts on the intended GPU under controlled benchmark
conditions. CI should verify deterministic emit-and-load behavior, not commit a
winner selected from noisy shared-runner timing.

## Compiling configs in parallel

`flydsl.precompile.precompile(jobs, max_workers)` compiles kernel
specializations into the JIT disk cache in up to `max_workers` worker
processes. A later call to each specialization loads it instead of compiling
it, so a search then spends its time benchmarking. Adding or removing a
`precompile` call changes only how long the calls take.

Write a `CompileJob` like the `flyc.compile` call that would compile the same
specialization. Tensors only need the right shape, strides and dtype, so CPU
tensors work:

```python
from flydsl.precompile import CompileJob, precompile

x = torch.empty(4096)
out = torch.empty(4096)
jobs = [CompileJob(launch, x, out, N=4096, BLOCK=block) for block in (128, 256)]
errors = precompile(jobs, max_workers=16)
```

Each entry of `errors` is None when its job compiled, or the reason it did not.
A job that did not compile costs nothing extra: its call compiles as usual.

Give compile hints as `flyc.compile[hints]` takes them:

```python
CompileJob[{"waves_per_eu": 2}](launch, x, out, N=4096, BLOCK=256)
```

For a kernel built by a factory, pass a `functools.partial` of the factory,
then the call of the launcher it returns:

```python
CompileJob(functools.partial(build_kernel, BLOCK=256), x, out)
```

### In an autotune search

With `@autotune(..., precompile_workers=16)`, each search compiles its configs
in up to 16 workers before it benchmarks them. A tuner that leaves
`precompile_workers` unset uses `FLYDSL_AUTOTUNE_PRECOMPILE_WORKERS`.

Every `precompile` call starts its workers again, and each search makes its own
call. When you will tune many shapes, build the jobs for all of them and compile
them in one call instead. Create each job in its config's hint context,
as the autotuner does when it compiles the config:

```python
from flydsl.compiler.kernel_function import CompilationContext

jobs = []
for x, out in inputs:  # one pair per shape you will tune
    for c in configs:
        with CompilationContext.compile_hints(c.compiler_opts()):
            jobs.append(CompileJob(launch, x, out, N=x.numel(), **c.all_kwargs()))
precompile(jobs, max_workers=64)
```

`CompileJob` takes the jit function, not the tuner. When `@autotune` decorates
the jit function directly, the module-level name is the tuner, and its `.fn` is
the jit function:

```python
@autotune(configs=configs, key=["N"])
@flyc.jit
def launch(x, out, N: fx.Constexpr[int], BLOCK: fx.Constexpr[int]): ...


CompileJob(launch.fn, x, out, N=4096, BLOCK=128)
```

### For another library's tuner

Run `precompile` before the tuner, in the same process or an earlier one with
the same `FLYDSL_RUNTIME_CACHE_DIR`, with one job for each call the tuner will
make. To check that the jobs match those calls, run the tuner once with
`FLYDSL_RUNTIME_RUN_ONLY=1`: a call that was not precompiled then raises.

`precompile` needs no GPU. To build a cache for machines of another
architecture, use CPU tensors, set `FLYDSL_GPU_ARCH` to that architecture, and
run where there is no GPU or one of that architecture, since a kernel may query
the GPU it finds. The machines that load the cache need the same FlyDSL build.

### Limits

- Workers import the jit function, or the factory, by its module and name. Define
  it at the top level of a module the workers can import: not inside another
  function, and not in the script you run, which the workers never run. The same
  applies to an `@fx.struct` type the kernel takes. A job that breaks this is
  reported, and its call compiles as usual.
- Workers import the kernel's module afresh. If the caller has changed state
  the kernel reads since import, the call compiles instead of loading.
- A module that starts the GPU runtime when imported cannot be precompiled.
- Kernels with post-load initializers, such as a `link_extern` with a
  `module_init_fn`, are never written to the disk cache.
- Nothing is compiled while the disk cache is off or IR dumps
  (`FLYDSL_DUMP_IR`, `FLYDSL_DEBUG_DUMP_ASM`) bypass it.
- Precompiling needs Linux. On other platforms, every job reports why, and its
  call compiles as usual.
