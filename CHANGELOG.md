# Changelog

All notable changes to ProbJax will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — preparing 0.2.0

### Added
- `smc(logprior_fn, loglikelihood_fn, move=...)`: one tempered-SMC kernel on
  the shared kernel contract, driven by `run`. A fixed schedule passes each
  temperature as a step argument; `target_ess=` picks them adaptively and
  further steps after temperature 1 are no-ops. `waste_free=p` keeps every
  move state (Dau and Chopin), `batch_size=` bounds memory, and a vector
  temperature with a per-observation likelihood tempers data in one by one.
  The state carries `log_evidence`; params are the move's params
- `default_adaptation` for `smc`: `particle_adaptor` (inverse mass matrix from
  the population) and `acceptance_rate_adaptor` (step size from the move's
  acceptance), plain adaptors usable on their own
- `diffusion_smc(guide, times)`: twisted SMC along a reverse diffusion. A
  `Guide` supplies the reverse SDE, the twist and optional gradient guidance;
  `neural.diffusion_guide(model, V)` steers a pretrained diffusion model
  towards `p_phi(x) exp(V(x))` (the twist fades in with the signal and its
  gradient steers the proposal, with the matching weight correction), and
  `score_guide(log_target, schedule, shape)` is training-free reverse
  diffusion SMC
- `run(..., collect=None)` (the new default) records the position of
  MCMC-style states and nothing for population states
- `rejection(logdensity_fn, proposal=density)`: rejection sampling as a
  kernel whose every step is one exact, independent draw. Params are the
  proposal's params and `log_bound`; `bound_adaptor` (the default adaptation)
  learns the bound, and a learned proposal trains with `fit_adaptor`
- Learned components in `probjax.inference`: `Density(sample, logpdf)` takes
  its parameters explicitly, so a proposal's or variational family's weights
  live in a kernel's `params`, inside the compiled scan, `vmap`-able over
  chains. `probjax.inference.neural.density(model)` and `bijection(flow)` split
  a `probjax.nn` model into such pure functions and their `nnx.Param`s
- `fit_adaptor(loss_fn, optimizer, ...)` trains learned parameters on the
  chain's own output inside the scan: a ring buffer, `every`/`num_fit_steps`
  for in-scan or block-wise fitting, and `axis_name=` to train one set of
  parameters shared by chains run under `jax.vmap`.
  `maximum_likelihood(density)` is the matching weighted-NLL loss
- `probjax.inference.run`, `warmup` and `step`: three compiled functions that
  drive any kernel. `run(key, kernel, state, params, num_steps, *, args,
  adaptor, collect, thin, progress)` returns a `RunResult(state, params,
  trace)`, where `trace` maps each collected state or info field to its values
- Adaptation while sampling: `run(..., adaptor=...)` updates the parameters
  after every transition and returns the finalized ones
- `pooled(adaptor, axis_name="chain")` shares one adaptation across chains run
  under `jax.vmap(..., axis_name=...)`
- `linear_drift(A=...)` accepts structured linear operators (or a function of
  time returning one) for `odeint`/`sdeint`; `linear_exact` and
  `linear_exact_sde` dispatch `expm`/`phi_1` on the operator, and
  `linear_exact_sde` now supports diagonal (vector) drifts with a closed-form
  process-noise covariance
- `odeint`/`sdeint` take `solver=` as a registered name, a solver class or a
  configured solver: calling a solver class with only options configures it,
  e.g. `odeint(f, y0, ts, solver=implicit_euler(jacobian="matrix_free"))`
- `newton_raphson`/`root` take `jacobian="dense" | "matrix_free"` and any
  `linear_solver` accepted by `probjax.utils.linear_operator.solve` (including
  lineax solvers); the implicit Runge-Kutta, implicit Euler and Adams-Moulton/BDF
  builders forward these options to their Newton solves
- Structured linear operators `Diagonal`, `Dense` and matrix-free `Function`
  over arbitrary input/output shapes, with `solve`/`logdet` choosing the method
  by structure (`psd=` selects Cholesky/CG), `expm`/`phi` matrix functions,
  matrix-free `Jacobian(fn, x, args)`, and optional lineax solvers
  (`pip install 'probjax[lineax]'`)
- `DiffusionTransformer` accepts an optional per-token `local_context` of shape
  `[..., tokens, local_context_dim]`, concatenated to the tokens at the input,
  next to the global `context`
- Generative models take any named model input, not only `context`: `fit(rng,
  {"data": x, "local_context": l})` (the data dict carries every model input; the old
  `fit(..., context=)` keyword is gone, `weights=` stays a loss option), `dist.sample(rng, shape, local_context=...)`,
  `as_dist(inputs_spec={...}, inputs={...})` and `dist.condition(...)` forward
  them to the network as keyword arguments (`context=` stays as sugar)
- Spline `tail_logit_scale` option to damp boundary-slope predictions while
  retaining learnable tails and identity initialization.
- Composable normalizing-flow mixing with `CompositeMixingConfig`, FFT-based
  `CirculantDiagonalMixingConfig`, and explicit fixed feature permutations.
- Zensical documentation with executable examples, rendered PPL/Kalman/SMC
  tutorials, migration notes and a release checklist
- Stateful training with `FitState`/`FitInfo`, resumable optimizer/RNG state,
  optional EMA, callbacks, auxiliary metrics and explicit JIT IO callbacks
- Required, overridable event specifications for diffusion and flow matching;
  `DiffusionTransformer` with time and optional global-context conditioning
- Adaptive diagonal-noise SDE step doubling with weak and strong Brownian
  controllers; ODE inverse log-determinants with exact or Hutchinson traces
- Complementary inverse beta/gamma functions (`betainccinv`, `gammainccinv`)
  and stable log-space helpers (`log1mexp`, `logdiffexp`)
- Temporal inference over physical time: per-particle `StateSpaceModel`s (pytree
  latent states) or exact `LinearGaussianModel`s, `run_temporal_filter`, joint
  path sampling and particle Gibbs, and SMC² via `temporal_smc` with a default
  adaptive random-walk move; runs continue by passing the longer dataset
- BlackJAX-backed adaptation controls and shared inference runners
- ISAB key masks and per-example attention biases folded into query/key features
- `MultiHeadLatentAttention` (DeepSeek-style MLA): one `kv_lora_rank`-wide latent
  per token is the shared cache, with `mode="absorbed"` (`K = V = C`) or
  `mode="materialized"` (per-head `K`/`V` up-projections). One encoder can be
  shared across several layers via `kv_encoder=False` (the layer then treats its
  `inputs_kv` as the bank directly); absorbed mode
  folds the heads into the query axis so neither training nor inference
  broadcasts `K`/`V`, drops the value projection with `value_projection=False`,
  and takes an optional `query_scale_cls` (e.g. `QASSMaxQueryScale`) fed
  `kv_len`
- `Autoregressive.sample` supports prefix conditioning (`prefix=`, `prefix_len=`)
  and KV-cached decoding (`use_cache=`) for transformer conditioners, with the
  cached loop compiled to a single program via `nnx.scan`
- `TransformerARConditionerConfig` accepts an `attention_fn` kernel and works
  with discrete families
- `probjax.enable_rv_tracing()` context manager (plus
  `probjax.rv_tracing_enabled()` predicate) to opt sampling code into
  traceable `random_variable` sites outside the PPL transformations
- `probjax.disable_custom_inverse()` context manager (plus
  `probjax.custom_inverse_enabled()` predicate) to opt out of
  `custom_inverse` handling, e.g. for a buggy registered inverse: forward
  calls run the plain function with no `custom_inverse_call_p` primitive,
  and `inverse`/`inverse_and_logabsdet` fall back to structural inversion.
  Set `PROBJAX_DISABLE_CUSTOM_INVERSE=1` to disable globally
- `LinearOperator.solve(rhs, ...)` and `LinearOperator.logdet(...)`
  encapsulating the dense-vs-iterative (CG/Lanczos) dispatch previously
  hand-rolled by each consumer; `default_solve`/`default_logdet` in the
  Kalman filter are now thin wrappers with unchanged signatures
- `probjax.nn.utils.PrecisionMixin`: stores the
  dtype/precision/param_dtype/preferred_element_type quartet and exposes
  `cast_output(x)`; adopted by 22 NN module classes
- `fit(schedule=...)` also accepts any optax schedule; `FitMixin` gains a
  `_fit_batch` hook so a model decides what its batches look like
- `tests/test_layering.py` guards the package layering: `stats` and `core`
  never import flax or optax, and the flax-free training modules stay so
- MCMC convergence diagnostics in `probjax.inference.diagnostics`
  (`effective_sample_size`, `potential_scale_reduction`/`rhat`, `ess_bulk`,
  `ess_tail`, `pareto_khat`, and the HMC `divergence_concentration*`
  report), re-exported from BlackJAX 1.7 so future bumps track upstream
- Kernelized Stein discrepancy (`imq_kernel`, `kernelized_stein_discrepancy`
  with V- and U-statistics, PyTree-native inputs), vendored from unreleased
  BlackJAX main until the pin reaches a release containing it
- SMC numerics vendored from unreleased BlackJAX main (remove on the next
  BlackJAX bump): offset-stabilized `ess`/`log_ess` with a stopped-gradient
  max-subtraction, and boundary-safe systematic resampling that never selects
  zero-weight particles; all `ess` call sites and `systematic` defaults now
  use the probjax copies
- Shared helper modules for the patterns that were inlined in more than one
  place: `probjax.inference.particles` (`take_batch`, `normalize_log_weights`),
  `probjax.stats.spherical` (`normalize_vector`, `sphere_area`,
  `sample_sphere`, `sample_uniform_sphere`), plus helpers in
  `probjax.stats.utils`, `probjax.utils.jaxutils`, `probjax.nn.utils` and
  `probjax.inference.filtering.base`
- JAX 0.11.2 is now the version the test suite runs against; the supported
  range is unchanged (`jax >= 0.9.0, < 0.12`), and a fresh install resolves
  to the newest release in it

### Changed
- **Breaking:** `run_rdsmc` and `run_pretrained_diffusion_smc` are one kernel,
  `diffusion_smc`, with `score_guide` and `neural.diffusion_guide`; its
  results come from `run` (`state.log_evidence`, `trace`), not `SMCResult`.
  `diffusion_guide` takes the potential at full strength (scale it to temper)
- **Breaking:** Data tempering uses a vector temperature with a per-observation likelihood
  instead of `PartialPosteriorsPath`; geometric annealing from a reference
  uses `logprior_fn=reference`, `loglikelihood_fn=target - reference`
- **Breaking:** `ars` is a kernel: `ars(logdensity_fn).init(key, position,
  init_points=None, lower=-inf, upper=inf)`, then `run` collects exact iid
  draws while the envelope refines in the state. The old
  `ars(state, rng, num_samples, log_density_fn)` function is gone. Initial
  points on one side of the mode are now moved outwards automatically
- **Breaking:** `imh(logdensity_fn, proposal=density)` takes a `Density`; the
  kernel's params are the proposal's params, and its default adaptation trains
  them with `fit_adaptor` on Rao-Blackwellised chain output (`imh_data`). It
  replaces both the old `imh(proposal_fn=, proposal_logpdf=)` and
  `neural_imh(logdensity_fn, model)`: use
  `proposal, params = neural.density(model)` and
  `imh(logdensity_fn, proposal=proposal)`. IMH info is now `IMHInfo` with the
  proposed and previous positions
- **Breaking:** `flow_vi(logdensity_fn, family, optimizer)` takes a `Density`
  and returns a `Kernel` whose step is one optimizer step, so `run` drives it:
  `run(key, kernel, kernel.init(params), None, n, collect=("loss",))`. The
  per-step objective is `FlowVIInfo.loss` (was `elbo`, which held the negative
  ELBO) and `FlowVIState` holds `params` (was `flow_params`)
- **Breaking:** `neutra(logdensity_fn, bijection, params)` takes a map
  `(z, params) -> (x, log|det J|)`, e.g. from `neural.bijection(flow)`, instead
  of a flow
- **Breaking:** the `MCMC` runner class is replaced by `run`/`warmup`/`step`.
  `MCMC(kernel).sample(key, state, n, params, thin=t)` becomes
  `run(key, kernel, state, params, n * t, thin=t).trace["position"]`;
  `MCMC(kernel, collect_info=(...)).run(...)` becomes `run(..., collect=(...))`;
  `MCMC(kernel).warmup(key, policy, state, params, n)` and `.adapt(...)` become
  `warmup(key, kernel, state, params, n, policy)`. Warmup policies return a
  `RunResult` whose `trace` holds their diagnostics (formerly `info`)
- **Breaking:** the training options moved into a `FitConfig`, and the data
  contract is stated instead of guessed. `FitConfig` holds `max_steps`,
  `batch_size`, `learning_rate`, `schedule`, `clip_norm`, `ema_decay`,
  `use_ema`, `callback_every` and `optimizer`, plus a `DataSpec` that says how
  the data reaches the loss: `DataSpec(primary="data")` names the batch entry
  handed to the loss positionally (every other entry arrives as a keyword
  argument) and `DataSpec(stream=True)` says the source yields batches rather
  than being one. Both readings are pytrees of arrays, so the structure cannot
  tell them apart and guessing it silently trained a flow on a stacked array.
  A configuration overrides only the fields it names, so a model's own recipe
  in `default_config()` survives
- **Breaking:** the loss is called as `loss_fn(params, rng, primary, **context)`
  and may return `(loss, metrics)`. Both `fit` and `model.fit` return a
  `FitResult` carrying the trained `params`, the `losses`, any `metrics` and a
  resumable `state`; training is resumed with `state=result.state`
- **Breaking:** callbacks run on the host between compiled chunks and cannot be
  used inside JIT. The explicit `callback_mode="io"` path, the legacy
  `on_step`/`log_every` pair and the `on_step` history padding are gone
- **Breaking:** `num_steps="auto"` and `batch_size="auto"` are gone. The step
  count and the batch size are the caller's to choose, so the same call trains
  the same model on different data
- The training configuration owns the optimizer recipe. `FitConfig.optimizer`
  takes a built `optax.GradientTransformation`; without one, the configuration
  builds the default from its own `learning_rate`, `schedule` and `clip_norm`
  (`FitConfig.default_optimizer()` builds it explicitly, and the training loop
  expects a built optimizer rather than making one). The defaults are a
  learning rate of `1e-3` and a one-cycle cosine schedule
  (`schedule="onecycle_cosine"`); `schedule` also accepts any optax schedule
  or `None` for a constant rate
- A training stream is the caller's iterator and is not restarted: `fit` looks
  at the first batch to check the format and then takes batches one at a time.
  A stream that runs out raises instead of silently repeating, and a ragged
  later batch is no longer checked against the first one
- `odeint` and `sdeint` return one shape: the filtered trajectory
  (`collect_trace=True`) or the filtered terminal state. `sdeint` takes the
  auxiliary data (Brownian path, solver state) through a single `return_aux`
  flag that hands back a separate `SDEIntAux` record, replacing the
  `return_brownian`/`return_state` pair
- `step_size_adaptor` configures the adaptive integrators and is rejected for
  a fixed-step solver instead of being silently ignored. The adaptive SDE
  integrator is selected with `solver="adaptive"`
- One kernel API for every algorithm family. `probjax.utils.KernelAPI` is the
  shared base of the MCMC, SMC, filtering, ODE and SDE API classes, and
  `probjax.utils.Configured` is the value that binds an API's options before
  the model is known: `hmc.configure(num_integration_steps=10)` returns a
  hashable, comparable-by-value `Configured` that can be applied to the model
  later and passed to anything that expects a kernel API (`path_smc`,
  `adaptive_smc`, `pseudo_marginal`). Bound options are routed to whichever of
  `init`, `init_params` and `build_step` accepts them. `ConfiguredSolver` is
  now an alias of `Configured`
- Built kernels carry their algorithm's name and docstring everywhere, not just
  for MCMC and SMC: a built filter is a `FilterKernel` subclass named after the
  filter, and likewise for ODE and SDE solvers
- `build_step` is the canonical name for building a filter's step function;
  `build_kernel` still works on every filter class
- `Filter` shares the runners' progress machinery (`RunnerMixin`) with `MCMC`
  and `SMC`, and the temporal filter/SMC and training kernels are callable
  like every other kernel
- The packages below are now fully annotated and pass `pyright` cleanly:
  `probjax.utils`, `probjax.core`, `probjax.inference`, `probjax.stats` and
  `probjax.nn` (368 pyright errors before in the first four, 362 in
  `probjax.nn`, 0 after). The shared `PyTree` alias is no
  longer a recursive generic (which no type checker could resolve, so
  `PyTree[Array]` rejected every pytree at every call site); it now names
  the leaf type and keeps the containers permissive. `probjax.utils` also
  gains the `Scalar`, `ScalarLike`, `Shape`, `ArrayTree` and `ArrayLikeTree`
  aliases
- The normalizing-flow bijector configs (`ShiftBijectorConfig`,
  `AffineBijectorConfig`, the spline, monotone and CDF families) take their
  natural parameters positionally, matching the `BijectorConfigProtocol`
  contract they always implemented: `apply(x, loc)` still works, but
  `apply(x, loc=...)` is no longer accepted
- `probjax.utils.protocols` gains `mean_reduction`, the default reduction
  used by the loss builders; `probjax.nn.generative` gains the `SamplerLike`
  protocol every family's sampler implements
- **Breaking:** filtering internals re-layered to remove circular imports
  (no aliases): model types live in `filtering.state_space`, observation
  validation in `filtering.observations` (`validate_observations`), and shared
  Gaussian math in `filtering.gaussian` (`GaussianFilterState/Info`,
  `gaussian_unpack`, `innovation_loglik`, `kalman_update`,
  `gaussian_filter_template`, `default_solve`/`default_logdet`,
  `scan_history` lives on `filtering.temporal`). Update imports that reached
  into `filtering.temporal._observations`/`_scan` or
  `filtering.kalman_filter.{default_solve, default_logdet, _innovation_ll}`.
  `filtering` and `smc` package roots are now lazy (PEP 562) with unchanged
  symbol surfaces
- **Breaking:** inference helper renames (no aliases): `smc.base._filter_kwargs`
  is now `filter_kwargs`; `hmc._init_hmc_like_params`/`_scale_step_size_by_grad`
  are now `mcmc.base.init_hmc_like_params`/`scale_step_size_by_grad`; the
  compiled `MCMC.sample_kernel` scan now lives in `mcmc.base.sample_kernel`
  (re-exposed on the runner unchanged), so kernels no longer import the runner
- `odeutil.solvers.exponential.compute_phi_functions(A, dt, k)` is replaced by
  `probjax.utils.linear_operator.phi_matrices(A * dt, k)`
- `odeint(method=...)`/`sdeint(method=...)` are deprecated aliases of `solver=`
- Newton roots (and so implicit ODE steps) are differentiated with the implicit
  function theorem via `jax.lax.custom_root` instead of through the iterations
- MCMC and SMC kernels (`hmc`, `nuts`, `mala`, `adaptive_smc`, ...) are
  explicit `MarkovKernelAPI`/`SMCKernelAPI` subclasses with their own
  docstrings, so `help(hmc)`, IDE hover and the API reference describe each
  kernel and its options. Built kernels are instances of a per-algorithm
  `Kernel` subclass (`hmc.kernel_type`), so `help(kernel)` and `repr(kernel)`
  show the algorithm too. The internal `make_kernel_api`/`make_smc_api`
  factories are gone; subclass the API classes instead
- **Breaking:** `LinearOperator(fn, in_dim, out_dim)` is now
  `Function(fn, in_shape)`, `LinearOperator.from_array` is `Dense`, `.operator`
  is `.mv`, operators scale by scalars only, and `solve`/`logdet` take
  `solver=`/`psd=` instead of size-based `method=`/`dense_mem_limit=`
  (see the migration guide)
- Declare tailed histogram support as the real line so AR support validation
  accepts observations outside the central bin region.
- Standardize transformer/AR/flow-transformer defaults to width 256, eight 32D
  heads, four layers, and 4× FFN expansion, retaining explicit compact overrides.
- Correct induced/spatial attention's adaptive total QKV width so the head
  dimension is not divided by the head count twice.
- Use categorical linear/sigmoid continuous-rate defaults 0.1 to 20 and cosine
  rate cap 50, correcting weak terminal mixing with the sampling base prior.
- Apply learnable boundary-logit damping to AR spline distribution heads too.
- Publish configuration benchmark runners under `docs/scripts/` and measured
  tables/raw results under `docs/benchmarks/`.
- Gradient-based training moved from `probjax.stats.fit` to
  `probjax.nn.training` (`batches`, `kernel`, `loop`, `mixin`), so
  `probjax.stats` no longer depends on flax or optax. `probjax.nn` still
  exports `fit`, `FitMixin` and the `Fit*` types
- `FitMixin` no longer assumes a batch layout: by default `model.fit(rng, data)`
  trains on `data` as given and calls `loss(rng, batch)`. The
  `{"data", "context", "weights"}` convention now belongs to `GenerativeModel`
- Default spline boundary-logit scale to 0.1, damping learnable tail predictions
  while preserving identity initialization. Set `tail_logit_scale=1.0` for the
  previous undamped behavior.
- Prefer `condition`, `do`, and `log_joint_fn`; their legacy aliases and
  `trace(kind_labels="legacy")` now emit `DeprecationWarning`.
- Target Python 3.11–3.13 and JAX `>=0.9.0,<0.12`; require Flax >=0.12.6 and BlackJAX 1.7.1
- BlackJAX 1.6.2 → 1.7.1: compiled `lax.scan` loops (faster NUTS/HMC), the
  MCLMC `L=inf` fix for constant coordinates, and the isokinetic-integrator
  and slice stepping-out corrections arrive automatically through the
  delegated kernels; the native slice implementation already evaluates its
  slice test in the loop body so the `vmap` non-termination fix does not
  apply. `window_warmup` keeps using the `window_adaptation.base`
  init/update/final primitives (the bound `window_adaptation` /
  `staged_adaptation` replacements don't fit the `Warmup` protocol) and
  silences the upstream deprecation until the protocol migrates
- Migrate legacy Pallas features to `probjax.nn.pallas_kernels` and remove the
  old package; use an associative-scan fallback for multi-tile GPU Mamba
- Accept optional `event_spec` defaults for continuous diffusion and flow
  matching, with per-distribution overrides; mean-flow and categorical diffusion
  still require a constructor default
- Expose architecture builder overrides and shared training hooks on NN modules
- Keep runtime microbenchmarks opt-in; retain structural/correctness checks in CI
- Mean-flow training objective now defaults to the Improved MeanFlow (iMF)
  v-loss formulation (arXiv:2512.02012): the JVP tangent is the network's
  own boundary-condition velocity `u(z_t, t, t)` (marginal-velocity
  estimate, lower variance) instead of the conditional velocity, and the
  regression target no longer depends on the network. Same optimum, more
  stable optimization. The original objective remains available via
  `loss_kwargs={"imf": False}` or `model.loss(..., imf=False)`. Results
  will differ from previous versions
- Mean-flow `(t, r)` pair sampling (`SigmoidPairFlowTrainingConfig`):
  distinct pairs are now min/max of two independent logit-normal draws
  (matching the MeanFlow paper) instead of `r = clip(t + r_raw)`, which
  pinned ~70% of distinct pairs at exactly `r = 1` and starved interior
  small-gap pairs needed for few-step sampling. This changes the training
  distribution, so mean-flow results will differ from previous versions
- Renamed `AutoregressiveModel` to `Autoregressive` (hard rename, no alias)
- Sampling is now primitive-free by default: `Distribution.rvs`/`sample`
  no longer emit the `random_variable` primitive unless PPL tracing is
  enabled. The PPL transformations (`trace`, `joint_sample`,
  `log_joint_fn`/`log_potential_fn`/`log_prob_fn`, `intervene`/`do`,
  `condition`/`observe`, `substitute`) enable it automatically around their
  internal tracing, so probabilistic programs are unaffected, while plain
  `jit`/`vmap`/`scan` sampling skips the forward-jaxpr trace and the
  site-name counter. Raw `jax.make_jaxpr(model)` on sampling code no longer
  contains `random_variable` equations -- wrap it in the new
  `probjax.enable_rv_tracing()` context if you inspect jaxprs for sites or
  call `interpret` manually. Set `PROBJAX_RV_TRACING=1` to restore the legacy
  always-emit behaviour globally
- `LinearOperator` arithmetic now propagates dtypes truthfully via
  `jnp.result_type` (previously `__add__`/`__mul__`/`__neg__`/`__sub__`
  kept `self.dtype` verbatim, e.g. `(f32_op * 2.0)` reported float32 while
  materializing float64); construction infers dtype by probing the operator
  once, falling back to the global default when probing is impossible
- `LinearOperator` supports `2 * A`, `A / c`, `c - A`, and 0-d operands as
  `c * I` in `__add__`/`__sub__`; shape violations now raise `ValueError`
  at construction instead of deferred XLA errors or `AssertionError`
  (asserts are compiled out under `python -O`)
- `LinearOperator.T` is cached per instance instead of retracing
  `jax.linear_transpose` on every access
- `rv_continuous.support` is now a concrete default returning the
  whole-real-line constraint; the 13 whole-real-line distributions inherit
  it instead of repeating the body
- `symmetrize_matrix` transposes the last two axes instead of using `.T`, so
  a batch of matrices is symmetrized correctly (`A.T` reverses *all* axes of a
  3-D array)
- A one-component `ARFamily.mixture` is centred on the data: the component
  offsets are now computed from the shared helper, so a single component is
  not shifted by the spread
- `ensure_tuple` accepts any `numbers.Integral` (so NumPy integers work) and
  `ensure_int`/`ensure_tuple` take a `message=` override for callers whose
  error text mentions alternatives the check does not cover
- `nn.io_util.ensure_tuple` rejects an empty `spatial_shape` instead of
  silently accepting it
- `probjax.utils.functions.ravel_callable` is public (it was the private
  `_ravel_callable`); `probjax.utils.jaxutils.filter_kwargs` gained
  `exclude_positional_only`, and the `nn`-local copy of that filter now
  delegates to it
- `probjax.utils.jaxutils` exposes `pjit_p` (the `pjit` primitive, whose
  location moved between JAX releases) and `leading_dim` (the size of the
  leading axis of a pytree of arrays)
- Distribution per-method docstrings trimmed to one-liners (parameter docs
  stay on the class docstring)
- CI: tests marked `slow` are skipped on PR runs (full suite on `main`),
  the 3-Python version matrix runs on `main` only (PRs use 3.12), the
  coverage job runs on `main` only, and a persistent JAX compilation cache
  is shared across CI runs
- `pytest` runs affected tests by default via `pytest-testmon` (`--testmon` in
  `addopts`); the CI coverage job opts out with `--no-testmon`

### Removed
- The `SMC` runner class, `SMCResult`, `SMCKernelAPI`, `make_mcmc_adapter`
  and BlackJAX's SMC state types: SMC kernels run with `run`
- `adaptive_smc`, `path_smc`, `GeometricPath`, `PartialPosteriorsPath`,
  `waste_free_smc` and `waste_free_strategy` (all `smc` options now),
  `tuned_smc` (use an adaptor) and `pretuned_smc`
- `persistent_smc` and `adaptive_persistent_smc`
- `DenoisedPotentialTwist`, `NullTwist`, `TwistFn` and
  `is_score_and_marginal` from `probjax.nn` (the guides replace them),
  `RDSMCState`/`Info` and `PretrainedTwistState`/`Info`
- `RejectionSampler`, `ProposalDistribution` and `estimate_ratio_bound` (use
  `rejection`), and the envelope internals `ARSState`/`init_ars_state`/
  `update_ars_state` (now `rejection.univariate.Envelope`/`envelope`/`refine`)
- `neural_imh`, `neural_imh_warmup` (its `data=` seed data: fit the flow on
  the draws before `neural.density`), `NeuralIMHParams`/`NeuralIMHState`/
  `NeuralIMHInfo`/`NeuralIMHWarmupInfo`, `IMHParams`, `wrap_logpdf`,
  `proposal_gaussian`/`proposal_gaussian_logpdf` (use `gaussian_proposal`), and
  `probjax.inference.vi.rebuild` (the fitted params are `state.params`)
- `MCMC`, `MCMCResult`, `AdaptationResult`, `AdaptationTrace`, `WarmupResult`,
  `adapt`, `adapt_step`, `as_warmup` and `mcmc.base.sample_kernel`; use `run`,
  `warmup`, `step` and `RunResult`
- Compatibility aliases and dead code in `probjax.inference`: `MarkovKernel`
  (use `Kernel`), the empty `Params` type and the `Params`/`State`
  re-exports, `SGLDParams`/`SGHMCParams`/`SGNHTParams` (use `SGMCMCParams`),
  the unused `FilteringResult`, `Position` and `LogDensityFn`, the filters'
  `build_kernel` name (use `build_step`; `make_filter_api` takes
  `build_step_fn=`), and the pass-throughs `adaptive_smc_kernel`,
  `persistent_smc_kernel` and `adaptive_persistent_smc_kernel` (use
  `adaptive_smc`, whose `path` now defaults to `GeometricPath()`,
  `persistent_smc` and `adaptive_persistent_smc`)
- The unused `in_dim`/`out_dim` arguments of `make_linearized_transition` and
  `make_linearized_observation`, and `in_dim`/`materialize` of
  `make_continuous_transition`
- The custom-kernel extension point of the training loop: `FitKernel`,
  `build_fit_kernel` and `fit(..., kernel=)`. The loop owned the update and
  only had to reverse-engineer a custom kernel's contract at runtime; the
  optimizer is now chosen with `FitConfig.optimizer`
- `has_aux` and `model_state` as a fourth loss argument. A loss returns
  `(loss, metrics)`, and the module's non-parameter state travels through the
  loop in the state itself
- `return_result=True` (both entry points return a `FitResult`),
  `initial_state=` (use `state=`) and `is_batch_stream` (the source kind is
  stated in `DataSpec`)
- The unused, misnamed `linear_operator()` factory function (zero call
  sites repo-wide; pre-1.0 API cleanup)
- Dead ravel/flatten helpers in `probjax.utils.jaxutils`
  (`ravel_first_arg_`, `ravel_args_`, `flatten_args_`, `flatten_fun`,
  `ravel_fun`, `ravel_first_arg_fun`); the live `ravel_arg_fun` and public
  `ravel_args` are unchanged
- Dead code with no call sites: `probjax.core.registry.LazyJaxpr`,
  `probjax.stats.constraint_registry.generate_pdm` (and the registration it
  overwrote), the `rk43` solver and its `build_rk43_tablau` helper in
  `probjax.utils.odeutil.solvers.rk_explicit`, and the unused
  `probjax.utils.guards.ensure_array`/`as_concrete_bool`

### Fixed
- Reverse diffusion SMC weighted particles by the twist ratio alone, omitting
  the forward noising kernel over the proposal density; on a unit Gaussian
  target it halved the variance and biased `log Z` by about +0.5
- The pretrained-diffusion twist used the full potential at the denoised point
  from the first step, which collapsed the weights for sharp potentials
  (`log Z` off by several nats); it now fades in and steers the proposal
- `RejectionSampler.run` indexed with a boolean mask inside `while_loop` and
  so could never run under JAX
- ARS/ARMS envelope refinement dropped the support bounds `lb`/`ub` and
  computed segment masses with padding points, which could break
  sampling after a refinement
- `pseudo_marginal` inherited default adaptations that need a fixed target
  (MCLMC's) unbound, so they failed when called; it now has none for them
- `pathfinder_warmup` and `mclmc_warmup` passed the kernel to `jax.jit` as a
  traced argument, so they failed whenever they ran
- The default warmups of `mclmc`, `adjusted_mclmc` and `adjusted_mclmc_dynamic`
  never ran. **Breaking:** `mclmc_warmup(logdensity_fn, **options)` now tunes
  the kernel it runs with (so a custom integrator is respected) and loses its
  `mclmc_kernel` and `adjusted` arguments; the adjusted variants default to
  step-size dual averaging towards a 0.9 acceptance rate. Default-adaptation
  factories that take `logdensity_fn` receive the kernel's target
- `slice` accepted points with a NaN log density, kept the last *rejected*
  point when shrinking ran out of evaluations, and failed for an integer
  `step_size`. Its default adaptation targeted the total number of
  evaluations, which is not monotone in the step size and drove it to
  infinity; **breaking:** `slice_step_size_adaptor(target=0.5)` now balances
  bracket expansions and contractions (`SliceInfo` gains `num_expansions`)
- `latent_slice` carried the log density of a rejected point (or the slice
  level) when shrinking ran out of evaluations
- `gauss_rwmh`'s default warmup tuned the step size for the initial proposal
  scale and then replaced the scale, ending far below its 0.234 acceptance
  target; it now estimates the scale first and tunes the step size for it
- Two tests were sensitive to the platform's float arithmetic and failed on
  CI (x86_64) while passing locally (arm64), because a 600-step training run
  amplifies small differences and a single loss step or a single chain seed
  is not a stable quantity:
  - `test_reaches_reasonable_likelihood` compared the last and first loss of
    a training run, but the curve spikes by orders of magnitude mid-training
    (606, 643, 1.9e6) before converging. It now compares window medians of
    the curve's two ends, which are stable on every platform
  - `test_trained_density_is_normalised` required the total mass to be
    within 10% of one. The band is now 0.6-1.5, which still separates a
    dropped log-determinant (the regression it exists for) from a flow that
    trained slightly differently
  - `test_neutra_improves_mixing_on_the_funnel` asserted a 1.5x ESS ratio
    from a single chain seed. The ratio of a 4000-draw funnel chain ranges
    from 0.05 to 40 across seeds, and per-seed ratios are anti-correlated
    between JAX versions, so neither a seed nor an aggregate over seeds is
    reproducible. The funnel now asserts that the transformed chain still
    explores, and the efficiency claim moved to a correlated ridge, where it
    measures 6.4-10.0x over ten seeds on every supported JAX version
- Every method of `empirical_frozen` raised `TypeError`: its ten overrides
  passed the frozen `values`/`weights` to the base class *and* to the
  overridden method. The overrides are gone; the frozen distribution now
  inherits the base implementations
- `empirical.sample` raised `IndexError`: the requested number of samples was
  passed in `jax.random.categorical`'s `axis` slot instead of its `shape` slot
- `mixture.ppf` raised `AttributeError` (`jax.scipy.optimize.root` does not
  exist); it now bisects on the CDF, like `mixture_kernel` already did
- `norm.moment` raised `AttributeError` (`jnp.math.comb` never existed) and
  passed a list to `jnp.sum`
- The 1-D Wasserstein distance raised `AttributeError` (`jnp.trapz` was
  removed from JAX; it is now `jnp.trapezoid`)
- `t.sf`/`t.isf`/`t.logcdf`, `uniform.sf`/`uniform.isf` and `pareto.isf`
  raised `AttributeError`: those functions are not in `jax.scipy.stats`. The
  t-distribution has no CDF in probjax, so its `sf`/`isf`/`logcdf` now raise
  the same clear `NotImplementedError` as its `cdf`/`ppf`; the uniform and
  Pareto compute theirs from `cdf`/`ppf`
- `inverse(fun, static_argnums=1)` raised `TypeError` because
  `static_argnums` was sorted as if it were a sequence
- Iterative `linear_operator.solve` is structurally linear in the right-hand
  side again (needed to transpose it, e.g. inside `custom_root`)
- Newton root finding: the tolerance floor used `finfo.precision` (a digit
  count, 6 for float32), convergence flipped back and forth, and the second
  return value was the solution instead of a convergence flag
- Adams–Moulton/BDF correctors use Newton steps by default (the fixed-point
  default diverged on stiff problems) and no longer break under `jit` with `tol>0`
- `compute_phi_functions` returned `dt * exp(A dt)` instead of `phi_k(A dt)`,
  which made the general exponential Euler/midpoint/RK4 integrators inexact
- Exact linear ODE/SDE solvers use the matrix `phi_1` for dense drifts with a
  bias, include the drift in the diagonal-noise covariance, and drop an extra
  `dt` factor from the Van Loan covariance
- Explicit Runge-Kutta tableaux: `ralston3`, `ssprk3`, `ralston4` and `rk6` had
  wrong coefficients and only converged at first order; `rk6` is now Butcher's
  7-stage order-6 method. All registered fixed-step methods are tested against
  their nominal order
- `odeint(..., dtype=jnp.float64)` now builds the Runge-Kutta coefficients in
  that dtype (they stayed float32, flooring the error of `rk3`/`rk4` near 1e-8)
- Adaptive ODE solvers (`dopri5`, `tsit5`, ...) with `collect_trace=False` returned
  the internal solver state, which overshoots `ts[-1]`, instead of the solution
  at `ts[-1]`
- `sdeint(..., sde_type="stratonovich")` was silently treated as Ito for
  diagonal noise. `milstein` now implements it; solvers that cannot
  (Euler-Maruyama, SRI, exponential, adaptive stepping) raise
  `NotImplementedError` unless the noise is additive
- Adaptive SDE stepping rejected steps on their own noise, which biased the
  quadratic variation (and so the Ito correction) at every tolerance: the weak
  adaptor's variance collapsed as `rtol` tightened and the strong adaptor kept
  a constant ~15% bias. Steps are now always accepted and the error only sets
  the next step size
- `brownian_tree` reused one key for every node, giving wrong variances
  (e.g. `Var W(0.1) = 0.30`); each node now draws independent noise. The
  default `tree_depth` of `StrongStepSizeAdaptor` is now 20, and its step
  size is kept above 64 leaf widths
- Commutative-noise iterated integrals were missing a factor 1/2 (Milstein
  converged at order 0.5); Stratonovich is now supported for them too
- `method="srk"`, documented but never registered, is now an alias of `sri1`
- `tailed_histogram` reported the bounded support `[low, high]` of `histogram`, so
  autoregressive models with tails rejected out-of-range data
- The dropless grouped-matmul Pallas kernels called `pallas.triton.dot`, which no
  longer exists; they use `jnp.dot` (this broke them on CPU interpret mode)
- Run on JAX 0.11: batching rules no longer use the removed `batching.not_mapped`,
  Pallas kernels no longer use the removed `pl.dot`, `scan` inverses read the new
  `ft_in` layout, and the new `log2` primitive is invertible; JAX 0.9–0.11 are
  all tested
- Correct the classical RK4 third-stage tableau and stop the stage loop at its
  actual bound; add accuracy and parameter-gradient regressions
- VP/cosine preconditioning, time-rescaled SDE coefficients, stochastic reverse
  V-solver sampling and nonuniform exponential AB2 integration
- `LinearOperator` rectangular shapes, dense composition order, transpose dtype
  and JIT-tracer support; compiled Kalman operator/dense parity
- Upper-tail quantile cancellation, numerical edge cases across distributions,
  inverse propagation, filtering and SMC evidence/resume handling
- Background data-loader error propagation and deterministic prefetch regression
  checks; preserve coverage artifacts when the external Codecov upload fails
- `MeanFlowMatcher.__call__` no longer leaks `t`'s tangent into `r` through
  `jnp.clip(r, min=t)` during the training-time JVP (`stop_gradient` on the
  bound). Loss-neutral for `r == t` pairs (the corrupted term is multiplied
  by `(t - r) = 0`), but removes a latent trap for custom pair configs
- Transformer conditioner full-forward path is now causal: the explicit
  `mask=None` used to override the attention kernel's baked-in causal mask,
  so every position read the future and the density was invalid. Cached
  decoding, naive sampling, and training now agree
- `flex_attention` no longer crashes on stateful masks (their data children
  are not spatial dims and must not be padded)
- `pareto.support(b)` returned the whole-real-line constraint; now returns
  the `[b, ∞)` interval (also fixes `chi2.support` → `[loc, ∞)` and
  `truncnorm.support` → `[a, b]`, the same copy-paste bug). This repairs
  `biject_to` transforms and support-aware validation for these families
- `preferred_element_type=None` (the default) is now a true no-op in all NN
  modules via `PrecisionMixin.cast_output`: previously the unguarded
  `.astype(None)` silently corrupted dtypes on older JAX releases and raises
  `TypeError` on newer JAX (7 sites in conv/unet layers)
- SDE integration of plain-callable drifts now routes through
  `RaveledDrift`, so non-`Drift` pytrees keep array leaves as explicit
  differentiable pytree children on the SDE side, matching the ODE side

### Compatibility and validation

See [migration notes](docs/guides/migration.md) before upgrading. Adaptive ODEs
still use a custom reverse-mode adjoint; a gradient-mode selector and Jacobian
stride controls are not public APIs in this release. GPU performance of the
multi-tile Mamba fallback remains unverified. Beta shape gradients use finite
differences and beta inverse forward AD is unsupported. Codecov upload currently
requires repository setup independently of the passing coverage test suite.

## [0.1.1] - 2026-08-01

Published release preceding the changes above. See the tagged source for its
API and dependency versions.

## 0.1.0 development history - 2024

### Added
- Core functionality:
  - Automatic function inversion (`inverse`, `inverse_and_logabsdet`)
  - Function tracing (`trace`)
  - Log probability computation (`log_prob_fn`, `log_joint_fn`)
  - Interventions and conditioning (`intervene`, `observe`, `condition`)
  - Custom JAX primitives for probabilistic programming

- Distributions module:
  - Base classes: `rv_generic`, `rv_continuous`, `rv_discrete`, `rv_multivariate`, `rv_exponential_family`, `rv_spherical`
  - Continuous distributions: `norm`, `gamma`, `beta`, `uniform`, `expon`, `laplace`, `logistic`, `cauchy`, `chi2`, `t`, `pareto`, `skewnorm`, `truncnorm`, `gennorm`, `genpareto`, `vonmises`, `watson`, `bingham`, `wrapcauchy`, `dirichlet`, `multivariate_normal`
  - Discrete distributions: `bernoulli`, `binomial`, `categorical`, `poisson`, `geometric`, `dirac`, `empirical`
  - Higher-order distributions: `transformed`, `mixture`, `indep`
  - Bijective transforms for normalizing flows

- Neural networks module:
  - Architectures: `MLP`, `ResNet`, `Transformer`, `UNet`, `DeepSet`, `SSMModel`, `Sequential`
  - Normalizing flows: `NormalizingFlow`, `AffineCouplingFlow`, `AdditiveCouplingFlow`, `NeuralSplineFlow`, `NeuralAutoregressiveFlow`, `GaussianizationFlow`, `LinearFlow`, `maf`, `realnvp`, `nsf`, `naf`
  - Diffusion models: `DiffusionDenoiser`, `EDM`, `VP`, `VE`, `MultinomialDiffusion`, `FlowMatcher`, `MeanFlowMatcher`
  - Layers: `MultiHeadAttention`, `MaskedLinear`, `ResnetBlock`, `ConvBlock`, `LRUCell`, `MambaCell`, and more
  - Loss functions for flow matching, score matching, and denoising

- Inference module:
  - MCMC: `hmc`, `nuts`, `mala`, `mh`, `slice`, `elliptical_slice`, `latent_slice`, `mclmc`, `dynamic_hmc`, `arms`, `a2rms`, `pseudo_marginal`, `sgld`, `sghmc`, `sgnht`
  - MCMC runner with progress bar support
  - SMC: `smc`, `adaptive_smc`, `persistent_smc`, with geometric and partial posterior tempering
  - Filtering: `kalman_filter`, `extended_kalman_filter`, `ukf`, `particle_filter`, and smoothing algorithms
  - Rejection sampling: `ars` (Adaptive Rejection Sampling), `RejectionSampler`

- Utilities module:
  - ODE/SDE integration: `odeint`, `sdeint`
  - Linear algebra: Cholesky updates, matrix-vector operations
  - Interpolation: linear and polynomial interpolation
  - Graph utilities: JAX-compatible graph algorithms
  - Special functions: `betaincinv`, `digammainv`, `gammaincinv`
  - Solvers: Newton-Raphson root finding

### Features
- JAX-native implementation throughout
- JAX-native inference kernels and compiled runners
- Support for both CPU and GPU (CUDA and Metal)
- Type hints throughout
- Comprehensive example notebooks

### Dependencies
- JAX >= 0.4.34
- NumPy >= 2.0.0
- Flax >= 0.12.0
- Optax, Chex, Einops
- NetworkX, SymPy

[Unreleased]: https://github.com/mackelab/probjax/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/mackelab/probjax/releases/tag/v0.1.1
