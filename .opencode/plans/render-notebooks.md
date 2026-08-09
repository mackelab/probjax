# Plan: Render Example Notebooks on Documentation Website

## Problem
The current `docs/examples.md` only links to notebooks on GitHub. The user wants the notebooks rendered directly on the documentation website using `myst-nb`.

## User Decisions
- Use **category pages** (separate RST files per category)
- **Replace** `examples.md` with new tutorials section

## Solution
Create RST files that reference notebooks via toctree using relative paths (`../examples/...`). The `myst_nb` extension renders `.ipynb` files as HTML. Notebooks are NOT executed during build (`nb_execution_mode = "off"`), so pre-existing outputs are shown.

## Files to Create (8 files)

### 1. `docs/tutorials.rst`
Main tutorials landing page with toctree linking to all category pages.

### 2. `docs/tutorials/basics.rst` (1 notebook)
- `../examples/basics/particle_filter.ipynb`

### 3. `docs/tutorials/core.rst` (3 notebooks)
- `../examples/core/graph.ipynb`
- `../examples/core/ppl.ipynb`
- `../examples/core/trace_random.ipynb`

### 4. `docs/tutorials/inference.rst` (4 notebooks)
- `../examples/inference/filtering_smoothing.ipynb`
- `../examples/inference/kalman_filter.ipynb`
- `../examples/inference/mcmc.ipynb`
- `../examples/inference/smc.ipynb`

### 5. `docs/tutorials/nn.rst` (17 notebooks, grouped by subtopic)
**Flows:**
- `../examples/nn/flows/ar.ipynb`
- `../examples/nn/flows/flow_matching.ipynb`
- `../examples/nn/flows/flow_matching_mnist.ipynb`
- `../examples/nn/flows/mean_flow_matching.ipynb`
- `../examples/nn/flows/mean_flow_matching_mnist.ipynb`
- `../examples/nn/flows/normalizing_fows.ipynb`
- `../examples/nn/flows/normalizing_fows_mnist.ipynb`

**Transformers:**
- `../examples/nn/transformers/attention.ipynb`
- `../examples/nn/transformers/decoder_transformer.ipynb`
- `../examples/nn/transformers/flex_attention.ipynb`
- `../examples/nn/transformers/sharding.ipynb`
- `../examples/nn/transformers/simformer.ipynb`
- `../examples/nn/transformers/sorting_transformer.ipynb`
- `../examples/nn/transformers/transformers_binary_mnist.ipynb`
- `../examples/nn/transformers/transformers_mnist.ipynb`

**RNNs:**
- `../examples/nn/rnns/rnns.ipynb`
- `../examples/nn/rnns/sorting_lru.ipynb`

**Bayesian NNs:**
- `../examples/nn/bnns/bnn.ipynb`

### 6. `docs/tutorials/stats.rst` (1 notebook)
- `../examples/stats/distirbutions.ipynb`

### 7. `docs/tutorials/utils.rst` (5 notebooks)
- `../examples/utils/betaincinv.ipynb`
- `../examples/utils/bijective.ipynb`
- `../examples/utils/gammaincinv.ipynb`
- `../examples/utils/odeint.ipynb`
- `../examples/utils/sdeint.ipynb`

## Files to Modify (2 files)

### `docs/index.rst`
Replace `examples` with `tutorials` in toctree.

### `docs/examples.md`
Delete - replaced by tutorials section.

## Total: 32 notebooks rendered
