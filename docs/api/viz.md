# Visualization (pyAMICA.viz)

Top-level, backend-agnostic plots for a fitted model's output
(`pyAMICA.numpy_impl.load.AmicaOutput`, as returned by
`pyAMICA.numpy_impl.load.loadmodout`). Unlike the legacy
`pyAMICA.numpy_impl.viz` module, these functions **return a `Figure`** and
accept an optional `ax`/`axes` to draw on, instead of returning `None` and
mutating pyplot's global state.

- **`plot_pmi_heatmap`** — a components-by-components pairwise mutual-information
  heatmap (see `pyAMICA.metrics.pairwise_mi`), reordered to cluster related
  components near the diagonal.
- **`plot_model_probability`** — for a multi-model fit, two stacked panels: each
  model's posterior probability over time, and the log-likelihood of the most
  probable model at each timepoint.
A per-component scalp-topography plot is not included yet: deriving source
activations from a loaded `AmicaOutput` depends on an unsettled `W` convention
question, tracked in [#159](https://github.com/sccn/pyAMICA/issues/159).

```python
from pyAMICA import plot_pmi_heatmap, plot_model_probability
```

::: pyAMICA.viz.plot_pmi_heatmap

::: pyAMICA.viz.plot_model_probability
