# MNE-Python compatibility (AMICAICA)

`AMICAICA` fits AMICA directly from an [MNE-Python](https://mne.tools)
`Raw`/`Epochs` and hands the result back through the standard MNE ICA surface.
It is **additive**: the scikit-learn-style [`AMICA`](amica.md) interface and the
byte-identical [EEGLAB output](../guides/eeglab.md) are unchanged; this is a
second entry point for MNE users, not a replacement.

MNE is an optional dependency, so `import pamica` never requires it. Install the
extra and import the wrapper explicitly:

```bash
pip install pamica[mne]
```

```python
import mne
from pamica.mne_compat import AMICAICA

raw = mne.io.read_raw_eeglab("subject.set", preload=True)

ica = AMICAICA(n_mix=3, random_state=42).fit(raw, picks="eeg", max_iter=100)

sources = ica.get_sources(raw)     # an mne.io.RawArray of component activations
maps = ica.get_components()        # scalp maps, shape (n_channels, n_components)
ica.plot_components()              # native mne.viz topographies

# Reconstruct with some components removed:
clean = ica.apply(raw.copy(), exclude=[0, 3])
```

`fit` accepts a `Raw` or `Epochs` (epochs are concatenated along time, as MNE's
own ICA does), any MNE `picks` selector, and forwards remaining keywords
(`max_iter`, `lrate`, `do_newton`, ...) to [`AMICA.fit`](amica.md).

## Interoperating with `mne.preprocessing.ICA`

`to_mne_ica()` returns a fully-populated
[`mne.preprocessing.ICA`](https://mne.tools/stable/generated/mne.preprocessing.ICA.html),
so the entire MNE ICA ecosystem (plotting, `find_bads_eog`/`_ecg`, exclusion
workflows) works on an AMICA decomposition:

```python
mne_ica = ica.to_mne_ica()
eog_idx, scores = mne_ica.find_bads_eog(raw)
mne_ica.plot_scores(scores)
```

The wrapper's `get_sources`, `apply`, `get_components` and `plot_components`
delegate to this object, so they reproduce `AMICA.transform` exactly: MNE
computes sources as `unmixing_matrix_ @ pca_components_ @ (X - pca_mean_)`, and
the export maps pamica's mean, symmetric-ZCA sphere and unmixing into those
matrices (writing the sphere as `V diag(1/√e) Vᵀ` with `V` orthonormal so MNE's
scalp maps come out in channel space). The equivalence
`to_mne_ica().get_sources(raw) == AMICA.transform(X)` is pinned by the test
suite on real sample EEG.

!!! note "Single-model in this release"
    `AMICAICA` covers the single-model case (`n_models == 1`). Multi-model
    exposure for MNE users (per-model sources and model dominance) is tracked in
    issue #141.

::: pamica.mne_compat.AMICAICA
