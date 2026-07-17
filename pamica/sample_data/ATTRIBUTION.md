# Sample data and reference artifacts: provenance and licenses

The files in this directory are third-party artifacts bundled solely to validate
pamica against the reference AMICA implementation. They are used for testing and
documentation and are not part of the installable `pamica` package.

- **`amica15mac`** - the reference AMICA binary (macOS x86_64), compiled from the
  AMICA source distributed by the Swartz Center for Computational Neuroscience
  (SCCN), UC San Diego: <https://github.com/sccn/amica> (Palmer, Kreutz-Delgado,
  and Makeig). Redistributed here for parity testing only. See the SCCN AMICA
  repository for its license terms.

- **`loadmodout15.m`** - the AMICA output reader from the EEGLAB AMICA plugin
  (SCCN): <https://github.com/sccn/amica>. Used to document and check the EEGLAB
  round-trip. Its license is that of the EEGLAB AMICA plugin.

- **`eeglab_data.fdt`, `eeglab_data.set`** - the EEGLAB sample EEG dataset,
  distributed with EEGLAB (SCCN): <https://github.com/sccn/eeglab>. Used as real
  sample data for the tests and benchmarks.

- **`amicaout/`, `input.param`, `sample_params.json`, `pyresults/`** - reference
  AMICA output, parameter files, and derived results produced from the artifacts
  above for the validation harness.

If you redistribute or repackage these files, retain the attribution and comply
with the upstream SCCN/EEGLAB license terms.
