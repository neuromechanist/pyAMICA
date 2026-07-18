# Draft: "Applications" section (for a fuller paper, not the JOSS 1750-word draft)

Contributed by Scott Makeig (2026-07-14), suggested to go before or after "State of the Field"
in `paper.md`. Held here rather than merged into the JOSS paper: the JOSS draft is word-budget-
constrained and this section is scoped for a longer paper/preprint. Revisit when drafting that.

---

## Applications

The ever-advancing compute speed of desktop computing can facilitate routine applications of
advanced brain data modeling features available in AMICA:

1. **Source separation.** We recently have shown (Gwen et al., 202?) that the default max
   iterations (2000) proposed in AMICA is a convenience value -- AMICA source separation continues
   to increase slowly as the number of training iterations is increased. At 25 ms per iteration, a
   2000-iteration decomposition (e.g., of the 70-channel example EEG datasets used in this paper)
   should require less than an hour to compute, in many cases making available a larger compute
   horizon within which to further optimize source separation.

   An efficient measure of source separation performance is Mutual Information Reduction (MIR)
   introduced by Palmer in (Delorme et al., 2012).

   Scott's strong recommendation: add MIR as a built-in pamica option, applicable at decomposition
   end and/or optionally at specified decomposition waypoints. See the MIR/PMI port epic (tracked
   as a GitHub issue) for the implementation side of this.

2. **Brain dynamic instability** is a hallmark of human brain dynamics, both normal and
   pathologic -- yet source and source network instability is not yet commonly measured in M/EEG
   studies. In its multi-model mode, AMICA separates its training data into domains fit to
   different ICA models that compete for data points during training; this separation of the
   training data into source-model domains has been shown to be powerful for brain state
   monitoring, during sleep, quiet rest, and active task performance (Hsu et al., 201?). Accurate,
   data-driven segregation of unlabeled datasets into as many as 20 rest and active emotion
   imagination periods has also been demonstrated (Hsu et al., 201?). A plug-in EEGLAB toolbox for
   evaluating and plotting multi-model AMICA solutions is available (Ozgur...).

3. **Source and source network stability.** Artoni et al. (191?) have demonstrated and
   contributed RELICA, an EEGLAB plug-in for estimating the stability of sources returned by ICA.
   Akalin Acar & Makeig (202?) showed that the pattern of scalp projection variance of clusters of
   near-identical sources returned across bootstrap training data decompositions can reveal the
   nature of biological source network instability. Again, at 25 ms per training iteration,
   twenty-five 2000-iteration bootstrap decompositions of a given dataset can be computed in less
   than a day, allowing assays of source and source network instability as well as stability.

All these possibilities, once beyond the reach of routine desktop computing, are now practical to
apply using current desktop hardware. Using pamica in still more powerful compute environments can
only increase the depth of detail and statistical power of studies of brain dynamics in complex or
even real-life protocols -- either exploratory or confirmatory. For applications in supercomputer
environments, however, Palmer's FORTRAN version (AMICA 5.??) customized for use at the San Diego
Supercomputer Center (freely available via the Neuroscience Gateway (nsgportal.org; ????, 201?))
might prove still more efficient.

---

Notes for whoever drafts the fuller paper:
- All the "???"/"201?" citations above are as Scott sent them; need real citations before use.
- The 25 ms/iteration figure matches the MLX benchmark (`.context/issue-77/benchmark_findings.md`),
  not every backend -- qualify per-backend if reused.
- Ties into [[amica-parity-epic-status]] and the postAmicaUtility MIR/PMI port epic.
