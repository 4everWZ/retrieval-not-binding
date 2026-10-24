# When Retrieval Is Not Binding

**A Diagnostic Benchmark for Visual Episodic Memory**

Weizheng Wang · Junyi Wu · Liqin Tian · Qing Tian

Research code for an ICASSP 2027 manuscript investigating the distinction between useful entity content and observation–event-position binding in visual memory.

## Abstract

Visual-memory retrieval can improve without preserving which observation belonged to which episode. We study this distinction through matched interventions that fix queries, candidates, visual features, and serialized memory envelopes while changing memory content or its assignment to event positions. A two-source development cohort contains 48 objects, 384 observations, and 192 cross-condition queries. Candidate coverage reaches 98.96%, with memory-off Recall@1 of 86.46%. On 128 queries with visible memory, four training-free readers gain one to three correct answers over memory-off and five to seven over wrong-entity memory. Reversing the content–position assignment leaves three readers unchanged and changes a recency-aware reader by one correct answer. A separate evaluator-only oracle changes from 96/96 to 0/96 under the intervention. These results distinguish useful entity content from binding-sensitive retrieval and motivate reporting direct binding controls alongside retrieval accuracy.

## 1. Research Question

Familiarity, entity retrieval, and event-conditioned observation recall require different information. This benchmark predicts entity identity and intervenes on observation–position pairs to measure whether entity retrieval depends on their association. Measuring event recall would require an event-valued prediction target.

The study contributes:

- A matched intervention protocol separating the benefit of memory content from dependence on its event-position assignment.
- Four interpretable, training-free readers evaluated with shared queries, candidates, features, and memory envelopes.
- Paired rescue/harm analysis, source-specific effects, and an evaluator-only mapping oracle.

## 2. Method

### 2.1 Matched Memory Interventions

Let $q$ be a normalized query, $g_c$ the representation of candidate entity $c$, and $H=\lbrace(e_j,t_j)\rbrace_{j=1}^{m}$ the visible history, where $e_j$ is an observation feature and $t_j$ its event position.

| Arm | Intervention | Diagnostic purpose |
|---|---|---|
| Correct (C) | Preserve the correct observation–position pairs | Reference condition |
| Memory-off (O) | Retain the envelope but bypass the memory contribution | Benefit of access to memory content |
| Wrong-entity (W) | Substitute another entity's history from the same source and condition | Benefit of correct entity content |
| Shuffled-binding (S) | Reverse the assignment of observations to positions while preserving their multiset | Dependence on content–position pairing |

Every arm uses the same top-eight candidates. Query state is reset between arms, and queries never write back to memory. Reversal is the identity for a single-observation history.

For memory-enabled arms, the fused query and prediction are

$$
\widetilde q=\mathrm{norm}\bigl(0.75q+0.25m(q,H)\bigr),
$$

$$
\widehat y=\arg\max_{c\in\mathcal C}\widetilde q^\top g_c.
$$

Memory-off uses $q$ directly. With Recall@1 denoted by $R_a$ for arm $a$, the paired contrasts are

$$
\Delta_{\mathrm{off}}=R_C-R_O,\quad
\Delta_{\mathrm{wrong}}=R_C-R_W,\quad
\Delta_{\mathrm{bind}}=R_C-R_S.
$$

The first two contrasts measure content benefit; the third directly measures dependence on observation–position pairing in this retrieval task.

### 2.2 Training-Free Readers

| Reader | Memory readout |
|---|---|
| Mean/prototype | Normalized mean of all visible observations |
| Reservoir | Select at most two exemplars without replacement using deterministic hash priorities and seed `20260806`, then normalize their mean |
| $k$-center | Start at the centroid medoid, then use cosine farthest-first selection, retaining at most two exemplars |
| ERM (fixed recency) | Aggregate observations with query-content and normalized-age weights |

The fixed-recency ERM uses $d_j=(10-t_j)/\max_k(10-t_k)$ and $a_j=\mathrm{softmax}_j(8q^\top e_j-0.5d_j)$, yielding $m(q,H)=\mathrm{norm}(\sum_j a_je_j)$. This is the simplified form of prototype-plus-episodic-residual aggregation under the fixed setting.

The first three readers ignore event positions during selection and aggregation, making them structurally invariant to the binding reversal. ERM's age term allows the assignment to affect its representation.

Implementation: [fixed readers](src/retrieval_not_binding/readers.py), [episodic residual memory](src/retrieval_not_binding/memory.py), and [benchmark configuration](configs/benchmark.json).

## 3. Experimental Setup

### 3.1 Cohort and Representation

| Source | Entities | Observations | Queries | Top-8 coverage | Memory-off correct |
|---|---:|---:|---:|---:|---:|
| Google Scanned Objects (GSO) | 24 | 192 | 96 | 94/96 | 90/96 |
| uCO3D | 24 | 192 | 96 | 96/96 | 76/96 |
| Overall | 48 | 384 | 192 | 190/192 | 166/192 |

Queries span view, occlusion, environment, and detail conditions, with different query and gallery conditions. Candidates are retrieved within a 24-entity source–condition stratum, and the top eight are fixed across interventions. There are 32 entities with visible memory, yielding 128 queries, and 16 pseudo-unseen entities, yielding 64 queries. Histories contain one to four observations at event positions 0–9; the query occurs at position 10. Entities and their derived observations remain in a single data role.

Condition-visibility and answerability checks passed for **32/32** reviewed items; a separate occlusion check passed for **16/16**. These checks assess visual conditions and task answerability.

A frozen SigLIP 2 Base NaFlex encoder and its official processor produce 768-dimensional global float32 features. Detail queries average five prescribed processor views before normalization; other conditions use a single view. Exact model revisions, preprocessing, and input identities must be bound to the original execution manifests. Code defaults alone do not establish the settings of a reported run.

### 3.2 Budget, Metrics, and Positive Control

Each application uses a five-slot memory envelope with a measured serialized size of **178 bytes**: **34,176 bytes per arm** over 192 applications. The shared visual feature bank is external to this envelope. Encoder, features, candidates, and evaluator are shared across arms.

In addition to Recall@1 and candidate coverage, the evaluation reports rescue (O incorrect, C correct), harm (O correct, C incorrect), and source-specific differences. Net benefit satisfies $n\Delta_{\mathrm{off}}=N_+-N_-$.

An evaluator-only oracle checks intervention sensitivity on the 96 queries with at least two observations. It accesses gold and the correct observation–position mapping after ordinary reader outputs are complete. The oracle is a protocol control and does not participate in reader ranking or method selection.

## 4. Results

Experimental settings and results in this README follow the manuscript's Sections 3–4 and Tables 1–3. The cohort table above corresponds to Table 1. All counts in the fixed-reader comparison below (Table 2) refer to the **128 seen-memory queries**. Differences are measured in correct-query counts. Reader settings were fixed before observing these outcomes.

**Correct-query counts (out of 128).**

| Reader | C | O | W | S |
|---|---:|---:|---:|---:|
| Mean/prototype | 114 | 113 | 109 | 114 |
| Reservoir | 114 | 113 | 109 | 114 |
| $k$-center | 115 | 113 | 109 | 115 |
| ERM (fixed recency) | 116 | 113 | 109 | 115 |

**Paired differences and source effects.** The final column gives the source-specific net benefit over memory-off.

| Reader | C−O | C−W | C−S | Rescue/harm | GSO / uCO3D |
|---|---:|---:|---:|---:|---:|
| Mean/prototype | +1 | +5 | 0 | 2/1 | −1/+2 |
| Reservoir | +1 | +5 | 0 | 2/1 | −1/+2 |
| $k$-center | +2 | +6 | 0 | 3/1 | −1/+3 |
| ERM (fixed recency) | +3 | +7 | +1 | 4/1 | −1/+4 |

Correct entity content benefits all four readers. Memory-off makes **15 errors** on these 128 queries; correct memory rescues **two to four** and harms **one** previously correct answer per reader. The three content-only readers remain unchanged under binding reversal, while fixed-recency ERM changes by one correct answer (**0.78 percentage points**). The mapping oracle achieves **96/96** under the correct assignment and **0/96** under reversal, establishing that the evaluator distinguishes the two mappings.

### 4.1 Additional Controls and Uncertainty

A complementary **profile-selected ERM evaluation** (Table 3) uses four entity-disjoint folds. For each reserved fold, it selects among four fixed profiles using the remaining folds, then evaluates the selected profile on the reserved fold. Its results are separate from the fixed-reader comparison above because the profile-selection procedure differs.

| Diagnostic | Value |
|---|---:|
| Rescue / harm | 4 / 1 |
| Correct minus off / wrong | +3 / +7 |
| Correct minus shuffled / raw multiset | 0 / 0 |
| Pseudo-unseen false-familiarity errors | 2 / 64 |
| Source-macro Δoff (percentage points) | 2.34 |
| 95% entity-bootstrap interval (percentage points) | [−0.78, 5.47] |

Count contrasts use **128 seen-memory queries**; false familiarity uses **64 pseudo-unseen queries**. Here a false-familiarity error occurs when consulting memory changes a correct answer to an incorrect one on a pseudo-unseen query. The raw-multiset arm adds a duplicate reference to the first episode; its zero contrast indicates no net accuracy benefit from deduplication under this evaluation.

The **2,000-replicate** bootstrap resamples entities within each source, retaining all queries of a sampled entity together. The source-macro interval includes zero and describes uncertainty in the profile-selected evaluation; it is not an interval for the fixed-reader binding contrasts.

### 4.2 Interpretation and Limitations

The evidence covers a small, two-source development cohort. Content gains differ by source: GSO net benefit is negative, while uCO3D net benefit is positive. Across all 192 queries, memory-off makes **26 errors**: two have targets missing from the fixed candidates, while the remaining 24 have their targets available. The fixed-recency ERM's single-answer binding difference indicates useful dependence on the tested assignment, with an entity-valued retrieval target.

Reversal probes one form of event-position dependence. Equal aggregate accuracy alone does not imply identical predictions or the absence of temporal information; for the three content-only readers, invariance additionally follows from their aggregation rules. Position-dependent score changes that preserve the top-ranked candidate are invisible to Recall@1. Supplied entity linkage isolates memory readout from upstream object association. Direct event-grounded questions and independent cohorts would be needed to evaluate episodic recall more broadly.

## 5. Code and Reproduction

### 5.1 Repository Structure

```text
configs/benchmark.json            Fixed manuscript settings and encoder revision
src/retrieval_not_binding/
  cohort.py                      Observation schedules and matched histories
  preprocessing.py               Context, detail views, and frame selection
  occlusion.py                   Foreground-aware occlusion
  encoder.py                     SigLIP 2 global feature adapter
  retrieval.py                   Fixed gallery ranking
  dataset.py                     Opaque reader inputs and separate gold labels
  protocol.py                    Typed inputs and 178-byte memory envelopes
  memory.py                      Episodic residual memory
  readers.py                     Four fixed training-free readers
  evaluation.py                  Paired controls, oracle, and profile selection
  statistics.py                  Source-stratified entity bootstrap
  io.py                          Feature, label, and ranking serialization
  cli.py                         Ranking and evaluation commands
tests/                           Focused synthetic and interface tests
pyproject.toml                   Package and optional dependencies
LICENSE                          MIT License
```

The package contains the manuscript's data preparation, readers, controls, and evaluation. Raw data, model weights, feature caches, and experiment outputs are supplied separately. The results above follow the manuscript's main text and Tables 1–3.

### 5.2 Local Contract Tests

The core requires **Python ≥ 3.12** and uses the standard library. Run commands from the repository root using a project-local `.venv`. To reuse an existing read-only Python/Conda provider:

```bash
/path/to/approved/provider/bin/python -m venv --system-site-packages .venv

PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Image preparation uses **Pillow 12.0.0** for the fixed occlusion renderer. The global encoder adapter accepts an already-loaded SigLIP 2 model and its official processor from a compatible PyTorch/Transformers environment. Optional dependencies are declared in `pyproject.toml`; reuse installed model libraries when preparing the project environment.

Tests cover reader invariance, fixed envelope size, profile selection, paired metrics, entity bootstrap, preprocessing, and the feature-only CLI. Encoder tests use synthetic tensors and mock model outputs. Tests do not download data or weights; synthetic results establish code contracts.

### 5.3 Preparing Inputs

[Data preparation](src/retrieval_not_binding/dataset.py) accepts the frozen entity selection, observation metadata, normalized global features, query-view feature IDs, and modifier-off enrollment feature IDs. `prepare_inputs(...)` constructs shared top-eight candidates, opaque identifiers, matched envelopes, and separate evaluator labels. Retain its 32-byte token nonce with the preparation metadata for exact replay.

Use [preprocessing](src/retrieval_not_binding/preprocessing.py), [foreground-aware occlusion](src/retrieval_not_binding/occlusion.py), and the [global encoder adapter](src/retrieval_not_binding/encoder.py) when preparing observations and features. Detail queries use the original view plus four prescribed overlapping crops. Entity selection, licensed source assets, reviewed observations, and model files must be supplied externally.

The JSON interfaces are defined in [io.py](src/retrieval_not_binding/io.py):

| Input | Required contents |
|---|---|
| `reader_input.json` | Schema `retrieval_not_binding.reader_input.v1`; `features`, `episode_features`, and `queries`. Features are 768-dimensional unit vectors. Each query contains opaque candidate references and the five memory applications, whose envelopes are encoded as hexadecimal strings. |
| `gold.json` | Schema `retrieval_not_binding.gold.v1`; `rows` containing query token, entity cluster token, gold candidate token, fold, source, pseudo-unseen status, query kind, and memory count. |

After calling `prepare_inputs(...)`, serialize its two return values separately:

```python
from dataclasses import asdict
from retrieval_not_binding.io import reader_input_dict, write_json

write_json("data/reader_input.json", reader_input_dict(reader_input))
write_json("data/gold.json", {
    "schema_version": "retrieval_not_binding.gold.v1",
    "rows": [asdict(row) for row in gold_rows],
})
```

### 5.4 Ranking and Evaluation

Run ranking without evaluator labels, then join the saved predictions with gold in a separate command:

```bash
mkdir -p results

PYTHONPATH=src .venv/bin/python -m retrieval_not_binding rank \
  --input data/reader_input.json \
  --output results/rankings.json

PYTHONPATH=src .venv/bin/python -m retrieval_not_binding evaluate \
  --input data/reader_input.json \
  --rankings results/rankings.json \
  --gold data/gold.json \
  --output results/report.json
```

The ranking command saves the four fixed-reader outputs and all four ERM profiles. Evaluation reports coverage, memory-off retrieval, paired fixed-reader contrasts, the mapping oracle, fold-specific profile selections, supplementary ERM controls, and the entity-bootstrap interval. The complete profile-selected evaluation requires the manuscript's 192-query, 48-entity cohort. Existing output files are preserved; choose fresh paths for another run.

`configs/benchmark.json` records the fixed reader and representation settings. Ranking files include this configuration, and evaluation requires the same configuration. Full empirical reproduction requires the frozen cohort, original feature inputs, preparation metadata, and compatible encoder environment in addition to this code.

## 6. Citation

Please cite this repository as research code. This entry does not indicate conference acceptance.

```bibtex
@misc{wang2026retrievalnotbinding,
  title  = {When Retrieval Is Not Binding: A Diagnostic Benchmark for Visual Episodic Memory},
  author = {Wang, Weizheng and Wu, Junyi and Tian, Liqin and Tian, Qing},
  year   = {2026},
  url    = {https://github.com/4everWZ/retrieval-not-binding},
  note   = {Research code}
}
```

## 7. License

Code is distributed under the [MIT License](LICENSE). Datasets and models remain subject to their publishers' licenses; the code license does not grant redistribution rights for those assets.
