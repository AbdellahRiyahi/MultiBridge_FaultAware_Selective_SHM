# MultiBridge Fault-Aware Selective SHM

### Fault-aware selective prediction for bridge monitoring: an acquisition-level study on three bridges

**Abdellah Riyahi · Mohammed Mestari · Bouchra Bouihi**  
2IACS Laboratory, ENSET Mohammedia, Hassan II University of Casablanca, Mohammedia, Morocco

Research code and retained execution evidence for an acquisition-level study of **measurement quality, structural classification, abstention and instrumental verification** on Z24, Route345 and openLAB.

> **Main message:** a low error rate among accepted predictions is not sufficient evidence of useful monitoring. Report it together with class-specific coverage, corruption sensitivity and the availability of instrumental decisions. In the retained experiment, cross-fitting the training quality scores does **not** provide a uniform advantage across the three bridges.

| Item | Reference |
| :--- | :--- |
| Associated conference | [ICAM3E 2026](https://icam3e.sciencesconf.org/), 22–24 October 2026, Béni Mellal, Morocco |
| Associated manuscript | EPJ Web of Conferences proceedings submission; final bibliographic metadata pending |
| **Article DOI** | **`00000000` — placeholder; not an assigned DOI** |
| Implementation | `1.2.0` — `quality_crossfit` |
| Retained experiment | `runs_v12/V1_2_MAIN` — `full` mode |
| Recorded execution status | `COMPLETE_WITH_LIMITATIONS` |
| Evidence snapshot | [`bd383f2747bdc97974b8dab168029ed2a475cc0f`](https://github.com/AbdellahRiyahi/MultiBridge_FaultAware_Selective_SHM/tree/bd383f2747bdc97974b8dab168029ed2a475cc0f) |
| Interface | Jupyter notebook; no shell commands required inside the workflow |
| Corresponding author | [a.riyahi@enset-media.ac.ma](mailto:a.riyahi@enset-media.ac.ma) |

**Start here:** [Notebook](MultiBridge_FaultAware_Selective_SHM_V1_2.ipynb) · [Evidence summary](runs_v12/V1_2_MAIN/EVIDENCE_SUMMARY.json) · [Main results](runs_v12/V1_2_MAIN/tables/ARTICLE_MAIN_RESULTS.csv) · [Results archive](runs_v12/V1_2_MAIN/ARTICLE_RESULTS.zip) · [Audit archive](runs_v12/V1_2_MAIN/AUDIT_PACKAGE.zip)

## Contents

[1. Inspect the results](#1-inspect-the-results-without-retraining) · [2. Repository guide](#2-repository-guide) · [3. Data](#3-datasets-and-access) · [4. Method](#4-method-and-protocol) · [5. Findings](#5-selected-results-from-v1_2_main) · [6. Environment](#6-recorded-environment) · [7. Run the notebook](#7-running-the-notebook) · [8. Reproducibility](#8-reproducibility-and-integrity) · [9. Limitations](#9-scope-and-limitations) · [10. Citation](#10-citation) · [11. Rights and contact](#11-licensing-acknowledgements-and-contact)

## 1. Inspect the results without retraining

**For reading and checking the reported results, start with the exports—not `Run All`.** No source recordings, GPU, fitted-model loading or neural-network training are required for this route.

Download [ARTICLE_RESULTS.zip](runs_v12/V1_2_MAIN/ARTICLE_RESULTS.zip) using GitHub's file download control. The archive is approximately **23.7 MB** and contains the retained tables, predictions, figures, configuration and audit material. Downloading the complete repository also retrieves much larger cache and model files that are unnecessary for this task.

Place `ARTICLE_RESULTS.zip` beside a new, separate Jupyter notebook. This example reads the archive without extracting it or executing the research notebook. It requires `pandas` in addition to the Python standard library.

```python
from pathlib import Path
import json
import zipfile
import pandas as pd

archive = Path("ARTICLE_RESULTS.zip")
if not archive.is_file():
    raise FileNotFoundError(f"Place ARTICLE_RESULTS.zip here: {Path.cwd()}")

with zipfile.ZipFile(archive) as exported:
    evidence = json.loads(exported.read("EVIDENCE_SUMMARY.json"))
    results = pd.read_csv(exported.open("tables/ARTICLE_MAIN_RESULTS.csv"))

assert evidence["version"] == "1.2.0"
assert evidence["run_id"] == "V1_2_MAIN"
assert evidence["mode"] == "full"
assert evidence["variant"] == "quality_crossfit"
assert evidence["status"] == "COMPLETE_WITH_LIMITATIONS"
print({key: evidence[key] for key in ("version", "run_id", "variant", "status")})

model_names = {
    "Logistic": "Logistic regression",
    "RandomForest": "Random Forest",
    "ExtraTrees": "ExtraTrees",
    "HealthyMahalanobis": "Mahalanobis",
    "StandardNN": "StandardNN",
    "FaultAware_Control": "FA-IS",
    "FaultAware": "FA-CF",
}
primary = results.loc[
    results["model"].isin(model_names)
    & results["coverage_target"].sub(0.8).abs().lt(1e-12)
].copy()
assert len(primary) == 84  # 7 predictors × 3 bridges × 4 subsets

nominal = primary.loc[primary["subset"].eq("nominal")]
ba = nominal.pivot(
    index="model", columns="task", values="balanced_accuracy_forced_all"
).reindex(index=list(model_names), columns=["Z24", "Route345", "openLAB2025"])
print("\nNominal balanced accuracy before abstention (%):")
print((100 * ba).rename(index=model_names).round(2).to_string())
```

The table contains additional selectors and coverage targets. **Filter explicitly** rather than averaging every row. The primary comparison comprises seven predictors; the presence of twelve model labels in some exports does not mean twelve primary models were reported.

<details>
<summary><strong>Read-only integrity check for the 18 files listed in the results manifest</strong></summary>

This checks the internal manifest against the bytes stored in the archive. It does not load model objects or establish independent scientific validity. For provenance, obtain the archive from the pinned evidence snapshot above.

```python
from pathlib import Path
import hashlib
import json
import zipfile

archive = Path("ARTICLE_RESULTS.zip")
with zipfile.ZipFile(archive) as exported:
    evidence = json.loads(exported.read("EVIDENCE_SUMMARY.json"))
    entries = evidence["results_files"]
    if len(entries) != 18:
        raise ValueError("This is not the expected 18-entry V1_2_MAIN manifest.")
    for relative_path, expected in entries.items():
        content = exported.read(relative_path)
        if len(content) != expected["bytes"]:
            raise ValueError(f"Size mismatch: {relative_path}")
        if hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative_path}")
print(f"Verified {len(entries)} manifest entries; no models were executed.")
```

</details>

## 2. Repository guide

The following is a **selected view**, not a complete file listing. The reference snapshot predates this README.

```text
MultiBridge_FaultAware_Selective_SHM/
├── README.md
├── MultiBridge_FaultAware_Selective_SHM_V1_2.ipynb
├── runs_v12/
│   └── V1_2_MAIN/
│       ├── EVIDENCE_SUMMARY.json
│       ├── environment.json
│       ├── STUDY_LOCK.json
│       ├── RESULTS_FOR_PAPER.md
│       ├── REPORT_ARTICLE.html
│       ├── ARTICLE_RESULTS.zip
│       ├── AUDIT_PACKAGE.zip
│       ├── audit/
│       ├── tables/
│       ├── predictions/
│       ├── figures/
│       └── folds/
│           ├── Z24/
│           ├── Route345/
│           └── openLAB2025/
└── cache_v12/
    ├── features/
    ├── windows/
    └── native/             # Signal arrays and metadata; see data-rights notice below
```

| File or directory | Purpose |
| :--- | :--- |
| [EVIDENCE_SUMMARY.json](runs_v12/V1_2_MAIN/EVIDENCE_SUMMARY.json) | Run status, cohort counts, predictor provenance, validation gates, limitations and output hashes |
| [environment.json](runs_v12/V1_2_MAIN/environment.json) | Recorded package versions, configuration and execution environment |
| [STUDY_LOCK.json](runs_v12/V1_2_MAIN/STUDY_LOCK.json) | Configuration and implementation signature of the retained study |
| [ARTICLE_MAIN_RESULTS.csv](runs_v12/V1_2_MAIN/tables/ARTICLE_MAIN_RESULTS.csv) | Pooled outer-test classification and selection metrics |
| [CONFUSION_WITH_ABSTENTION.csv](runs_v12/V1_2_MAIN/tables/CONFUSION_WITH_ABSTENTION.csv) | Decision counts, retaining abstentions, at the primary operating point |
| [QUALITY_ACQUISITION_TEST.csv](runs_v12/V1_2_MAIN/tables/QUALITY_ACQUISITION_TEST.csv) | Instrumental availability and unconditional/conditional detection |
| [QUALITY_CALIBRATION_FEASIBILITY.csv](runs_v12/V1_2_MAIN/tables/QUALITY_CALIBRATION_FEASIBILITY.csv) | Per-fold support, threshold feasibility and unavailable decisions |
| [CROSSFIT_GROUP_AUDIT.csv](runs_v12/V1_2_MAIN/tables/CROSSFIT_GROUP_AUDIT.csv) | Acquisition-group exclusions in quality cross-fitting |
| [BASELINE_REUSE_AUDIT.csv](runs_v12/V1_2_MAIN/tables/BASELINE_REUSE_AUDIT.csv) | Provenance of reused comparator models |
| [PAIRED_BOOTSTRAP.csv](runs_v12/V1_2_MAIN/tables/PAIRED_BOOTSTRAP.csv) | Paired, conditional uncertainty estimates; inspect the subset and clustering fields |
| [predictions/](runs_v12/V1_2_MAIN/predictions/) | Out-of-fold event predictions and secondary-analysis outputs |
| [folds/](runs_v12/V1_2_MAIN/folds/) | Fold-specific splits, calibrators, diagnostics and separately deposited fitted-model files |
| [REPORT_ARTICLE.html](runs_v12/V1_2_MAIN/REPORT_ARTICLE.html) | Generated execution report; download and open locally to view the HTML report |

The CSV/JSON exports are the numerical reference. Manuscript or presentation figures may use different labels, typography and comparator subsets from the generated execution figures. A rendered plot is not a substitute for its underlying table.

## 3. Datasets and access

### Primary cohorts

These are the **selected study cohorts**, not the sizes of the complete provider archives.

| Cohort | Physical recordings | Reference / altered | Response panel | Native sampling rate |
| :--- | ---: | :---: | :--- | :---: |
| Z24 PDT/AVT | 144 | 18 / 126 | 5 acceleration channels; 3 physical sensor groups | 100 Hz |
| Route345, version 2 | 25 | 6 / 19 | 58 acceleration channels; 29 physical sensor groups | 128 Hz |
| openLAB 2025 | 12 | 6 / 6 | 2 geophone velocity channels; 2 physical sensor groups | Approximately 1000 Hz |

**Reference** and **altered** denote documented structural states within each bridge, not equivalent damage severities across bridges. **Nominal** and **corrupted** describe the absence or presence of an *added synthetic measurement corruption*. The two distinctions are independent.

| Source | Access and study-specific preparation |
| :--- | :--- |
| **Z24** | Obtain the data through the [KU Leuven Structural Mechanics benchmark page](https://bwk.kuleuven.be/bwm/z24). The study uses the ambient-vibration part of progressive-damage testing, not the long-term monitoring classification benchmark. Channels: `R1V`, `R2L`, `R2T`, `R2V`, `R3V`; states 01 and 08 are references; state 02 is excluded. Absolute acceleration units remain unverified in the local audit; no new SI amplitude conversion is claimed. |
| **Route345** | [Zenodo version 2, DOI 10.5281/zenodo.15497673](https://doi.org/10.5281/zenodo.15497673). Input: `Rt345Bridge.h5`; expected source MD5: `5c3e2aff20e1984fce15bd18f57f62eb`. Use stored `sample_rate = 128 Hz`, not internal acquisition rate 512 Hz. Static strain is not fused with dynamic acceleration. The documented historical exclusion of A14 applies to both axes and is not a confirmed hardware-fault label. |
| **openLAB 2025** | [Vibration-excitation release, DOI 10.5281/zenodo.18546379](https://doi.org/10.5281/zenodo.18546379). Retain the campaign log and native file structure. Responses: `21_1K21_WZ_V47` and `33_1K33_WZ_V86`, in m/s; shaker accelerations are not substitute structural responses. Selection uses random excitation, vehicle position `Achse_A40`, documented reference/cut states and recordings before the logged shaker replacement. |
| **openLAB 2024 — secondary only** | See [Monitoring data of the openLAB research bridge — Part 1: Reference condition](https://doi.org/10.1016/j.dib.2025.111624) and its linked data record. This separate, reference-only acceleration experiment is not pooled with 2025 velocity responses. The local manifest contains 526 compatible recordings; this is not a claim about the size of the published release. |

### Data-rights notice

Original recordings remain subject to their providers' access, citation and reuse conditions. **The Z24 provider permits non-commercial research subject to acknowledgement and citation requirements and states that the data may not be transferred to third parties.** Obtain authorised access directly from the provider.

The historical repository snapshot contains `cache_v12/native/*.npy` signal arrays and associated provenance metadata, including Z24-derived responses. These caches are **not merely model weights or summary features**, and their presence does not establish permission to redistribute the underlying measurements. Verify the applicable permissions before using or redistributing any data-derived cache. Do not treat this repository as an alternative source of authorisation for restricted measurements.

For inspecting the paper's results, use the CSV/JSON evidence package; the native-signal caches are unnecessary. Fresh execution still relies on source discovery, original metadata and source hashes, so caches alone do not constitute a complete portable input dataset.

## 4. Method and protocol

### Two outputs, not one combined verdict

```text
Native responses → grouped recordings → 20 s windows → validity masks
                                      │
                  ┌───────────────────┴────────────────────┐
                  │                                        │
         Structural features                       Quality features
         137 per channel                           16 per channel
                  │                                        │
           Channel encoder                      Separate ExtraTrees
                  │                              suspicion score q
                  │                                        │
        Reliability-weighted fusion ←───────────────────────┤
                  │                                        │
        Window probabilities                   Acquisition suspicion score
                  │                                        │
        Average within acquisition,               Instrumental threshold
        then across ensemble members              calibrated separately
                  │                                        │
        Sensor eligibility + entropy              CHECK / NO_CHECK /
                  │                               UNDETERMINED
        ALERT / NO_ALERT / ABSTAIN
```

The quality estimator is trained to recognise **injected channel–window corruptions**, not independently annotated real hardware failures. Its score is a suspicion score, not a calibrated probability of sensor failure. An instrumental `CHECK` does **not** directly command structural abstention.

### Signal preparation

Partitions are established before window generation and learned transformations. The retained analysis uses **20-second windows**, with 50% overlap for training and no overlap for development, calibration or test. Native sampling rates and amplitude scales are retained; there is no additional common time-domain bandpass or per-window signal standardisation.

Welch PSDs use 4-second Hann segments with 50% segment overlap. Spectral features are interpolated on 64 frequencies over 0.5–25 Hz. Spectral shape, band power, temporal descriptors and training-reference contrasts form the structural representation. Twelve direct quality descriptors precede short-gap repair; peer coherence is calculated afterwards when the channel metadata support it. Unavailable coherence is not interpreted as physically zero coherence.

Short internal gaps may be interpolated only within the configured missing-fraction and duration limits. Masks remain explicit. A structural decision requires **at least two usable physical sensor groups in every included window**. Reference profiles and learned transformations use training groups, with the quality transformations refitted inside each cross-fitting split.

### Predictors and cross-fitting

| Export label | Manuscript label | Role |
| :--- | :--- | :--- |
| `Logistic` | Logistic regression | Conventional structural comparator |
| `RandomForest` | Random Forest | Conventional structural comparator |
| `ExtraTrees` | ExtraTrees | Conventional structural comparator; distinct from the separately fitted quality estimator |
| `HealthyMahalanobis` | Mahalanobis | Reference-fitted distance comparator |
| `StandardNN` | StandardNN | Validity-masked neural fusion without quality weighting |
| `FaultAware_Control` | **FA-IS** | Fault-aware structural training with in-sample quality scores |
| `FaultAware` | **FA-CF** | Fault-aware structural training with acquisition-group out-of-fold quality scores |

For FA-CF, every training acquisition is scored by an inner quality estimator whose fitting and learned quality preprocessing exclude that acquisition and all its variants. The two variants subsequently share the **same final quality estimator** for development, calibration and test. Cross-fitting changes the quality inputs used to train the structural network; it does not automatically improve or replace the final instrumental detector.

For a channel with validity mask $v$ and suspicion score $q$, the implemented fusion weight is

$$
r = v\,[0.05 + 0.95(1-q)].
$$

The 0.05 floor applies **before weight normalisation**; it is not a guaranteed 5% contribution to the fused representation. Weighted latent means and standard deviations feed the structural head. The auxiliary loss acts on the quality-unweighted **structural** branch, which still uses validity masks. The separate quality estimator receives no gradients from that loss.

### Evaluation and calibration

Three grouped outer folds test each retained acquisition once. Structural models are fitted separately for each bridge. Five corruption families—missing blocks, freezing, gain, clipping and noise—are used in fitting/development/calibration; **drift is excluded from these stages and added at test**. Severities are `0.25`, `0.60` and `1.00`, giving 18 corrupted test events plus the nominal event per recording. These are repeated conditions of the same physical recording, not 19 independent experiments.

| Quantity | Meaning in the primary experiment |
| :--- | :--- |
| Structural threshold | `0.5`, applied to the acquisition-level score |
| Structural acceptance | Sufficient usable sensors **and** entropy at or below a fold-calibrated threshold |
| Main coverage target | `0.80` on weighted calibration events; not a guaranteed test coverage or accuracy |
| Instrumental nominal-alert budget | `0.05`, empirical and applied separately to acquisition-level quality calibration |
| `NO_CHECK` | An active instrumental rule does not trigger; not certified sensor health |
| `UNDETERMINED` | No useful admissible instrumental threshold or insufficient support; not a negative diagnosis |

Probability recalibration remained at the identity because each fold lacked the required classwise calibration support. Consequently, threshold selection should not be described as evidence that the predicted probabilities are calibrated.

<details>
<summary><strong>Realised acquisition partitions: reference / altered</strong></summary>

| Bridge / fold | Training | Development | Calibration | Test |
| :--- | :---: | :---: | :---: | :---: |
| Z24, each fold | 8 / 56 | 2 / 14 | 2 / 14 | 6 / 42 |
| Route345, fold 1 | 2 / 8 | 1 / 2 | 1 / 2 | 2 / 7 |
| Route345, folds 2–3 | 2 / 9 | 1 / 2 | 1 / 2 | 2 / 6 |
| openLAB 2025, each fold | 2 / 2 | 1 / 1 | 1 / 1 | 2 / 2 |

Use the saved splits for exact assignments. Generic configuration fractions are not substitutes for these realised counts. Separation within folds does not eliminate dependence between recordings from related states or campaigns.

</details>

## 5. Selected results from `V1_2_MAIN`

The following values come from the retained exports. They are **not new experiments**, and they should not be interpreted as performance guarantees for operational monitoring.

### Nominal balanced accuracy before abstention

All entries are percentages. The denominators are 144 Z24, 25 Route345 and 12 openLAB recordings.

| Predictor | Z24 (%) | Route345 (%) | openLAB 2025 (%) |
| :--- | ---: | ---: | ---: |
| Logistic regression | 90.87 | 66.67 | 83.33 |
| Random Forest | 72.22 | 58.33 | 75.00 |
| ExtraTrees | 75.00 | 75.00 | 75.00 |
| Mahalanobis | 50.00 | 50.00 | 50.00 |
| StandardNN | 82.94 | 61.40 | 83.33 |
| FA-IS | 85.32 | 61.40 | 83.33 |
| FA-CF | 85.32 | 61.40 | 83.33 |

FA-IS and FA-CF have identical nominal binary predictions, although their scores and accepted populations differ. On Z24, both have zero errors among accepted nominal decisions, but FA-IS accepts **6/18 references** and FA-CF **4/18**, while both accept **109/126 altered recordings**. Zero accepted-case error therefore conceals a substantial class-specific coverage imbalance. The Mahalanobis result reflects its implemented operating point, not a general failure of distance-based methods.

### All six corruption families at the 80% calibration target

Coverage is the accepted fraction of all test events in the stated subset. Selective risk is the error fraction **among accepted events only**. All table values are percentages; each model may accept a different population.

| Bridge | Model | Total coverage (%) | Reference coverage (%) | Altered coverage (%) | Selective risk (%) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Z24 | FA-IS | 78.55 | 39.81 | 84.08 | 0.54 |
| Z24 | FA-CF | 77.01 | 29.32 | 83.82 | 0.25 |
| Route345 | FA-IS | 60.00 | 50.00 | 63.16 | 6.67 |
| Route345 | FA-CF | 64.44 | 63.89 | 64.62 | 11.38 |
| openLAB 2025 | FA-IS | 56.94 | 44.44 | 69.44 | 2.44 |
| openLAB 2025 | FA-CF | 58.33 | 47.22 | 69.44 | 2.38 |

Corrupted-event totals are **2,592**, **450** and **216**, derived respectively from 144, 25 and 12 physical recordings. On Z24, lower accepted-case risk comes with lower reference coverage and more errors before abstention. On Route345, both coverage and selective risk increase. There is no uniform FA-CF advantage.

### Separate instrumental outcomes

| Bridge | Availability (%) | Nominal checks (%) | Non-drift detection (%) | Drift detection (%) |
| :--- | ---: | ---: | ---: | ---: |
| Z24 | 100.00 | 4.86 | 67.78 | 22.92 |
| Route345 | 68.00 | 4.00 | 42.40 | 25.33 |
| openLAB 2025 | 100.00 | 8.33 | 82.22 | 5.56 |

Availability uses all nominal and corrupted requests. Nominal checks use all nominal recordings. Detection uses all corrupted events in the corresponding subset, **including `UNDETERMINED` events in the denominator**. These are requests to inspect measurements, not verified hardware-fault diagnoses or confirmed fault localisations.

Route345 fold 2 has no useful instrumental threshold under the empirical 5% budget. Its eight test acquisitions, or 152 acquisition-events, receive `UNDETERMINED`. For non-drift corruptions, detection is **159/375 = 42.40% overall**, versus **159/255 = 62.35% among available decisions**. Reporting only the second number would hide unavailable verification.

### Reading the exports correctly

| Export subset | Interpretation |
| :--- | :--- |
| `nominal` | No added measurement corruption |
| `seen_faults` | The five non-drift families used during fitting/calibration; not the same physical test recordings |
| `unseen_fault` | Drift, excluded from fitting, development and calibration |
| `all_faults` | All six corruption families |

CSV rates are generally stored as **fractions**; multiply by 100 for percentages. `balanced_accuracy_forced_all`, overall accuracy, selective risk and instrumental detection are different quantities. Undefined accepted-case metrics remain undefined when their denominators or classes are missing. Do not replace them by zero.

Risk–coverage rankings accept complete tied-score blocks and respect physical eligibility. They are descriptive test rankings, not newly calibrated deployment rules. The openLAB non-drift coverage ceiling of 80% arises because 36 of 180 events fail the two-sensor requirement; it is not proof that a calibration target was successfully reached.

Secondary outputs include quality transfer between bridges, openLAB 2024 reference-only monitoring, and Route345 maximum-versus-temporal-quantile aggregation. They do not establish cross-bridge structural transfer, damage detection on openLAB 2024 or a continuous hourly alarm rate.

## 6. Recorded environment

The retained experiment ran through Jupyter on **HPC MARWAN** using **CPU**, with four configured CPU threads. GPU support is implemented, but the recorded results are not GPU benchmark results.

Versions below are taken from [environment.json](runs_v12/V1_2_MAIN/environment.json), not inferred from the current latest packages.

| Component | Recorded version |
| :--- | :--- |
| Python | 3.11.7 |
| NumPy | 1.23.5 |
| pandas | 2.1.4 |
| SciPy | 1.11.4 |
| scikit-learn | 1.2.2 |
| PyTorch | 2.7.1+cu118 |
| h5py | 3.14.0 |
| threadpoolctl | 2.2.0 |

The notebook also imports `joblib`, `matplotlib` and IPython/Jupyter components; their exact versions are not recorded in that manifest. This is therefore an **environment record, not a complete dependency lockfile**. The reference snapshot does not contain a packaged installer or a standalone `requirements.txt`.

Use an isolated environment with a Jupyter kernel. For loading retained scikit-learn models, match the original environment rather than assuming cross-version compatibility. CUDA-labelled PyTorch can still execute on CPU; successful CUDA detection and a working device test are required for the automatic GPU path. A different hardware or package environment is not guaranteed to reproduce identical floating-point results.

**Security:** inspect downloaded notebooks before execution. Pickle-based formats, including `joblib` and some PyTorch checkpoints, must be loaded only from trusted sources. A checksum helps identify bytes; it does not by itself make deserialisation safe. See the [scikit-learn persistence documentation](https://scikit-learn.org/stable/model_persistence.html).

## 7. Running the notebook

### Choose the correct task

| Goal | Route | Original measurements required? |
| :--- | :--- | :---: |
| Read retained metrics and check export hashes | Section 1; do not run the research notebook | No |
| Exercise software logic on synthetic fixtures | `mode="selftest"` | No |
| Audit real-data paths, metadata and selection gates | `mode="audit"` | Yes |
| Perform a reduced execution check | `mode="smoke"` | Yes |
| Execute the research pipeline | `mode="full"` | Yes |

`audit`, `smoke` and `selftest` outputs are **not substitutes for the retained full-study results**. Self-tests use synthetic fixtures and cannot validate scientific performance on the three bridges.

### Configure before executing

Open a **working copy** of `MultiBridge_FaultAware_Selective_SHM_V1_2.ipynb`. Its first code cell defines `Config`, creates `CFG = Config()`, imports dependencies and derives output paths.

**Edit the configuration before this first cell is executed.** Do not change `CFG.project_root` or `CFG.study_name` in a later cell: `RUN`, `CACHE` and related paths have already been derived. Use a fresh output directory rather than writing into the downloaded `V1_2_MAIN` evidence.

For an isolated synthetic check, replace the original `CFG = Config()` line in the first code cell with:

```python
CFG = Config(
    project_root=str(Path.home() / "multibridge_selftest"),
    study_name="LOCAL_CHECK",
    mode="selftest",
    variant="quality_crossfit",
    reference_policy="never",
    reference_run_root="",
    reference_cache_root="",
    trust_own_local_reference_models=False,
    reuse_verified_native_cache=False,
    device_preference="cpu",
    cpu_threads=2,
)
```

`Path` and `Config` are defined earlier in that cell. This configuration prevents accidental reuse of local predecessor model files. The notebook creates its synthetic fixtures in self-test mode. Select **Restart Kernel and Run All Cells**, then read the software checks and final evidence status rather than assuming completion from the last visible progress message.

### Real-data execution on MARWAN or another authorised host

For a fresh run, configure these fields in the first code cell:

| Field | What to supply |
| :--- | :--- |
| `project_root` | A writable, separate experiment directory |
| `study_name` | A new identifier, for example `REPRO_V1_2`; preserve the original `V1_2_MAIN` |
| `z24_root` | Authorised Z24 progressive-damage archive with native state/setup structure |
| `route345_file` | The verified version-2 `Rt345Bridge.h5` file |
| `openlab_2025_root` | Native 2025 vibration-excitation files **and campaign log** |
| `openlab_2024_root` | Separate reference archive when `run_reference_2024=True` |
| `mode` | Start with `audit`, then `smoke`; use `full` for the intended experiment |
| `device_preference` | `auto` for tested CUDA with CPU fallback, or `cpu` |
| `cpu_threads` | Within the compute allocation granted to the Jupyter session |
| `reference_policy` | See the provenance distinction below |

Use a compute session authorised by your HPC provider and respect its resource limits. The notebook does not require Bash or the presence of `SLURM_JOB_ID`, but that is **not permission to run expensive jobs on a shared login node**. If CUDA is unavailable, a CPU path exists; no GPU speed-up is claimed for the retained execution.

The original configuration enables the secondary quality-transfer and openLAB 2024 tasks. Disabling them is permissible for a clearly labelled reduced run, but no longer reproduces all retained analyses. Similarly, keep `enable_minirocket=False` for the reported seven-predictor comparison.

### Reuse versus retraining

The retained run reused **six verified V1.1 comparator configurations** and newly trained **27 FA-CF ensemble members**: three seeds in each of nine bridge/fold combinations. It did not train every comparator afresh in that session.

| `reference_policy` | Behaviour |
| :--- | :--- |
| `prefer_compatible` | Reuse a compatible, verified V1.1 bundle when available; otherwise explicitly refit if the bundle is missing. Incompatible existing bundles can raise errors rather than silently refit. |
| `require_compatible` | Require the complete compatible V1.1 bundle; stop if required files are unavailable. |
| `never` | Do not reuse V1.1 models; perform a separately identified refitting experiment. |

The root of the reference snapshot contains `runs_v12`, not the original `runs_v11` study. Deposited V1.2 fitted models document the final predictors but should **not** be relabelled as the V1.1 predecessor to bypass compatibility checks. Reproducing the original reuse path requires the corresponding verified predecessor metadata and artefacts.

For a fresh installation without that predecessor, use a **new study** with `reference_policy="never"` and record the refitting provenance. Such a run is a reproduction attempt, not a claim that the historical fitted predictors or all numerical results will be bitwise identical.

`resume=True` is intended for compatible checkpoints in the same study. If configuration or inputs change, use a new study rather than deleting lock files to force reuse.

## 8. Reproducibility and integrity

There are three different levels of reproduction:

**Export inspection** checks reported values and decisions from saved CSV/JSON files. **Predictor-level reproduction** additionally requires trusted fitted artefacts, compatible dependencies, feature transformations and appropriate inputs. **End-to-end reproduction** also requires authorised source recordings, original metadata, the documented training procedure and either compatible predecessor artefacts or an explicitly recorded refit.

The results and audit ZIP files are evidence exports. Serialised fitted models are deposited separately under `folds/`; the ZIP files alone are not a full raw-data-to-prediction execution environment.

| Identifier | Value |
| :--- | :--- |
| Evidence commit | `bd383f2747bdc97974b8dab168029ed2a475cc0f` |
| Recorded implementation fingerprint | `d1bdf0edaca80030db4e397f4be56e2bbd02fba1cb538ebd417b785897ed8505` |
| Recorded study signature | `76d5aa8dbc67b01f07bfd911a69d7671fe32a83111c968d9198422c927abdccf` |

These identifiers have different meanings. The implementation fingerprint is a notebook/run provenance field; it is **not** the raw SHA-256 checksum of the `.ipynb` file. Notebook outputs can change its file hash without changing its code cells. When citing results, identify the run and the evidence commit, not an unspecified future state of `main`.

Keep additional analyses in a new, documented location. In particular, this README does not claim that reviewer-requested sensitivity experiments were performed merely because they were proposed. The saved primary study retains the 0.05 reliability floor and the 5% instrumental budget.

<details>
<summary><strong>Common pitfalls</strong></summary>

| Symptom | Interpretation / action |
| :--- | :--- |
| `COMPLETE_WITH_LIMITATIONS` | Inspect the limitations and calibration records. Completion does not imply every decision is available or operational validity has been demonstrated. |
| `NO_USEFUL_SENSOR_ALARM_THRESHOLD` | A scientific calibration outcome; retain `UNDETERMINED`. Do not silently substitute a threshold or count it as `NO_CHECK`. |
| Missing original source paths | Update configuration and obtain authorised data; downloading caches does not reconstruct all source metadata. |
| V1.1 compatibility or version error | Restore the verified predecessor environment, or explicitly refit in a new study with `reference_policy="never"`. |
| Study-lock or checksum mismatch | Investigate the changed input/configuration. Do not overwrite the retained evidence or remove safeguards to force acceptance. |
| Notebook preview unavailable on GitHub | Download the `.ipynb` and inspect it in Jupyter; preview failure is not proof that the file is invalid. |
| A path mentions `ICARC2027` | Historical local directory naming; the associated conference contribution is ICAM3E 2026. Replace local paths as appropriate, not the scientific configuration. |
| A table appears to have too many models or rows | Filter primary model labels, subset and `coverage_target=0.8`; several diagnostic selectors and coverage targets share the exports. |

</details>

## 9. Scope and limitations

The experiment is an **acquisition-level comparison on related benchmark recordings**, not an operational bridge-safety system.

- Synthetic window corruptions are not independently confirmed physical sensor failures. Perturbations are applied after the providers' native processing and do not simulate every aspect of a continuous hardware failure.
- Recordings may share structural states, campaigns and excitation conditions. Holding out recordings is not equivalent to holding out complete states, campaigns or bridges. openLAB reference and altered recordings also differ in day and experimental context.
- Calibration sets are small. They contain 16, 3 and 2 nominal acquisitions per fold for Z24, Route345 and openLAB respectively; one nominal request represents 6.25%, 33.33% or 50%. The empirical 5% constraint is not a population false-alarm guarantee.
- Earlier benchmark test results informed pipeline revisions. The saved status explicitly records `REVISED_ANALYSIS_ON_PREVIOUSLY_INSPECTED_BENCHMARK`; this is not a new confirmatory evaluation on untouched tests.
- Bootstrap intervals are conditional on the fitted models and observed recordings. They do not remove shared-campaign dependence, overlapping outer-fold training sets or uncertainty from refitting.
- The lack of a uniform FA-CF gain concerns this implementation and these cohorts. It is not evidence against cross-fitting in general, nor proof of equivalence between variants.

The study does not establish remaining useful life, an optimised maintenance policy, a complete digital twin, structural safety certification or generalisation of a structural classifier to a previously unseen bridge.

## 10. Citation

Please cite the associated paper when its final reference is available, and identify the evidence commit when using the retained implementation or numerical exports. Cite the original dataset providers separately when using their measurements.

**Provisional paper reference**

> Riyahi, A., Mestari, M., and Bouihi, B. *Fault-aware selective prediction for bridge monitoring: an acquisition-level study on three bridges*. ICAM3E 2026 contribution; associated EPJ Web of Conferences manuscript. DOI pending.

The following BibTeX is deliberately a **provisional `@misc` entry**, not a claim that volume, pages or an article number have already been assigned. `00000000` is an editing placeholder. Replace it with the complete assigned article DOI and update the bibliographic fields when publication metadata are available; do not cite the placeholder as a real DOI.

```bibtex
@misc{riyahi_multibridge_faultaware_2026,
  author       = {Riyahi, Abdellah and Mestari, Mohammed and Bouihi, Bouchra},
  title        = {Fault-aware selective prediction for bridge monitoring:
                  an acquisition-level study on three bridges},
  year         = {2026},
  howpublished = {ICAM3E 2026 conference contribution and research repository},
  url          = {https://github.com/AbdellahRiyahi/MultiBridge_FaultAware_Selective_SHM},
  doi          = {00000000},
  note         = {Article DOI pending; evidence run V1_2_MAIN, implementation 1.2.0,
                  commit bd383f2747bdc97974b8dab168029ed2a475cc0f}
}
```

**Evidence snapshot:** [browse the referenced commit](https://github.com/AbdellahRiyahi/MultiBridge_FaultAware_Selective_SHM/tree/bd383f2747bdc97974b8dab168029ed2a475cc0f). A future article DOI and a future software-archive DOI would identify different objects; do not substitute one for the other.

## 11. Licensing, acknowledgements and contact

### Software and data rights

No standalone software `LICENSE` file is present in the reference snapshot. **This README does not assign an MIT, Apache, GPL or Creative Commons licence to the code.** Contact the authors regarding reuse permissions, and consult any subsequently added licence in the relevant revision. Public visibility and a publication licence are not substitutes for an explicit software licence; see [GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

Original and data-derived materials remain subject to the respective providers' terms. The data-rights notice in Section 3 applies to native-signal caches as well as source archives. Third-party dependencies retain their own licences. This research implementation is not certified for safety-critical decision making.

### Acknowledgements

We acknowledge the **KU Leuven Structural Mechanics Section** as the source of the Z24 data, the Route345 and openLAB data custodians, and the **MARWAN computing infrastructure** used for the analysis. For Z24, the benchmark description includes J. Maeck and G. De Roeck, *Description of Z24 benchmark*, Mechanical Systems and Signal Processing 17, 127–131 (2003), [DOI 10.1006/mssp.2002.1548](https://doi.org/10.1006/mssp.2002.1548).

### Contact and issue reports

**Abdellah Riyahi** — [a.riyahi@enset-media.ac.ma](mailto:a.riyahi@enset-media.ac.ma)  
2IACS Laboratory, ENSET Mohammedia, Hassan II University of Casablanca, Morocco

Use the repository's [Issues](https://github.com/AbdellahRiyahi/MultiBridge_FaultAware_Selective_SHM/issues) for reproducible software or documentation problems. Include the commit, run identifier, mode, relevant package versions, task/fold and a minimal error traceback. Do not post restricted recordings, credentials or private administrative documents. For dataset access, contact the original provider rather than attaching restricted source files to an issue.
