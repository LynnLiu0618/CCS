# Train CCS-FD on prepared source features

From the repository root, install the CSV reader and training dependencies, prepare the source features from the BJTU-RAO recordings, and fit a model:

```bash
python -m pip install -e ".[prepare]"
python -m training.prepare_data source --data-root /path/to/BJTU_RAO_Bogie_Datasets --output-dir ./source
python -m training.fit --data ./source --seed 1994217509 --output ./runs/1994217509.pt
```

Training uses CUDA when available. Add `--device cpu` for CPU execution. The command fits both networks from scratch, calibrates the component cutoffs and state thresholds, and writes a checkpoint accepted by `ccs_fd.Predictor`. It reads only the three input files below.

## Input files

| File | Contents | BJTU paper protocol |
| --- | --- | --- |
| `train.npz` | Source fitting features and aligned metadata. | 16,002 rows including 10,665 clean rows |
| `calibration.npz` | Source calibration features and the same metadata. | 11,718 rows including 7,812 clean rows |
| `calibration_augmented.npy` | One perturbed verifier calibration view for **every clean row** of `calibration.npz`, with matching row order. | `(7812, 2628)`, float32 |

The two NPZ files have the following keys. Use numerical or Unicode arrays so that they load with `allow_pickle=False`.

| Key | Shape | Meaning |
| --- | --- | --- |
| `x` | `(N, 2628)` | Float32 features from the public window extractor, before classifier or encoder standardization. |
| `names` | `(N,)` | `A`, `B`, `C`, `D`, `E`, `AB`, or `CD`. Fitting and calibration use measured single faults and **source-synthesized** compounds. |
| `context` | `(N,)` | Integer operating-condition index, 0–8. In the reference data the load order is 0, +10, −10 kN with 20, 40, 60 Hz in each load. |
| `groups` | `(N,)` | Original sampling group, e.g. `A` or `AB:block:0.25`. Compound groups encode the synthesis family and the amplitude ratio. |
| `augmentation` | `(N,)` | `clean` for the base view, `both` for classifier augmentation. |
| `recipe` | `(N, >=4)` | First four columns are the first parent's fault ID and window index, then the second parent's ID and index. IDs 0–4 correspond to A–E; singletons use −1 for the absent second parent. The paper's recipe has 12 columns. |

Use the original disjoint fitting and calibration time intervals. Keep the archived order of features and metadata within each split for comparisons across seeds. Classifier augmentation uses selected source rows. The verifier requires a separate perturbation aligned to the **whole** clean calibration set.

## What the training command fits

1. **Component prediction.** Fifty epochs with 48 updates of 256 observations each. Sampling balances condition, source group and augmentation. The training network predicts A–E plus an auxiliary indicator for an unaffected bearing. At export the five fault heads are retained and consecutive linear maps are combined. The source calibration set determines five probability cutoffs.
2. **Candidate-state verification.** The 128-dimensional encoder trains on clean source observations in 40 epochs of 32 episodic updates. Within each source recording, training parent indices below 54 provide reference observations; indices starting at 54 provide queries. Both parents of a synthetic compound must belong to the same pool to enter an episode.
3. **Reference banks and calibration.** The fitted encoder embeds all clean fitting rows into five banks (`H`, `C`, `D`, `E`, `CD`). The verifier computes squared fifth-neighbor distances and calibrates each state with the maximum of the per-label 97.5th percentiles over clean and paired perturbed views.

The paper reports the sample standard deviation across five separate fits on the same temporal split. To repeat them, run the command once per seed: `1994217509`, `1454668731`, `1061685734`, `577998202`, and `1919707711`. In the reference environment, regenerating the source features from the original CSVs and training seed `1994217509` reproduced both networks, the normalization parameters, the evidence cutoffs, all five reference banks and their distance thresholds exactly. Reproduction of exact tensors depends on the input features, ordering, numeric libraries and hardware.
