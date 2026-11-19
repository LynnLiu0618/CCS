# Reference results

`paper_runs.csv` contains the seven window-level metrics for the five frozen models. `paper_summary.csv` gives their means and sample standard deviations in percent. To generate the same report from labeled evaluation features, run `python evaluate.py data/evaluation.npz --output report.json` from the repository root.

AUROC treats F as positive and ranks windows by the squared fifth-neighbor distance divided by the cutoff for the candidate state. Tied scores receive their average rank. The paper computed neighbor distances in float64. The packaged predictor computes distances in float32 and repeats the calculation in float64 near the decision cutoff. On the original 4,464 windows, both CPU and CUDA reproduced every candidate and final diagnosis for all five runs. Precision differences changed individual AUROCs by at most 0.000046 percentage points in this check. Both reproduced the reported AUROC of 99.29 ± 0.07%.
