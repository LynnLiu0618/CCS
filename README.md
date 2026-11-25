# CCS

English | [简体中文](README.zh-CN.md)

**CCS-FD** uses measured single faults to recognize previously unseen fault combinations. It scores each candidate combination using calibrated evidence for its constituent faults. It then accepts the candidate if at least five reference samples for the corresponding bearing state lie within the calibrated radius of the measured response. Otherwise it returns `UNKNOWN`.

This repository contains the BJTU-RAO implementation for *Open-Set Compound Fault Diagnosis via Composition Prediction and Local Support Learning*. [Five pretrained models](ccs_fd/checkpoints/) · [Per-run results](results/paper_runs.csv) · [Training guide](training/README.md)

## Run a first prediction

Clone the repository, install the package with Python 3.10 or later, and run the CPU checks. If you downloaded the source ZIP, start with the install command from the extracted folder. The reference environment used Python 3.10, NumPy 1.26.4 and PyTorch 2.7.1.

~~~sh
git clone https://github.com/LynnLiu0618/CCS.git
cd CCS
python -m pip install -e .
python -m unittest discover -s tests -v
~~~

The tests run on CPU with generated inputs. To diagnose recorded data, save the synchronized windows in a numeric NumPy file of shape **(N, 2048, 18)** and run:

~~~sh
python -m ccs_fd windows.npy --device cpu --output predictions.csv
~~~

The CSV gives the window index, candidate, final prediction and squared distance to the fifth nearest reference in the selected state. The final prediction is the candidate when this distance is at or below the state's calibrated threshold, and `UNKNOWN` otherwise. Use `--features` for an `(N, 2628)` feature array produced by this extractor. The default run is seed 1994217509. Use `--checkpoint ccs_fd/checkpoints/1454668731.pt` to select another run.

~~~python
import numpy as np
from ccs_fd import Predictor

model = Predictor(device="cpu")
windows = np.load("windows.npy", allow_pickle=False)
result = model.predict_windows(windows, return_detail=True)
print(result["label"][:5])     # candidates before verification
print(result["pred"][:5])      # final decisions
print(result["distance"][:5])  # squared fifth-neighbor distance
~~~

## BJTU-RAO input and labels

Each row is a 32 ms window sampled at 64 kHz. Preserve the original signal units and the order **CH1–CH9, CH11–CH19**. CH10 (the tachometer) is omitted. The channels comprise six motor-acceleration channels, three motor-current channels, six gearbox-acceleration channels and three left axle-box acceleration channels. The feature extractor returns 900 motor, 600 gearbox and 1128 bearing features. The trained checkpoints expect this sensor arrangement.

| Code | Fault |
| --- | --- |
| A | Motor short circuit |
| B | Missing gear tooth |
| C / D / E | Left bearing: inner race / outer race / cage |
| AB | Motor short circuit and missing gear tooth |
| CD | Inner- and outer-race damage |
| F | Rolling-element fault held out for evaluation; correct output is `UNKNOWN` |

Candidates A, B and AB select the reference state H (no seeded bearing fault). The remaining candidates select their corresponding bearing state. Only the verification branch uses the axle-box features alone.

Obtain the recordings from the [BJTU-RAO dataset folder listed in the authors' user guide](https://drive.google.com/drive/folders/1RlZvFw-v07VvsL2Ni9cS7iFrTPDIhn2r?usp=sharing). The test rig and dataset are described by [Ding et al., Mechanical Systems and Signal Processing, 210 (2024), 111175](https://doi.org/10.1016/j.ymssp.2024.111175). Supply your local dataset folder in the commands below.

## Reproduce the evaluation

~~~sh
python -m pip install -e ".[prepare]"
python -m training.prepare_data evaluation --data-root /path/to/BJTU_RAO_Bogie_Datasets --output-dir data
python evaluate.py data/evaluation.npz --output report.json
~~~

The evaluation uses 4464 measured windows, with 62 windows per fault and operating condition, eight fault states and nine conditions. The five runs use the same time split on one test rig. The numbers below are window-level means and sample standard deviations across runs, in percent.

| Metric | Mean ± SD |
| --- | ---: |
| Overall exact-label accuracy | 94.45 ± 0.36 |
| Single-fault exact-label accuracy | 95.86 ± 0.46 |
| AB recall | 93.91 ± 1.27 |
| CD recall | 92.37 ± 1.58 |
| F rejection | 90.07 ± 1.11 |
| Known windows rejected | 1.47 ± 0.21 |
| AUROC | 99.29 ± 0.07 |

[Per-run results](results/paper_runs.csv), [summary at full precision](results/paper_summary.csv) and [checkpoint SHA-256 digests](results/checkpoints.json) are provided. The [replay notes](results/README.md) describe the numerical comparison with the paper. Each checkpoint represents one run and includes its classifier, verifier, normalization statistics, reference embeddings and calibrated thresholds.

## Train on source data

~~~sh
python -m training.prepare_data source --data-root /path/to/BJTU_RAO_Bogie_Datasets --output-dir data
python -m training.fit --data data --seed 1994217509 --output outputs/model.pt
~~~

The fitting data use measured A–E windows and synthetic AB/CD combinations constructed from source single faults. Measured AB, CD and F windows are reserved for evaluation. Fitting, calibration, validation and evaluation use the half-open intervals [0, 3.488), [5.512, 6.504), [6.504, 7.496) and [8.012, 9.996) seconds, respectively. We checked candidate labels and the order of neighbor distances on validation windows. See the [training instructions](training/README.md) for the source arrays, seed handling and model export.
