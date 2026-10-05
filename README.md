# Explainable Machine Learning for Predicting Flight Delay Propagation and Schedule Recovery in Airline Operations

**Module:** EAM04DS Machine Learning (MSc Data Science & AI), Emirates Aviation University
**Author:** Shamma Almheiri
**Data:** U.S. DOT Bureau of Transportation Statistics (BTS), Reporting Carrier On-Time Performance, January–December 2025

## Overview

When an aircraft arrives late, does the delay **recover**, **persist**, or **amplify** on its next flight?
This project rebuilds aircraft rotations from 7.0 million U.S. domestic flights (2025), labels each
delayed aircraft's next flight as *Recovered*, *Persistent* or *Amplified*, and compares four
classifiers (Multinomial Logistic Regression, Decision Tree, Random Forest, Support Vector Machine)
against two simple baselines. The models are then interpreted to explain *what drives delay recovery*.

## Repository contents

```
├── flight_delay_project.ipynb   # the complete, documented analysis (with outputs)
├── requirements.txt             # exact library versions used
├── outputs/
│   ├── figures/                 # all figures used in the report
│   └── tables/                  # all result tables (CSV) used in the report
└── README.md
```

The raw data, processed data and trained models are **not** included because of their size
(over 1 GB). They are fully reproducible by following the steps below.

## How to reproduce the results

### 1. Download the data

1. Open the BTS TranStats download page:
   <https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr>
2. Select **Year 2025** and one **month** at a time.
3. Tick at least these 26 fields:
   `FL_DATE, MONTH, DAY_OF_WEEK, OP_UNIQUE_CARRIER, TAIL_NUM, OP_CARRIER_FL_NUM, ORIGIN, DEST,
   CRS_DEP_TIME, DEP_TIME, CRS_ARR_TIME, ARR_TIME, CRS_ELAPSED_TIME, ACTUAL_ELAPSED_TIME,
   DEP_DELAY, ARR_DELAY, TAXI_OUT, TAXI_IN, CANCELLED, DIVERTED, DISTANCE, CARRIER_DELAY,
   WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY`
4. Download all 12 months, rename them `2025_01.csv` … `2025_12.csv`, and place them in `data/raw/`.

The notebook checks automatically that every required column is present and that each of the
12 months appears exactly once.

### 2. Set up the environment

Python 3.13 (Anaconda) was used. Install the exact library versions with:

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `flight_delay_project.ipynb` from the project folder (the folder containing `data/`) and run
**Kernel → Restart Kernel and Run All Cells**. A full run takes roughly 30–40 minutes on a laptop
and needs about 8 GB of RAM. All randomness is fixed with `SEED = 42`, so the results are identical
on every run.

The notebook saves checkpoints in `data/processed/` after each major stage, so later sections can be
re-run without repeating the slow data loading.

## Pipeline (OSEMN)

| Stage | What it does |
|---|---|
| **1. Obtain** | Loads and validates the 12 monthly files and merges them (7,001,619 flights). |
| **2. Scrub** | Removes cancelled/diverted flights and invalid records, builds time-zone-safe date-times, and rebuilds aircraft rotations from tail numbers (5,109,624 valid consecutive-flight links). Engineers features known at the prediction point (when the previous flight arrives), excluding all information about the current flight to prevent data leakage. |
| **3. Explore** | Keeps flights whose aircraft arrived 15+ minutes late (1,008,654), fixes a chronological split (train Jan–Oct, test Nov–Dec), and chooses the class threshold (T = 15 min) using the training months only. |
| **4. Model** | Tunes each model with month-grouped 5-fold cross-validation (macro-F1), evaluates once on the unseen test months, and reports 95% confidence intervals from a day-level bootstrap. |
| **5. Interpret** | Logistic Regression odds ratios, Decision Tree rules, Random Forest permutation importance, a paired bootstrap model comparison, error analysis by airline and time of day, and a threshold sensitivity analysis. |

### Target definition

With Δ = current arrival delay − previous arrival delay and T = 15 minutes:

- **Recovered:** current arrival delay < 15 min, or Δ ≤ −T
- **Amplified:** Δ ≥ +T
- **Persistent:** otherwise

## Key results (unseen test months, Nov–Dec 2025)

| Model | Macro-F1 (95% CI) | Accuracy | ROC-AUC (OvR) |
|---|---|---|---|
| **Random Forest** | **0.516** (0.506–0.523) | 0.565 | **0.745** |
| Decision Tree | 0.506 (0.498–0.512) | 0.564 | 0.729 |
| RBF SVM (20k training sample) | 0.491 (0.481–0.499) | 0.534 | 0.705 |
| Linear SVM | 0.487 (0.479–0.496) | 0.581 | 0.716 |
| Logistic Regression | 0.483 (0.472–0.492) | 0.521 | 0.710 |
| Baseline: buffer rule | 0.347 (0.341–0.353) | 0.545 | – |
| Baseline: majority class | 0.232 (0.225–0.239) | 0.532 | 0.500 |

- The Random Forest is significantly better than every other model (paired bootstrap, 95% CI of the difference above zero).
- All three interpretable analyses agree that **scheduled turnaround time and the remaining turnaround buffer** are the main drivers of delay recovery.
- Performance is not uniform: the model under-detects amplification for some airlines and for early-morning departures (see the Interpret section).

## Licence and data

The BTS on-time performance data are public U.S. government data. This repository contains code
and derived summary outputs only, produced for academic assessment.
