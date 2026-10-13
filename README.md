# BeeFAME

**Fairness Analysis, Mitigation and Explainability** platform for ML models.

BeeFAME walks you through a four-step wizard to audit and reduce bias in ML classifiers. Pick a dataset, pick a classifier, apply a mitigation strategy, then compare fairness metrics before and after — side by side. For deeper inspection, **Beespector** lets you explore individual predictions and partial dependence plots.

**Live demo: [beefame.site](https://beefame.site).** It is a demonstration instance that serves two
benchmark datasets and four classifiers from a precomputed cache. To work with your own data or
model, clone this repository and add them ([Bring Your Own Dataset or Classifier](#bring-your-own-dataset-or-classifier)).

## Services

| Service | Description | Port |
|---|---|---|
| `frontend` | Next.js web UI | 3001 |
| `backend` | FastAPI — bias analysis & mitigation | 8000 |
| `beespector` | FastAPI — model inspection & explainability | 8001 |
| `redis` | Cache for analysis results | 6379 |

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (must be running)

## Running with Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/ITU-BeeFAME/BeeFAME.git
cd BeeFAME

# Start all services
docker compose up --build
```

Then open **http://localhost:3001** in your browser.

> **First build takes ~10–15 minutes** — the backend pulls PyTorch and several ML fairness libraries (~3 GB). Subsequent builds are fast thanks to Docker layer caching.

To stop everything:
```bash
docker compose down
```

## Demo Walkthrough

1. **Select Dataset & Classifier** — choose German Credit Data or Census Income, then pick a classifier (Random Forest, XGBoost, SVC, or Logistic Regression).
2. **Check Bias Metrics** — radar charts display six fairness metrics per sensitive attribute (Age, Gender, Race) before any mitigation.
3. **Select Mitigation** — choose one of nine strategies across three families:
   - *Preprocessing*: Relabeller, Prevalence Sampling, Data Repairer, Fairness Through Unawareness
   - *Inprocessing*: Exponentiated Gradient, Grid Search
   - *Postprocessing*: Reject Option Classification, Equalized Odds Postprocessing, Threshold Optimizer DP
4. **Review Results** — compare original vs. mitigated metrics side-by-side. Hit **Deep Dive with Beespector** to inspect individual predictions.

> Every result is cached in Redis after it is first computed. Classifiers run at their library
> defaults (shown read-only), so the configuration space is finite and `scripts/precompute.py` can fill
> the whole cache in advance, as on the demo site.

## Bring Your Own Dataset or Classifier

[beefame.site](https://beefame.site) is a demo instance with two benchmark datasets and four
classifiers. The tool itself is this repository: add your own dataset or classifier, run the stack
locally, and it goes through the same analysis, mitigation and comparison pipeline as the built-in
ones and appears next to them in the wizard. Locally every configuration is computed on demand and
then cached.

### Add a classifier

Any scikit-learn-compatible estimator works. In-processing methods need `fit(..., sample_weight=...)`;
probability-based views (AUC, Theil) use `predict_proba` when it exists.

1. `backend/app/data/classifier.json`: add an entry with its name, documentation URL and parameters
   with their defaults (shown read-only in the wizard).
2. `backend/app/model/classifier.py`: add the same name to `ClassifierName`.
3. `backend/app/service/utils/dataset_utils.py`, `initial_dataset_analysis`: construct it in the
   classifier `if`/`elif` chain (baseline stage).
4. `backend/app/service/evaluation_service.py`, the `classifiers` dict: construct it with the same
   defaults (mitigation stage). Both stages must train the same model.
5. Deep Dive (optional): add it to both `clf_map` dictionaries in `beespector/main.py`, keyed by the
   slug the wizard sends (the name in lowercase, spaces as `_`, `_classifier` and `_(svc)` removed).

### Add a dataset

Each protected attribute becomes one binary column (1/0), and the target must be 1 for the favourable
outcome.

1. `backend/app/data/dataset.json`: add an entry with a `slug`, name, URL, instance count,
   description and its protected attributes (name, privileged and unprivileged label).
2. `backend/app/model/dataset.py`: add the slug to `DatasetName`.
3. `backend/app/repository/dataset_repository.py`: register the data source in `DATASET_ID_MAP`, map
   each attribute to its binary column in `SENSITIVE_COLUMNS_MAP`, describe the columns in
   `COLUMN_PRIVILEGE_MAP` and `COLUMN_VALUE_LABELS`, choose `COMBINATION_SIZES` for intersections, and
   add the preprocessing to `get_subgroup_pairs`.
4. `backend/app/service/utils/dataset_utils.py`: add the columns to `COLUMN_VALUE_LABELS` and
   `COLUMN_DISPLAY`, and the same preprocessing (plus target mapping) to `initial_dataset_analysis`.
5. `backend/app/service/evaluation_service.py`: add the dataset to the `datasets` dict and the same
   preprocessing.
6. Deep Dive (optional): add it to `load_dataset` and to the feature setup in `initialize_context` in
   `beespector/main.py`, and define its protected groups in `protected_group` the same way as in step 3.

Datasets are downloaded from the [UCI repository](https://archive.ics.uci.edu/) by id. For a dataset
that is not on UCI, such as a CSV file, have `fetch_dataset` in `dataset_utils.py` and the download in
`get_subgroup_pairs` return its `(features, target)` DataFrames instead (for example with
`pd.read_csv`).

Then rebuild with `docker compose up --build`. To fill the cache in advance, run
`python3 scripts/precompute.py --dataset <slug>`.

## Fairness Metrics

The API returns unsigned gaps between groups (largest minus smallest rate; Disparate Impact as the
smallest over the largest rate). The UI transforms them so **higher = fairer**:

| Metric shown | Transform | Ideal |
|--------|-----------|-------|
| Statistical Parity Difference (1-m) | `1 - SPD` | 1.0 |
| Equal Opportunity Difference (1-m) | `1 - EOD` | 1.0 |
| Average Odds Difference (1-m) | `1 - AOD` | 1.0 |
| Disparate Impact | `1 - abs(DI - 1)`, clamped to [0,1] | 1.0 |
| Theil Between (1-m) | `1 - T_between` | 1.0 |
| Theil Within (1-m) | `1 - T_within` | 1.0 |

Disparate Impact uses a symmetric transform so that a ratio on either side of 1 counts as unfair. Theil-T is reported decomposed (Shorrocks 1980), not as a
single index.

## Subgroup (Intersectional) Analysis

Beyond single sensitive attributes, BeeFAME supports **intersectional fairness** — analysing groups defined by the combination of two or three sensitive columns at once (e.g. Gender × Age, Gender × Race × Age).

### How it works

1. **Select subgroup pairs** — in the wizard's Step 1, choose one or more attribute combinations to inspect. Available pairs are fetched from `GET /datasets/{slug}/subgroup-pairs`.
2. **Subgroups are formed automatically** — every unique value combination becomes its own group (e.g. *Gender=Female × Age=Young*, *Gender=Male × Age=Old*, …).
3. **Per-group metrics are computed** — each group is compared against the overall test-set baseline:

| Metric | Description |
|--------|-------------|
| SPD | Selection-rate gap vs. overall |
| EOD | True-positive-rate gap vs. overall |
| AOD | Average of TPR + FPR gap vs. overall |
| DI | Selection-rate ratio vs. overall |
| Theil | Entropy divergence for that group |
| FPR / FNR divergence | Absolute FPR and FNR gap vs. overall |
| PPV | Precision within the group |
| Subgroup AUC | ROC-AUC within the group |
| BPSN / BNSP / Pinned AUC | Background-positive/negative AUC variants |
| Accuracy | Accuracy within the group |

4. **Privilege status** — each group is labelled *privileged*, *unprivileged*, or *neutral* based on the average of its SPD, EOD, and AOD (threshold ±0.05).
5. **Worst-case group** — the unprivileged/neutral group with the highest weighted divergence is flagged as the worst case (Kearns 2018 γ-unfairness certificate).

### Fairness Index

Reported alongside each subgroup pair result:

```
Fairness Index = Σ (support_g × max(|SPD_g|, |FPR_div_g|, |FNR_div_g|))
                 for groups where support_g > 0.1
```

- **support_g** = fraction of the test set that belongs to group *g*
- Only groups representing at least 10% of the test set are included (small groups are noisy)
- **Lower = fairer** (0 means no measured divergence for any large group)
- Adapted from Lin, Gupta and Jagadish, *Mitigating Subgroup Unfairness in Machine Learning
  Classifiers: A Data-Driven Approach* (ICDE 2024), which sums one statistic at a time and also
  requires each group's divergence to be statistically significant (t-test); BeeFAME omits that test.

### Target Subgroup Mitigation

When running the evaluation wizard you can optionally nominate one group as the **target subgroup**. The chosen mitigation algorithm then uses a binary sensitive attribute (1 = target group, 0 = everyone else) instead of the original sensitive column, focusing bias reduction on that specific intersection. Fairness metrics are still reported against the original sensitive columns.

## Terminal Interface

The same pipeline is available as an interactive terminal menu, without a browser:

```bash
docker compose exec backend python -m cli.cli_macos_linux
```

For scripting, use the REST API below; `scripts/precompute.py` is an example client that sweeps every combination.

## API Docs

Once running, interactive API docs are available at:
- Backend: http://localhost:8000/docs
- Beespector: http://localhost:8001/docs

## Environment Variables

The browser never calls the APIs directly — it calls same-origin `/api/backend/*`
and `/api/beespector/*`, which the Next.js server proxies to the internal containers.
`docker-compose.yml` sets these already; copy `.env.example` to `.env` only for
non-compose setups:

| Variable | Default | Description |
|---|---|---|
| `BACKEND_INTERNAL_URL` | `http://backend:8000` | Backend URL, resolved server-side by the frontend container |
| `BEESPECTOR_INTERNAL_URL` | `http://beespector:8001` | Beespector URL, resolved server-side by the frontend container |
| `REDIS_HOST` | `redis` in Docker, else `127.0.0.1` | Redis host for the backend cache |
| `REDIS_PORT` | `6379` | Redis port |
| `SERVE_FROM_CACHE_ONLY` | `0` | `1` on the demo server: results come from the precomputed cache only; anything not precomputed answers 503 instead of being trained. Leave it unset locally, where every configuration is computed on demand |

## Project Structure

```
BeeFame/
├── backend/          # FastAPI — fairness analysis & mitigation API
│   ├── app/
│   │   ├── controller/   # Route handlers
│   │   ├── service/      # Business logic + Redis caching
│   │   ├── model/        # Pydantic schemas
│   │   ├── repository/   # Data access
│   │   ├── data/         # Static JSON (datasets, classifiers, methods)
│   │   └── requirements.txt
│   └── dockerfile
│
├── beespector/       # FastAPI — model inspection & partial dependence API
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/         # Next.js 13 — web UI
│   ├── src/
│   │   ├── pages/        # Next.js pages (demo, beespector, index)
│   │   ├── components/   # React components
│   │   ├── lib/          # Axios API clients
│   │   └── contexts/     # React context (shared state)
│   ├── Dockerfile
│   └── package.json
│
└── docker-compose.yml
```

## Running Services Individually (without Docker)

### Backend
```bash
cd backend/app
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Requires Redis running locally on port 6379. Note the requirements file lives in
`backend/app/`, and `uvicorn` must run from that directory — the imports are
top-level (`from controller import ...`), not package-relative.

### Beespector
```bash
cd beespector
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

### Frontend
```bash
cd frontend
npm install
npm run dev     # dev server on http://localhost:3000
npm start       # production server on http://localhost:3001 (what Docker uses)
```
