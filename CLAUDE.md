# BeeFAME

Academic AI fairness thesis project: **F**airness **A**nalysis, **M**itigation and **E**xplainability for ML classifiers.
Three services + Redis, orchestrated by the root `docker-compose.yml`.

| Service | Stack | Port | Entry point |
|---|---|---|---|
| `frontend` | Next.js 13 (pages router), MUI 5, TypeScript | 3001 | `frontend/src/pages/demo.tsx` |
| `backend` | FastAPI — fairness analysis & mitigation | 8000 | `backend/app/main.py` |
| `beespector` | FastAPI — per-datapoint inspection & explainability | 8001 | `beespector/main.py` |
| `redis` | Result cache (permanent; see Precomputation) | 6379 | — |

## Fairness domain work

For fairness metrics, mitigation methods, mathematical definitions, and literature-based
explanations: use the `fairness-academic-verifier` agent and the NotebookLM MCP server **if they
are configured locally**. Neither ships with the repo — `.claude/` is gitignored, so a fresh clone
has no agent definition and no MCP config. If they are unavailable, say so rather than
substituting recalled definitions silently.

When verifying a metric implementation:

1. Query NotebookLM for the standard definition.
2. Find the implementation in the repository.
3. Compare the implementation mathematically.
4. Verify required inputs: `y_true`, `y_pred`, prediction scores, sensitive attributes.
5. Only suggest renaming if mathematically equivalent.
6. If incorrect, explain the exact mismatch.

## Code modification policy

Before modifying code:

1. Explain the planned change.
2. Identify affected files.
3. Explain why the change is necessary extremely shortly — e.g. "we really need to do this because
   `[FUNCTION_NAME]` gives an error." or "we don't really need to do this but it is nice because it
   makes [...]".
4. Avoid unnecessary refactors.

## Commands

```bash
docker compose up --build     # everything; first build ~10-15 min (PyTorch + fairness libs)
docker compose down
```

Individual services (note the non-obvious paths):

```bash
# backend — requirements live in app/, and uvicorn runs from inside app/
cd backend/app && pip install -r requirements.txt && uvicorn main:app --reload --port 8000

# beespector
cd beespector && pip install -r requirements.txt && uvicorn main:app --reload --port 8001

# frontend — `npm run dev` serves 3000; `npm start` serves 3001 (what compose uses)
cd frontend && npm install && npm run dev
```

There is no test suite. Verification is manual through the wizard or the `/docs` pages
(http://localhost:8000/docs, http://localhost:8001/docs).

## Precomputation

The site is meant to serve every result from cache. `scripts/precompute.py` walks every
combination the UI offers (datasets × classifiers × methods × target subgroups, plus every
Beespector configuration) and requests each one, so Redis and Beespector's model store fill up.
Run it beside the services, never through nginx (60s gateway timeout):

```bash
python3 scripts/precompute.py --dry-run   # list jobs
python3 scripts/precompute.py             # run; safe to re-run, finished jobs are cache hits
```

What makes the combination space finite, and must stay true:
- Test size is locked at 0.2 in the UI.
- Selected subgroup pairs are **not** part of any cache key. A miss computes every pair the
  dataset offers and `filter_subgroup_rows` trims the response to the selection.
- Cache entries never expire unless `CACHE_TTL_SECONDS` is set.
- Every randomised component is seeded (`ExponentiatedGradient.predict`,
  `ThresholdOptimizer.predict`, `SVC(random_state=42)`), so a cached result equals a fresh one.
  Keep new mitigators seeded too.

## Backend architecture

Strict `controller → service → repository` layering under `backend/app/`:

- `controller/` — thin FastAPI routers, no logic. Prefixes: `/analysis`, `/evaluation`,
  `/datasets`, `/classifiers`, `/methods`. Plus `/status`.
- `service/` — all business logic, Redis caching, model training.
- `service/utils/dataset_utils.py` — **the fairness metric implementations** (SPD, EOD, AOD, DI via
  `fairlearn.MetricFrame`; Theil-T decomposition; per-subgroup metrics). Start here for metric questions.
- `service/utils/evaluation_utils.py` — the ten `run_*` mitigation wrappers.
- `repository/` — reads the static JSON in `data/`.
- `data/{dataset,classifier,method}.json` — the catalogue the UI renders. Adding an entry here is
  **not enough**: the matching `Enum` in `model/{dataset,classifier,method}.py` and the dispatch
  dict in `service/evaluation_service.py` must be updated too.

Two request paths, both Redis-cached with the same slugified key scheme (`version:dataset:classifier:...`).
Neither key includes the selected subgroup pairs (see Precomputation):

- `POST /analysis` → `AnalysisService.analyse` → `initial_dataset_analysis`: baseline metrics, no mitigation. Trains with the parameters the panel sends.
- `POST /evaluation` → `EvaluationService.evaluate`: applies a mitigation, then re-measures. **Ignores the panel's parameters** and builds classifiers from a hard-coded dict.

Both stages must train the same model, or every before/after comparison mixes the mitigation's
effect with a model change. The defaults in `data/classifier.json` are therefore set to exactly the
values the evaluation dict uses (library defaults, verified to give bit-identical models), and the
panel shows them read-only. **Change one, change the other**, and bump the cache versions.

Mitigations are dispatched by three sets in `evaluation_service.py`, and each set has a different
call signature and training flow:

- Preprocessing (`Relabeller`, `Prevalence Sampling`, `Data Repairer`) — transform `(X_train, y_train)`, then fit.
- `Fairness Through Unawareness` — special-cased: drops the sensitive column and **re-fits its own scaler**.
- Inprocessing (`Exponentiated Gradient`, `Grid Search`) — return a fitted fairlearn model.
- Postprocessing (`Reject Option Classification`, `Equalized Odds Postprocessing`, `Threshold Optimizer DP`) — return `y_pred` directly.

## Frontend architecture

- `pages/demo.tsx` (~1400 lines) is the four-step wizard: Select Dataset & Classifier → Check Bias
  Metrics → Select Mitigation → Review Results. `demo-backup.tsx` is a stale copy; do not edit it.
- `contexts/BeeFameContext.tsx` holds all wizard state **in memory only**. Consequence: `/beespector`
  loaded directly or after a refresh finds no selection and redirects back to `/demo`.
- Nothing calls the APIs directly from the browser. Two different proxy mechanisms:
  - `/api/backend/*` → hand-written Node proxy at `pages/api/backend/[...path].ts` (follows the
    FastAPI 307 redirects that trailing-slash routes emit).
  - `/api/beespector/*` → a `rewrites()` rule in `next.config.js`.
  Both read `BACKEND_INTERNAL_URL` / `BEESPECTOR_INTERNAL_URL`, set by compose.
- `next.config.js` sets `eslint.ignoreDuringBuilds` and `typescript.ignoreBuildErrors`. A green
  `npm run build` proves nothing about type correctness — run `npx tsc --noEmit` when it matters.

## Beespector

Self-contained; it does **not** import from `backend/`. It re-implements `run_data_repairer`,
`run_prevalence_sampling`, `run_relabeller` and `run_ftu` at the top of `main.py`. **A fix to a
mitigation in `backend/app/service/utils/evaluation_utils.py` does not propagate here** — check
both copies.

State lives in a single module-level `context` global set by `POST /api/initialize_context`; every
other endpoint reads it. The service is effectively single-session.

Trained models are stored in `model_cache/` (the `beespector_models` Docker volume), one file per
dataset/classifier/mitigation/params combination, and loaded on later visits instead of retrained.
Dataset copies are cached there too, so UCI is only contacted once. **Bump `MODEL_CACHE_VERSION`
whenever training code changes**, or stale models will keep being served.

## Known gotchas

- `beespector/pretrain_svc.py` is unused: the Adult + SVC special case it fed was replaced by the
  general model store.
- **The two services load different data.** Backend fetches the adult dataset as UCI id **2**;
  beespector and `pretrain_svc.py` use id **20**. Both derive German gender from `attribute9`
  (A92/A95 = female), but the backend drops `Attribute9` and beespector keeps it. Numbers are not
  expected to match across the two.
- German's target is **1 = good credit** (the favourable outcome) in both services, so SPD and DI
  measure who is granted good credit. Before 2026-10-12 it was inverted (1 = bad credit).
- `service/redis_client.py` reads `REDIS_HOST`/`REDIS_PORT`/`REDIS_DB`, and falls back to inferring
  the host from the presence of `/.dockerenv`. It does **not** understand a `REDIS_URL`-style
  connection string.
- `NEXT_PUBLIC_BEEFAME_API_URL` / `NEXT_PUBLIC_BEESPECTOR_URL` are **dead** — no source file has
  referenced them since the proxy refactor. If you see them in a deployed bundle, you are looking
  at a pre-refactor build.
- `"Label Flipping"` is implemented in `evaluation_utils.py` and has a cache version entry, but is
  absent from `method.json` and the `MethodName` enum, so it is unreachable from the UI.
- `backend/docker-compose.yaml` is a stale standalone file (references a missing `.env`). The root
  `docker-compose.yml` is the real one.
- Cache entries are permanent, so a code change that alters results will **not** show up until you
  bump the relevant version (`EvaluationService.METHOD_VERSIONS`, `AnalysisService.ANALYSIS_VERSION`,
  `DatasetService.SUBGROUP_PAIRS_VERSION`, Beespector's `MODEL_CACHE_VERSION`) and re-run the precompute.
  A change to one dataset's preprocessing bumps only `DATASET_CACHE_VERSIONS` in
  `repository/dataset_repository.py`, which invalidates that dataset's results and nothing else;
  Beespector has its own `DATASET_CACHE_VERSIONS` for its stored dataset copies and models.
- Beespector does not implement Threshold Optimizer DP: it falls through to a sample-weighting
  fallback, so Deep Dive shows a different model under that name.
- Comments and log messages are a mix of English and Turkish. Match the surrounding file.

## Conventions

- The backend returns **unsigned** group gaps: SPD, EOD, AOD and DI use fairlearn's
  `between_groups` (largest minus smallest, smallest over largest), so they say how large a gap
  is but not which group it favours, and DI never exceeds 1. The "higher = fairer" transform happens
  entirely in the frontend, in `toMetrics` (`frontend/src/pages/demo.tsx:461`): SPD, EOD, AOD and
  both Theil terms become `1 - m`, while Disparate Impact goes through `symDI` = `1 - |DI - 1|`
  clamped to `[0,1]` (both DI < 1 and DI > 1 are unfair).
- `random_state=42` throughout. Keep it — the subgroup code relies on every sensitive-column split
  producing the same `X_test`.
- Sensitive columns are internal names (`age_binary`, `sex_Female`, `race_binary`; German uses
  `sex_Female` too) mapped to display names via `COLUMN_DISPLAY` / `COLUMN_VALUE_LABELS` in
  `dataset_utils.py`. New sensitive attributes must be registered in both dicts.
