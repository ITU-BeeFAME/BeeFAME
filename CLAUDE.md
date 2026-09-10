# BeeFAME

Academic AI fairness thesis project: **F**airness **A**nalysis, **M**itigation and **E**xplainability for ML classifiers.
Three services + Redis, orchestrated by the root `docker-compose.yml`.

| Service | Stack | Port | Entry point |
|---|---|---|---|
| `frontend` | Next.js 13 (pages router), MUI 5, TypeScript | 3001 | `frontend/src/pages/demo.tsx` |
| `backend` | FastAPI — fairness analysis & mitigation | 8000 | `backend/app/main.py` |
| `beespector` | FastAPI — per-datapoint inspection & explainability | 8001 | `beespector/main.py` |
| `redis` | Result cache (24h TTL) | 6379 | — |

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

Two request paths, both Redis-cached with the same slugified key scheme (`version:dataset:classifier:...`):

- `POST /analysis` → `AnalysisService.analyse` → `initial_dataset_analysis` — baseline metrics, no mitigation. **Honours per-classifier hyperparameters.**
- `POST /evaluation` → `EvaluationService.evaluate` — applies a mitigation, then re-measures. **Ignores hyperparameters** — it instantiates classifiers from a hard-coded dict. Do not assume the two paths train identical models.

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

## Known gotchas

- **`beespector/model_cache/` is gitignored.** The `adult` + SVC path loads pre-trained `.pkl`
  files from it and returns **HTTP 501** on a fresh clone. Run `python pretrain_svc.py` inside
  `beespector/` to generate them.
- **The two services load different data.** Backend fetches the adult dataset as UCI id **2**;
  beespector and `pretrain_svc.py` use id **20**. They also encode German differently — backend
  uses the raw `Attribute9_A91` dummy, beespector derives a `sex` column from `attribute9`. Numbers
  are not expected to match across the two.
- `beespector/pretrain_svc.py` maps the German target as `1 if x == 1 else 0`, while
  `beespector/main.py` maps it as `0 if x == 1 else 1` — inverted. Only adult+SVC uses the
  pretrained models today, so this is latent; verify before relying on German pretrained models.
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
- Redis entries expire after 24h. To force recomputation without waiting, bump the relevant string
  in `EvaluationService.METHOD_VERSIONS` or `AnalysisService.ANALYSIS_VERSION` — that changes the key.
- Comments and log messages are a mix of English and Turkish. Match the surrounding file.

## Conventions

- The backend returns **raw signed** metric values. The "higher = fairer" transform happens
  entirely in the frontend, in `toMetrics` (`frontend/src/pages/demo.tsx:461`): SPD, EOD, AOD and
  both Theil terms become `1 - m`, while Disparate Impact goes through `symDI` = `1 - |DI - 1|`
  clamped to `[0,1]` (both DI < 1 and DI > 1 are unfair).
- `random_state=42` throughout. Keep it — the subgroup code relies on every sensitive-column split
  producing the same `X_test`.
- Sensitive columns are internal names (`age_binary`, `sex_Female`, `Attribute9_A91`,
  `race_binary`) mapped to display names via `COLUMN_DISPLAY` / `COLUMN_VALUE_LABELS` in
  `dataset_utils.py`. New sensitive attributes must be registered in both dicts.
