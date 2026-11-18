# ShopMind AI

[Live Demo](https://dashboardpy-atdyds5ymth55g6m623f5z.streamlit.app/) · [Portfolio](https://jiuxin77.github.io/) · [System Architecture](docs/system_architecture.md)

**Jiuxin (Harriet) Xiao · Independent Project**

ShopMind AI is an end-to-end e-commerce recommendation and shopping-agent project. It combines customer analytics, multi-channel retrieval, ranking, grounded LLM recommendations, visual similarity search, offline evaluation, and bad-case analysis in one reproducible demo.

## Three-Minute Product Tour

1. Explore the customer funnel, user segments, price sensitivity, category preferences, repeat purchases, and churn signals.
2. Compare popularity, ItemCF, BM25, TF-IDF, recent-interest, and basket-based retrieval through weighted Reciprocal Rank Fusion.
3. Ask the LLM shopping agent for a product. The agent either asks a necessary follow-up question or calls catalog search and returns recommendations backed by retrieved product evidence.
4. Review offline metrics and bad cases, including the M7 reranker that failed the release gate and was not promoted.
5. Try image-to-product retrieval. This module is a lightweight baseline on deterministic synthetic thumbnails, not a claim of real-photo or CLIP-level semantic retrieval.

## Scope and Metric Definitions

| Area | Definition |
| --- | --- |
| Online demo | 300 synthetic users, 200 products, and 5,000 demo interactions |
| Offline experiments | A separately prepared UCI Online Retail dataset and fixed experiment user slices |
| M6 comparison | Same 200-user test slice; HitRate@10: 9.5% to 11.0% (+1.5 percentage points); NDCG@10: 0.0385 to 0.0454 |
| Candidate recall and latency | Candidate Recall@50 remained 20%; P95 changed from 47.2 ms to 49.0 ms; latency is offline retrieval time, not webpage latency |
| M7 reranker | HitRate@10 was 10%, below the retained 11% baseline, so the offline release gate rejected it |
| LLM agent evaluation | Six fixed real-API cases: 100% tool-selection accuracy, 100% grounded-product rate, 100% constraint satisfaction, 0% fallback rate, and 2.64 s average latency |
| Automated checks | The current local regression suite contains 21 passing tests |

The later retrieval experiments repeatedly compare variants on the same held-out user slice. They are exploratory offline comparisons, not untouched final blind-test results or measured online business lift. A production iteration should freeze the selected approach and evaluate it on a new time-based holdout.

## Six Demonstrable Modules

1. **Customer and business analytics** — conversion funnel, customer segmentation, price sensitivity, category affinity, repeat purchase, and churn analysis.
2. **Recommendation baselines** — global popularity and category popularity as transparent comparison points.
3. **Multi-channel retrieval and ranking** — ItemCF, BM25, TF-IDF, recent-interest, basket co-occurrence, filtering, deduplication, and weighted RRF.
4. **Grounded LLM shopping agent** — OpenAI Responses API, structured outputs, tool calling, selective clarification, product-evidence constraints, and safe fallback.
5. **Visual product search** — lightweight visual features for demo-image similarity search, with an evaluation baseline for future real-image and CLIP integration.
6. **Evaluation and bad-case analysis** — Precision@K, Recall@K, NDCG@K, HitRate@K, coverage, P95 latency, error categories, and release gates.

Milestone `M12` includes all six modules, a real LLM agent, an independently deployable FastAPI service, reproducible experiments, system documentation, GitHub Actions CI, Docker support, and Streamlit Cloud configuration.

## System Flow

```text
User need
   |
   v
LLM decision ---- underspecified ----> Clarification question
   |
   | searchable request
   v
Catalog tool -> Multi-channel retrieval -> Constraint filtering -> Top-K evidence
   |                                                        |
   +---------------- Structured grounded answer <-----------+
                            |
                    Provider failure
                            v
                 Local rule-based fallback
```

### Retrieval Pipeline

```text
Popularity ---------+
ItemCF -------------+--> Weighted Reciprocal Rank Fusion --> Business filters --> Top-K
BM25 / TF-IDF ------+
Recent interest ----+
Basket co-occurrence+
```

Evaluation uses a leave-one-out protocol based on each user's last positive interaction. Reports include Precision@K, Recall@K, HitRate@K, NDCG@K, catalog coverage, mean latency, and P95 latency.

## Experiment History

### M3: Real-Data Baseline

M3 uses the UCI Machine Learning Repository's Online Retail dataset: 541,909 transactions from a UK online retailer collected in 2010–2011 under CC BY 4.0. In the first 50-user experiment, hybrid retrieval achieved 4% HitRate@10 versus 2% for each single-channel baseline. TF-IDF retrieval reached 43.38% item coverage. These are reproducible intermediate baselines, not statistically significant production gains.

### M4: ItemCF Precomputation

The evaluation expanded to 200 users and precomputed ItemCF similarities. ItemCF P95 latency dropped from 147.98 ms to 7.25 ms, while hybrid P95 dropped from 185.17 ms to 44.37 ms. TF-IDF achieved the highest HitRate@10 at 7.5%, while hybrid retrieval achieved the highest NDCG@10 at 0.0287.

### M5: Weight Selection and Independent Test Slice

The first 100 users were used for fusion-weight selection and the following 200 users for testing. Doubling the TF-IDF channel weight produced 9.5% HitRate@10, 0.0385 NDCG@10, and 63.13% coverage. Bad-case analysis found 160 candidate-recall failures and 21 ranking failures where the target was retrieved but missed the top ten.

### M6: Recall Expansion

Recent-interest and basket co-occurrence channels were added on the same 200-user test slice. With a recent-interest weight of 0.25, Candidate Recall@50 remained 20%, HitRate@10 improved from 9.5% to 11.0%, NDCG@10 improved from 0.0385 to 0.0454, and P95 latency increased from 47.2 ms to 49.0 ms. Basket co-occurrence did not improve this slice and was excluded from the default blend.

### M7: Reranking Release Gate

A logistic-regression reranker used 12 features, including RRF score, source ranks, item popularity, price deviation, and user-history length. Its 10% HitRate@10 was below M6's 11%, so the release gate rejected it. The experiment exposed a sparse candidate-training set with only 20 positive examples and motivates improved negative sampling, more users, and tree-based ranking models.

### M8: Constraint-Aware Shopping Assistant

The local assistant extracts category, price range, scenario, style, and exclusions, applies those constraints to retrieval, and generates reasons that include price and retrieval evidence. On a six-request, 36-slot regression set, slot accuracy and recommendation constraint satisfaction were both 100%. This is a functional regression metric, not an open-domain generalization claim.

### M9: Visual Retrieval Baseline

The visual module generates deterministic synthetic thumbnails and extracts brightness-invariant chromatic ratios, hue and saturation histograms, spatial color, and edge features. Cosine-similarity retrieval achieved 100% Top-1 and Top-5 accuracy on 50 synthetic queries with brightness scaled from 0.72x to 1.28x. The result measures robustness on generated demo images only.

### M12: Grounded LLM Agent and Deployable API

The agent must first call either `ask_clarification` or `search_catalog`. Retrieved product fields and evidence are returned to the model, and the final response is validated with a Pydantic structured-output schema. A product-ID allowlist removes unsupported recommendations. Timeouts, provider errors, and malformed outputs fall back to the M8 local pipeline.

The FastAPI service includes environment-based configuration, structured JSON logs, request IDs, liveness and readiness probes, consistent error responses, a Docker health check, and GitHub Actions release checks.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/generate_demo_data.py
streamlit run src/shopmind/dashboard.py
```

One-command demo startup:

```bash
python scripts/run_demo.py
```

Start the API:

```bash
uvicorn shopmind.api:app --reload
```

Configure the LLM agent through an environment variable or Streamlit Secrets. Never commit an API key:

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env
docker compose up --build
curl http://localhost:8000/health/ready
```

The agent endpoint is `POST /llm-shopping-agent`. Interactive API documentation is available at `http://localhost:8000/docs`.

For Streamlit Cloud, open the app settings, select **Secrets**, and add:

```toml
OPENAI_API_KEY = "your-server-side-key"
OPENAI_MODEL = "gpt-4.1-mini"
```

The public dashboard never asks visitors for an API key and never returns the configured key to the browser. To control portfolio-demo cost, the UI uses the configured mini model and permits three LLM calls per browser session.

## Reproduce Experiments

```bash
python scripts/prepare_uci_online_retail.py --max-users 500 --max-products 1000
python scripts/run_retrieval_experiment.py --users 50 --k 10
python scripts/build_item_cf_index.py
python scripts/tune_fusion_weights.py --validation-users 100 --test-users 200
python scripts/run_recall_expansion_experiment.py
python scripts/run_ranking_experiment.py
python scripts/evaluate_shopping_assistant.py
python scripts/evaluate_visual_search.py
OPENAI_API_KEY=... python scripts/evaluate_llm_agent.py --model gpt-4.1-mini
```

Raw and processed UCI files are stored in Git-ignored directories. The repository includes only download, cleaning, experiment, and result-generation code to avoid redistributing large data files.

## Testing and Release Checks

```bash
pytest
python scripts/release_check.py
python scripts/deployment_check.py
```

## Project Structure

```text
ShopMind-AI/
├── data/                  # Demo data and the fixed LLM-agent evaluation set
├── docs/                  # Product, metric, experiment, and architecture documentation
├── scripts/               # Data preparation, evaluation, and batch scripts
├── src/shopmind/          # Analytics, retrieval, agent, API, and dashboard code
└── tests/                 # Algorithm, API, grounding, fallback, and release tests
```

## Metric Integrity

This README reports only results produced by committed scripts and saved artifacts. It does not claim unmeasured online lift or pre-fill unsupported improvement percentages. LLM metrics depend on the fixed evaluation set, model version, prompt, and API conditions recorded at evaluation time.
