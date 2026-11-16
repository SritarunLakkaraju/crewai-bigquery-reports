# CrewAI Weekly Business Report on a BigQuery-style Warehouse

An analyst agent writes guarded SQL. Every query is checked read-only, dry-run against a byte budget, and
row-limited before it runs. A researcher agent pulls market notes, and a writer agent produces a Markdown report
with charts. The writer task's CrewAI guardrail is a numeral guard: any number not found in the query results
sends the report back for a rewrite. DuckDB stands in for BigQuery; `BigQueryWarehouse` is the real adapter.

```bash
python -m fde3_crewai_bigquery.data_gen    # data/warehouse.duckdb (6k users, 9.3k orders, 18.5k items)
python -m fde3_crewai_bigquery.evaluate    # SQL accuracy, guardrails, % numbers verified
python -m fde3_crewai_bigquery.deliver     # outbox/<week>/report.md + charts + report.eml
```
Cloud Scheduler / Cloud Run Job commands are in `deliver.py`.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m fde3_crewai_bigquery.data_gen
python -m fde3_crewai_bigquery.evaluate
pytest -q
```

Everything runs **offline**: no API keys or cloud account are needed. The steps that would use an LLM have
deterministic offline implementations, so tests are reproducible and answers cannot be hallucinated.

## Results (synthetic, seeded data)

- Text-to-SQL test set: **50/50** result sets match independently written gold SQL (a regression suite written with the parser, not a generalisation claim).
- Unsafe / over-budget queries rejected: 6/6.
- Report: 43 numbers, **100.0% verified** against query results; an injected invented number is caught by the task guardrail and removed on rewrite.
- 12 queries, 4,915,619 bytes scanned per report (≈ $0.000028 at BigQuery on-demand pricing).

## Switching to the real services

| Environment variable | Effect |
|---|---|
| `GEMINI_API_KEY` (+ `GEMINI_MODEL`) | use Gemini for the LLM steps |
| `GOOGLE_CLOUD_PROJECT` + `FDE_USE_VERTEX=1` | Vertex AI text-embedding-005 instead of the offline hashing embedder |
| `FDE_VECTOR_BACKEND=chroma` or `pgvector` (+ `PGVECTOR_DSN`) | Chroma or PostgreSQL/pgvector instead of the NumPy vector store |

The cloud adapters are written against the official SDKs but have not been run against live services yet.
Deploy to Cloud Run with the included `Dockerfile` (`gcloud run deploy --source .`).

## Layout

- `fde3_crewai_bigquery/`: the project (see the module docstrings; each file explains its role)
- `common/`: shared tokeniser + BM25, hashing embedder / Vertex embedder, vector stores (NumPy, Chroma, pgvector), Gemini client
- `docs/FDE_Projects_Guide.pdf`: a detailed write-up of this project and its five siblings

## Part of a series of six Forward Deployed Engineer projects

- [Agentic RAG Assistant on Google ADK](https://github.com/SritarunLakkaraju/agentic-rag-adk)
- [DealMind v2: Sales Agent on LangGraph](https://github.com/SritarunLakkaraju/dealmind-langgraph)
- [Event-driven Invoice Intelligence Pipeline](https://github.com/SritarunLakkaraju/invoice-intelligence-pipeline)
- [Vector Search from Scratch: HNSW vs FAISS vs Chroma](https://github.com/SritarunLakkaraju/hnsw-vector-search)
- [Customer Support Agent for a Small Business (LangGraph)](https://github.com/SritarunLakkaraju/support-agent-langgraph)
