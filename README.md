# Campaign QA-First AI Workflow

This repository contains a small Streamlit application for campaign data intake, QA, and AI-assisted reporting.

The app lets you upload campaign CSVs, standardizes their schema, merges them into one dataset, runs rule-based QA checks, and generates optional summaries using either a mock LLM provider or live Gemini/OpenAI calls.

## What the app does

The workflow in [app.py](/Users/shami/Desktop/hecakthon-ai%20workflow/app.py) is:

1. Accept up to four CSV uploads from the sidebar:
   - impressions
   - visits
   - search
   - web analytics
2. Fall back to generated dummy data when uploads are missing and dummy mode is enabled.
3. Standardize key columns such as `date` and `store_name`.
4. Derive simple metrics like `ctr` or `clicks` when one is missing.
5. Merge all datasets into a unified table.
6. Run QA rules against the merged data.
7. Show data previews, QA findings, missingness, and optional AI-generated summaries.

## Repository layout

- [app.py](/Users/shami/Desktop/hecakthon-ai%20workflow/app.py): Streamlit entrypoint and UI
- [requirements.txt](/Users/shami/Desktop/hecakthon-ai%20workflow/requirements.txt): Python dependencies
- [debug_data.py](/Users/shami/Desktop/hecakthon-ai%20workflow/debug_data.py): quick debug script for merge behavior
- [utils/data_loader.py](/Users/shami/Desktop/hecakthon-ai%20workflow/utils/data_loader.py): upload loading and dummy-data fallback
- [utils/dummy_data.py](/Users/shami/Desktop/hecakthon-ai%20workflow/utils/dummy_data.py): synthetic dataset generator
- [utils/standardize.py](/Users/shami/Desktop/hecakthon-ai%20workflow/utils/standardize.py): schema normalization, derived metrics, merging, missingness report
- [utils/qa_rules.py](/Users/shami/Desktop/hecakthon-ai%20workflow/utils/qa_rules.py): rule-based QA engine
- [utils/ai_workflow.py](/Users/shami/Desktop/hecakthon-ai%20workflow/utils/ai_workflow.py): mock and live LLM providers plus report-stage orchestration

## Requirements

- Python 3.12 is present in this workspace
- Packages from [requirements.txt](/Users/shami/Desktop/hecakthon-ai%20workflow/requirements.txt):
  - `streamlit`
  - `pandas`
  - `numpy`
  - `plotly`
  - `google-generativeai`
  - `scikit-learn`
  - `openai`

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Using the app

### Sidebar options

- `Campaign Name`: label shown in the page title
- `Use dummy data if uploads missing`: use generated sample data when no uploads are provided
- `Use Real AI`: switch from the built-in mock provider to Gemini or OpenAI
- `Select Provider`: choose `Gemini` or `OpenAI (ChatGPT)`
- `API Key`: enter the provider key directly in the sidebar

### Main tabs

- `Data Preview`: shows each raw dataset
- `Unified Dataset`: shows the merged table and missingness report
- `QA Center`: shows QA findings and lets you run focused AI summaries
- `AI Summary`: runs the multi-stage AI reporting workflow

### Download outputs

The app can export:

- `unified_dataset.csv`
- `qa_results.csv`
- `ai_report.md`

## Expected CSV inputs

The loader accepts arbitrary CSVs, but the merge and QA logic expect fields that can be normalized into this schema.

| Dataset | Core fields |
| --- | --- |
| Impressions | `date`, `store_name`, `impressions`, `clicks`, `ctr` |
| Visits | `date`, `store_name`, `total_visits`, `exposed_visits` |
| Search | `date`, `store_name`, `search_metric` |
| Web | `date`, `store_name`, `web_sessions`, `web_conversions` |

The standardization layer in [utils/standardize.py](/Users/shami/Desktop/hecakthon-ai%20workflow/utils/standardize.py) recognizes several aliases, including:

- `store`, `store name`, `shop`, `outlet` -> `store_name`
- `day`, `time`, `period` -> `date`
- `impression`, `imps` -> `impressions`
- `visits` -> `total_visits`
- `exposed` -> `exposed_visits`
- `sessions`, `page views` -> `web_sessions`

Store names are uppercased and stripped, and `date` values are parsed with `pandas.to_datetime(..., errors="coerce")`.

## QA checks

The rule engine in [utils/qa_rules.py](/Users/shami/Desktop/hecakthon-ai%20workflow/utils/qa_rules.py) currently performs these checks:

- `FAIL`: `exposed_visits > total_visits`
- `FAIL`: exposed share outside `5%` to `20%`
- `WARN`: exposed share outside `10%` to `14%`
- `WARN`: low-impressions / high-exposed mismatch by store quartiles
- `WARN`: `ctr > 10%` or `ctr < 0.1%`
- `FAIL`: any negative numeric value
- `WARN`: total visits above `2.5x` the store median
- `WARN`: more than `30%` of expected days missing

The UI rolls these into a run-level status of `PASS`, `WARN`, or `FAIL`.

## AI workflow

The AI layer in [utils/ai_workflow.py](/Users/shami/Desktop/hecakthon-ai%20workflow/utils/ai_workflow.py) supports three modes:

- `MockLLMProvider`: local placeholder summaries, no API key required
- `GeminiLLMProvider`: live calls through `google-generativeai`
- `OpenAILLMProvider`: live calls through the `openai` Python SDK

The staged workflow currently generates:

- Stage A: QA Summary
- Stage B: Performance Contributors
- Stage C: Store Drilldown
- Stage E: Final Executive Report

There are also separate AI actions for:

- feasibility analysis
- missingness summary
- QA summary

## Notes from the code analysis

These are current implementation details worth knowing before you rely on the app:

- The UI label says "Use dummy data if uploads missing", but the code only uses dummy data when all uploads are missing. Partial-upload fallback is not implemented.
- If datasets do not share `date`, the merge falls back to the intersection of available keys. When every dataset lacks `date`, merging on `store_name` alone can create a very large Cartesian-style expansion. Running [debug_data.py](/Users/shami/Desktop/hecakthon-ai%20workflow/debug_data.py) in this workspace produced a merged dataset with `4,617,605` rows from dummy inputs after dropping `date`.
- The AI workflow skips any Stage D logic; only A, B, C, and E are generated.
- There is no automated test suite in the repository at the moment.

## Quick smoke check performed

I verified that:

- `python3 --version` reports `Python 3.12.6`
- `python3 debug_data.py` runs successfully in this workspace

## Next improvements

If you want to keep building this project, the highest-value follow-ups are:

1. Implement partial dummy fallback for missing upload slots.
2. Guard the merge path against row explosion when join keys are incomplete.
3. Add tests for schema normalization, merge behavior, and QA rules.
4. Move API key handling to environment variables or Streamlit secrets for safer operation.
