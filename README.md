# Monty 2.0

Monty is an automated sourcing and recommendation system for Montage Ventures, a seed-stage VC firm. It discovers early-stage founders, filters them through a quality pipeline, and surfaces the best ones via weekly email updates and on-demand recommendation emails.

## What it does

**Founder sourcing** — Queries the Aviato API daily for LinkedIn profiles matching search criteria, enriches them with company data, and stores them in PostgreSQL.

**Quality filtering** — Before a founder reaches any recommendation, they pass through several gates:
- Must have "founder" or "co-founder" literally in their current title (CEO-only, founding engineer, EIR etc. are excluded)
- LLM pedigree check: prior experience at top-tier big tech, a $100M+ funded startup, or a prestigious research lab/university. Internships at these companies count. The same check also flags clear non-startups (consultancies, funds, agencies) — ghost/stealth companies are allowed through
- Company must not have raised Series A or later
- Company must have been founded within the last 2 years
- Profiles that fail are kept in the DB with `pedigree_passes = false` so mistakes can be manually corrected

**Enrichment** — After ingestion, `add_monty_data` runs on new profiles to add: product/market description, funding formatting, repeat founder detection, technical founder detection, company/school tags, and verticals.

**Taste tree classification** — Companies are classified into Montage's investment category hierarchy (stored in `data/taste_tree.json`) via an LLM call. This drives which user sees which recommendation.

**Weekly update email** — Sent to the Montage team with three sections: recent early-stage deals (sourced from newsletter emails), portfolio/pipeline tracking updates, and top new founder profiles ranked by the XGBoost ranker model.

**Recommendation emails** — Per-user emails with founder recommendations matched to each person's assigned taste tree categories.

**Deal processing** — Parses funding newsletters from Gmail (VCNewsBriefing, Fortune TermSheet, Fresh Funding, Crunchbase Daily Digest) into structured deal data.

**Company tracking** — Monitors pipeline companies for updates (funding, hires, product launches) via web scraping.

**Slack bot** — Interactive assistant for ad-hoc founder/deal queries.

## Tech stack

- **Language**: Python 3
- **Database**: PostgreSQL via Supabase
- **ML**: XGBoost ranker model for scoring and ranking founders
- **AI**: OpenAI (gpt-4o-mini) for pedigree checks, descriptions, deal synthesis, tree classification; Groq as fallback
- **APIs**: Aviato (founder/company data), Gmail API (deal emails + sending), Slack SDK, Notion (pipeline tracking), Firecrawl (tracking scraping), Exa/Parallel Search (discovery)
- **Deployment**: Railway (Procfile), scheduled via `schedule` library

## Environment variables

```
AVIATO_KEY
OPENAI_API_KEY
GROQ_API_KEY
DATABASE_URL
SUPABASE_URL
SUPABASE_KEY
GOOGLE_CREDENTIALS_BASE64
GOOGLE_TOKENS_BASE64
SLACK_BOT_TOKEN
SLACK_APP_TOKEN
PARALLEL_API_KEY
NOTION_API_KEY
FIRECRAWL_API_KEY
```

Optional:
```
CRON_ONLY=true          # skip Slack bot, run cron jobs only (Railway mode)
LOG_LEVEL=INFO          # default INFO, set WARNING to reduce noise
RUN_INITIAL_PROCESSING  # run processing pipeline on startup
RUN_INITIAL_DISCOVERY   # run discovery pipeline on startup
TREE_SOURCE=supabase    # load taste tree from Supabase instead of JSON
```

## Scheduled jobs

- **Daily 6 AM PDT**: `process_profiles_aviato` → `add_monty_data` → `add_tree_analysis`
- **Daily 10 PM PDT**: `aviato_discover`

## Project structure

```
main.py                        # Entry point — Slack bot + cron scheduler
workflows/
  aviato_processing.py         # Core ingestion: Aviato API → DB, pedigree check, enrichment
  weekly_update.py             # Weekly email assembly and sending
  recommendations.py           # Per-user recommendation emails
  profile_recommendations.py   # Profile fetching and ranking for weekly update
  deals.py                     # Deal ingestion from Gmail newsletters
  tracking.py                  # Pipeline company monitoring
services/
  database.py                  # PostgreSQL connection and query helpers
  openai_api.py                # OpenAI wrapper (descriptions, synthesis, scoring)
  groq_api.py                  # Groq wrapper with OpenAI fallback
  profile_analysis.py          # Founder detection, technical check, feature extraction
  tree.py                      # Taste tree loading, classification, funding filter
  weekly_formatting.py         # HTML email generation
  google_client.py             # Gmail auth and sending
  slack_bot.py                 # Slack bot handler
  notion.py                    # Notion pipeline integration
  ranker_inference.py          # XGBoost model inference
  model_loader.py              # Model loading utility
data/
  models/                      # Trained XGBoost ranker model
  taste_tree.json              # Investment category hierarchy
  deal_data/                   # Processed deal CSVs
  recommended_profiles.json    # Dedup tracking for sent recommendations
migrations/                    # DB schema migrations (run manually via psql)
```

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
python main.py
```

To run just the processing pipeline without the Slack bot:
```bash
CRON_ONLY=true python main.py
```

To trigger workflows directly:
```bash
python workflows/weekly_update.py
python workflows/aviato_processing.py
```
