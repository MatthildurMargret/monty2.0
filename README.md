# Monty 2.0

Automated founder sourcing and recommendation system for Montage Ventures. Discovers early-stage founders via Aviato, filters them through a quality pipeline, and surfaces the best ones in weekly email updates.

## How it works

Founders are ingested daily from Aviato, run through pedigree and startup-type checks (Claude API), and stored in PostgreSQL. Recommendation queries then filter by pedigree, funding stage (pre-Series A only), and company age (< 2 years). Surviving profiles are ranked by an XGBoost model and included in the weekly update email alongside recent deals and pipeline tracking updates.

## Running

```bash
pip install -r requirements.txt
# set env vars (see below)
python main.py
```

Set `CRON_ONLY=true` to skip the Slack bot (Railway mode).

## Key env vars

`AVIATO_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_KEY`, `GOOGLE_CREDENTIALS_BASE64`, `GOOGLE_TOKENS_BASE64`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`
