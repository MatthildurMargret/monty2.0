# Monty 2.0

Monty is an AI-powered assistant for Montage Ventures, an early-stage VC firm investing across fintech, healthcare, and commerce. Monty helps the team discover founders, analyze deals, track pipeline companies, and generate weekly investment updates.

## Features

- **Slack Bot Integration**: Interactive AI assistant accessible via Slack for real-time queries
- **Founder Discovery**: Automated discovery and analysis of founders using Aviato search
- **Deal Processing**: Processes and enriches funding deal data with AI-powered insights
- **Pipeline Management**: Tracks portfolio and pipeline companies with investment stage classification
- **Weekly Updates**: Generates automated weekly summaries of deals, recommendations, and market insights
- **Market Map**: Maintains a hierarchical tree structure of investment sectors, categories, and companies
- **ML-Powered Recommendations**: Uses XGBoost models for company classification and ranking
- **Profile Recommendations**: AI-driven recommendations for founder and company profiles

## Tech Stack

- **Language**: Python 3
- **ML/AI**: XGBoost, scikit-learn, OpenAI API, Groq API, Google Generative AI
- **Database**: PostgreSQL (via Supabase)
- **APIs**: Slack SDK, Google APIs, OpenAI Agents SDK
- **Data Processing**: pandas, numpy, pyarrow
- **Web Scraping**: BeautifulSoup4, Playwright
- **Deployment**: Railway (configured via Procfile)

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database (or Supabase)
- API keys for:
  - Slack (Bot Token & App Token)
  - OpenAI
  - Groq
  - Google APIs
  - SerpAPI (for web search)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd monty2.0-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_APP_TOKEN=your_slack_app_token
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
DATABASE_URL=your_postgresql_connection_string
# ... other API keys and configuration
```

4. Run the application:
```bash
python main.py
```

## Project Structure

```
monty2.0-main/
├── main.py                 # Application entry point
├── services/               # Core service modules
│   ├── slack_bot.py       # Slack bot implementation
│   ├── ai_scoring.py      # AI-powered scoring functions
│   ├── deal_processing.py # Deal data processing
│   ├── tree_tools.py      # Market map tree operations
│   └── ...
├── workflows/             # Business logic workflows
│   ├── weekly_update.py  # Weekly update generation
│   ├── deals.py          # Deal processing workflow
│   ├── recommendations.py # Recommendation system
│   └── ...
├── tests/                 # Test scripts and utilities
├── data/                  # Data files and models
│   ├── models/           # Trained ML models
│   ├── deal_data/        # Deal datasets
│   └── ...
└── config/               # Configuration files
```

## Usage

### Slack Bot

Once running, interact with Monty directly in Slack. Monty can:
- Search for founders and companies
- Analyze funding deals and trends
- Access portfolio and pipeline information
- Query the market map by sector
- Generate insights about companies and markets

### Cron Jobs

The application runs scheduled tasks:
- **Daily at 6 AM PDT**: Aviato profile processing pipeline
- **Daily at 10 PM PDT**: Founder discovery pipeline

### Workflows

Individual workflows can be run independently:
- `workflows/weekly_update.py`: Generate weekly investment updates
- `workflows/deals.py`: Process and enrich deal data
- `workflows/recommendations.py`: Generate company/founder recommendations

## Configuration

Key environment variables:
- `CRON_ONLY`: Run only cron jobs without Slack bot
- `LOG_LEVEL`: Set logging level (INFO, WARNING, ERROR)
- `RUN_INITIAL_PROCESSING`: Run processing pipeline on startup
- `RUN_INITIAL_DISCOVERY`: Run discovery pipeline on startup


