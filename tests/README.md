# Monty Agent Testing

This folder contains test scripts to validate the Monty agent and tools functionality locally without requiring Slack.

## Test Files

### `test_agent.py`
Tests the complete agent with tools integration using the OpenAI Agent SDK.

**Usage:**
```bash
# Run automated test suite
python tests/test_agent.py

# Run interactive mode
python tests/test_agent.py --interactive
```

### `test_tools_direct.py`
Tests individual tools directly without the agent layer.

**Usage:**
```bash
python tests/test_tools_direct.py
```

## Prerequisites

1. **Environment Variables**: Ensure your `.env` file contains:
   ```
   DATABASE_URL=your_postgresql_url
   OPENAI_API_KEY=your_openai_key
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Database**: Ensure your PostgreSQL database is accessible and contains the expected tables:
   - `founders`
   - `investment_theses`

## Test Scenarios

The automated tests cover:
- Basic database queries
- Location-based searches
- Experience filtering
- Funding analysis
- Score-based filtering

## Troubleshooting

- **Database Connection Issues**: Check your `DATABASE_URL` in `.env`
- **OpenAI API Issues**: Verify your `OPENAI_API_KEY` is valid
- **Import Errors**: Ensure you're running from the project root directory

## Interactive Testing

Use interactive mode to test custom queries:
```bash
python tests/test_agent.py --interactive
```

Then type queries like:
- "How many founders are in the database?"
- "Find AI founders in San Francisco"
- "Show me high-scoring founders"
