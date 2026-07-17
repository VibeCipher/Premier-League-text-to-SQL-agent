# ⚽ Premier League Data Agent

A natural-language-to-SQL agent for exploring real 2024-25 EPL data. Ask a question in
plain English, and the agent writes SQL, runs it, and answers — no query-writing required.

**Live demo:** _(add your Streamlit Cloud link here after deploying)_

## What it does

- Ask questions like *"top 5 forwards by goals scored"* or *"compare Arsenal and Liverpool clean sheets"*
- The agent (LangChain + Groq's free Llama/GPT-OSS models) writes real SQLite queries against the data
- Returns a natural-language answer, a data table, and an auto-generated chart when relevant
- Shows the exact SQL used for full transparency

## Data

Sourced from the [Fantasy Premier League dataset](https://github.com/vaastav/Fantasy-Premier-League)
(2024-25 season): 804 players, 20 teams, 27,605 gameweek-level performance records.

## Stack

- **LangChain** + **Groq** (free tier: Llama 3.1 8B, Llama 3.3 70B, GPT-OSS 120B)
- **Streamlit** for the UI
- **SQLite** for the data
- Optional: bring your own key for Google Gemini, OpenRouter, Cerebras, or Mistral instead

## About the LLM options

Groq is the default because its free tier is generous and fast enough for this kind of
back-and-forth SQL generation. That said, free-tier models (Groq's smaller ones especially)
can occasionally get a query wrong — wrong table, wrong aggregation, that kind of thing.
None of this is bulletproof, and that's expected with free models.

There's also an option in the sidebar to plug in your own key for Google Gemini, OpenRouter,
Cerebras, or Mistral instead, if you want to compare accuracy or work around a rate limit.
Heads up: some free-tier models (especially OpenRouter's `:free` slugs) get temporarily
rate-limited by high demand across all users, not just you — the app retries automatically
on that, but it's worth knowing if a query is slow to respond.

This is an open project — swap in a different model, tune the prompts, add a new provider,
whatever gets better predictions. The prompt logic lives in `app.py` (`generate_sql`,
`explain_result`) if you want to tweak how it reasons.

## Development notebook

See `notebooks/` for the Colab notebook walking through how the agent was built and tested,
step by step.

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"   # get one free at console.groq.com
streamlit run app.py
```

## Deploy

Deployed free on Streamlit Community Cloud, connected to this repo. Add `GROQ_API_KEY`
under the app's Secrets settings (not committed to the repo).
