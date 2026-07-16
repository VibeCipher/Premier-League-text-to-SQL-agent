import os
import re
import sqlite3
import time

import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="PL Data Agent ⚽",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------
# THEME (Premier League purple + pitch green)
# ---------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #2d0140 0%, #37003c 40%, #04010a 100%);
        color: #f5f5f5;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00ff85, #04f5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #c9c9d8;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: #1c0028 !important;
        color: #fff !important;
        border: 1px solid #00ff85 !important;
        border-radius: 10px !important;
    }
    .stChatMessage {
        background-color: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 0.6rem 1rem;
        border: 1px solid rgba(0,255,133,0.15);
    }
    .stDataFrame {
        border: 1px solid #00ff85;
        border-radius: 10px;
    }
    .pill {
        display:inline-block;
        background: rgba(0,255,133,0.12);
        color:#00ff85;
        border:1px solid #00ff85;
        padding:2px 10px;
        border-radius:20px;
        font-size:0.75rem;
        margin-right:6px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c0028, #04010a);
        border-right: 1px solid rgba(0,255,133,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚽ Premier League Data Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask questions in plain English. The agent writes the SQL, '
    'queries live 2024-25 EPL data, and answers.</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<span class="pill">804 players</span>'
    '<span class="pill">20 teams</span>'
    '<span class="pill">27,605 gameweek records</span>',
    unsafe_allow_html=True
)
st.write("")

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗂️ Schema")
    st.markdown("""
**teams** — name, strength ratings, W/D/L, league position

**players** — position, price, total points, goals, assists, ICT index

**gameweek_stats** — per-player, per-gameweek performance (27.6k rows)
    """)
    st.markdown("---")
    st.markdown("### 💡 Try asking")
    example_qs = [
        "Top 5 forwards by goals scored",
        "Which team has the best home attacking strength?",
        "Cheapest defenders with over 100 points",
        "Highest single gameweek score this season",
        "Compare Arsenal and Liverpool clean sheets",
    ]
    for q in example_qs:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q
    st.markdown("---")
    model_choice = st.selectbox(
        "Model (Groq — default, free)",
        [
            "openai/gpt-oss-120b (recommended — strongest free reasoning)",
            "llama-3.1-8b-instant (fastest, highest free limits)",
            "llama-3.3-70b-versatile (strong, lower daily limit)",
        ],
        index=0
    )

    st.markdown("---")
    with st.expander("🔑 Use your own API key instead (optional)"):
        st.caption("Leave blank to keep using Groq above. Fill this in to route through a different provider instead.")
        PROVIDER_CONFIG = {
            "Google Gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "default_model": "gemini-2.0-flash"},
            "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "default_model": "meta-llama/llama-3.3-70b-instruct:free"},
            "Cerebras": {"base_url": "https://api.cerebras.ai/v1", "default_model": "llama3.3-70b"},
            "Mistral": {"base_url": "https://api.mistral.ai/v1", "default_model": "mistral-small-latest"},
        }
        custom_provider = st.selectbox("Provider", list(PROVIDER_CONFIG.keys()))
        custom_key = st.text_input("API key", type="password", key="custom_api_key")
        custom_model = st.text_input("Model", value=PROVIDER_CONFIG[custom_provider]["default_model"])
        custom_base_url = PROVIDER_CONFIG[custom_provider]["base_url"]

# ---------------------------------------------------------------
# AGENT SETUP (cached so it's not rebuilt every rerun)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_agent(provider: str, model_name: str, api_key: str = None, base_url: str = None):
    db = SQLDatabase.from_uri("sqlite:///fpl.db")
    if provider == "groq":
        llm = ChatGroq(model=model_name, temperature=0)
    else:
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
    return llm, db

if custom_key:
    llm, db = get_agent("custom", custom_model, custom_key, custom_base_url)
else:
    if "gpt-oss-120b" in model_choice:
        model_name = "openai/gpt-oss-120b"
    elif "8b" in model_choice:
        model_name = "llama-3.1-8b-instant"
    else:
        model_name = "llama-3.3-70b-versatile"

    if not os.environ.get("GROQ_API_KEY"):
        try:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
        st.error("GROQ_API_KEY not set. Add it as an environment variable (local) or in Streamlit secrets (deployed).")
        st.stop()

    llm, db = get_agent("groq", model_name)
SCHEMA = db.get_table_info() + """

IMPORTANT DATA NOTES — read carefully, these prevent common mistakes:

1. TABLE GRAIN:
   - `players` = ONE ROW PER PLAYER, already containing full SEASON TOTALS (goals_scored, assists, total_points, minutes, etc). Never GROUP BY or SUM anything in this table for season totals — just SELECT, ORDER BY, LIMIT directly on the existing columns.
   - `gameweek_stats` = ONE ROW PER PLAYER PER GAMEWEEK. Only use this table for per-gameweek/per-round questions, trends across weeks, or fields not present in `players` (opponent_team, was_home, GW/round).
   - `teams` = ONE ROW PER TEAM, season-level team info (strength, league position, W/D/L). Does NOT contain clean_sheets — that only exists per-player-per-gameweek in gameweek_stats.

2. NEVER GROUP BY first_name OR second_name ALONE in the `players` table. Multiple different players can share the same first or last name (e.g. two different players both named "Mohamed") — grouping by name merges unrelated players into one fake combined total. If grouping/joining is ever needed, always use player_id (the true unique key), and only add second_name/web_name for display in the SELECT.

3. Always display players using `web_name` (the short, unique display name already in the data) rather than first_name or second_name alone, to avoid ambiguity.

4. TEAM-LEVEL MATCH EVENTS (clean sheets, goals conceded, wins) computed from gameweek_stats are duplicated once per player who played that match. To count these per team correctly, use COUNT(DISTINCT GW) filtered by team_name and the condition — NEVER SUM these columns directly, or you'll overcount by the number of players who played.

5. When a question doesn't name specific teams/players, don't filter to just one — return all relevant rows (GROUP BY with no WHERE, or ORDER BY + LIMIT only if "top N" is asked).

6. `now_cost` in `players` is already in millions (e.g. 5.5 means £5.5m) — don't divide or multiply it further.

7. Position values in `players` are one of: GK, DEF, MID, FWD, MNG (MNG = assistant manager, a newer FPL category — usually irrelevant unless asked about specifically).

8. Double check before finalizing: does the query's GROUP BY/WHERE actually match every distinct entity the question is asking about? If the question names 2+ specific things (players, teams), the query must return one row per each of them, not fewer.
"""


def invoke_with_retry(prompt: str, max_retries: int = 2):
    """Call the LLM, auto-retrying on transient 429s using the provider's suggested wait time."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait_match = re.search(r"retry_after_seconds['\"]?\s*:\s*([\d.]+)", err_str)
                wait_s = float(wait_match.group(1)) + 1 if wait_match else 5
                if attempt < max_retries:
                    time.sleep(min(wait_s, 30))
                    continue
            raise
    raise last_err


def clean_sql(text: str) -> str:
    """Strip markdown fences / stray text and pull out the SQL statement."""
    text = text.strip()
    text = re.sub(r"^```sql\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"(SELECT.*?;)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.rstrip(";") + ";"


def generate_sql(question: str) -> str:
    prompt = f"""You are a SQLite expert. Given this schema:

{SCHEMA}

Write ONE valid SQLite query to answer the question below.
Rules:
- Return ONLY the SQL query, no explanation, no markdown fences.
- ALWAYS include the actual metric/column the question asks about in the SELECT (e.g. if asked for "top scorers", SELECT the name AND the goals_scored column, not just the name).
- Use LIMIT when the question implies "top N".
- If the question compares two or more NAMED entities (e.g. "compare Arsenal and Liverpool"), the WHERE clause MUST include every single one of them (use IN (...) or OR) — never filter to just one.
- End the query with a semicolon.

Question: {question}
SQL:"""
    response = invoke_with_retry(prompt)
    return clean_sql(response.content)


def explain_result(question: str, sql: str, df: pd.DataFrame) -> str:
    preview = df.head(15).to_string(index=False)
    prompt = f"""Question: {question}
SQL used (check its WHERE/filter conditions): {sql}
Query result table:
{preview}

Write a short direct answer (1-3 sentences) using the EXACT values from the table above.
Every name mentioned must be paired with its exact number from the table. Never write "unknown" — the numbers are in the table above, read them.
If the SQL filters by position, team, or any other condition, mention that scope explicitly in the answer (e.g. "among forwards", "for Arsenal", "with over 500 minutes played") so the answer is self-explanatory and can't be confused with a differently-scoped question.
Don't mention SQL syntax itself."""
    response = invoke_with_retry(prompt)
    return response.content.strip()


def sql_has_unsafe_grouping(sql: str) -> bool:
    """Catch the exact bug we've seen: GROUP BY on first_name/second_name alone in players table (causes name-collision merges)."""
    return bool(re.search(r"group\s+by\s+.*\b(first_name|second_name)\b", sql, re.IGNORECASE)) and "players" in sql.lower()


def ask_pipeline(question: str):
    """Returns (answer_text, sql_used, result_df) — retries once on SQL error or unsafe grouping."""
    sql = generate_sql(question)

    if sql_has_unsafe_grouping(sql):
        fix_prompt = f"""This query groups by first_name or second_name, which is unsafe because different players can share a name:
{sql}

Rewrite it to remove any GROUP BY on first_name/second_name. If grouping is genuinely needed, group by player_id instead. If no grouping is needed at all (e.g. just ranking individual player rows), remove the GROUP BY entirely.
Return ONLY the corrected SQL."""
        response = invoke_with_retry(fix_prompt)
        sql = clean_sql(response.content)

    try:
        df = run_sql(sql)
    except Exception as e:
        # one retry: tell the model what went wrong
        retry_prompt = f"""The query below failed against this schema:
{SCHEMA}

Query: {sql}
Error: {e}

Write a corrected SQLite query. Return ONLY the SQL, no explanation."""
        response = invoke_with_retry(retry_prompt)
        sql = clean_sql(response.content)
        df = run_sql(sql)  # let it raise if it fails again

    answer = explain_result(question, sql, df)
    return answer, sql, df

# ---------------------------------------------------------------
# HELPER: run a raw SQL query directly
# ---------------------------------------------------------------
def run_sql(query: str) -> pd.DataFrame:
    conn = sqlite3.connect("fpl.db")
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df

# ---------------------------------------------------------------
# CHAT STATE
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("df") is not None:
            st.dataframe(msg["df"], use_container_width=True)
        if msg.get("chart_df") is not None:
            st.bar_chart(msg["chart_df"])

# ---------------------------------------------------------------
# INPUT (chat box + sidebar quick-question buttons)
# ---------------------------------------------------------------
question = st.chat_input("Ask about players, teams, or gameweek performance...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "df": None, "chart_df": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Writing SQL and querying the data..."):
            try:
                answer, sql_used, df = ask_pipeline(question)

                chart_df = None
                if df is not None and df.shape[1] == 2 and df.shape[0] >= 2:
                    num_col = df.select_dtypes(include="number").columns
                    cat_col = [c for c in df.columns if c not in num_col]
                    if len(num_col) == 1 and len(cat_col) == 1:
                        chart_df = df.set_index(cat_col[0])[num_col[0]]

                st.markdown(answer)
                if sql_used:
                    with st.expander("🔍 SQL used"):
                        st.code(sql_used, language="sql")
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)
                if chart_df is not None:
                    st.bar_chart(chart_df)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "df": df,
                    "chart_df": chart_df,
                })
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
                    err = "⚠️ Possible rate limit or quota issue with the selected model/provider."
                else:
                    err = "Something went wrong."
                st.error(err)
                with st.expander("🔍 Raw error (for debugging)"):
                    st.code(f"{type(e).__name__}: {e}")
                st.session_state.messages.append({"role": "assistant", "content": err, "df": None, "chart_df": None})
