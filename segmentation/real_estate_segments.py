# app.py
import os
import datetime
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Rental Real Estate Segmentation Expert", page_icon="🏠", layout="wide")
st.title("🏠 Rental Property Market Segmentation Expert")
st.caption("Quantify TAM/SAM in the US and produce 5–6 targetable segments with channels. Uses OpenAI Responses API.")

# ---------------- Sidebar: API & Model ----------------
with st.sidebar:
    st.header("🔐 API & Model")
    api_key_input = st.text_input("OpenAI API Key", type="password", help="Only used in this session.")
    effective_api_key = api_key_input or os.getenv("OPENAI_API_KEY", "")

    model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "o4-mini"],
        index=0,
        help="gpt-4o-mini is a good price/performance default."
    )
    use_web_search = st.checkbox(
        "Enable built-in web_search tool",
        value=True,
        help="Lets the model pull current stats and cite sources. Requires newer OpenAI SDK."
    )
    high_effort = st.checkbox(
        "High reasoning effort",
        value=True,
        help="May improve synthesis for research-heavy tasks. Disabled automatically if unsupported."
    )

# ---------------- System Instructions ----------------
st.subheader("🧠 System instructions (edit if you like)")
default_system = """You are a Rental Property Real Estate Market Segmentation Coach.
Your job:
1) Quantify the US Total Addressable Market (TAM) and Serviceable Addressable Market (SAM) for individuals interested in buying a rental property anywhere in the US (exclude those renting out their primary residences).
2) Identify demographics & psychographics of real buyers.
3) Produce a targetable list of 5–6 distinct segments (clear, mutually exclusive where possible).
4) Recommend the best marketing channels (and why) for each segment.
5) Always cite recent, reputable sources inline and at the end.
6) Add the link to the report next to the citation.
7) Show data only from 2025 onwards.
8) Identify atleast 10 states & metro cities with the highest investor activity in the US.
9) Identify atleast 100 zipcodes with the highest investor activity in the US.
10)

Scope and evidence:
- Use web research and prefer data from these domains (in priority order). If paywalled, quote high-level stats only and still cite and provide the link:
  • investopedia.com
  • nar.realtor
  • bls.gov (include bls.gov/eag/)
  • zillow.com/research and zillow.com/research/data/
  • cotality.com/insights and corelogic.com/intelligence/
  • realtor.com/research and realtor.com/research/data/
  • redfin.com/news and redfin.com/news/data-center/
  • costar.com/news
  • altosresearch.com
  • roofstock.com/blog
  • mashvisor.com/blog
  • biggerpockets.com/blog
  • realwealth.com/learn/
  • attomdata.com/news/
  • yardimatrix.com/news
  • realestateconsulting.com/blog/
  • urban.org/expertise/housing-finance
  • consumerfinance.gov/data-research/

Methods:
- Triangulate TAM/SAM using: # of households/investor households, investment-purchase shares, mortgage origination volumes, affordability, investor share of purchases, and labor/income trends.
- State assumptions. Align figures to the most recent year possible and note the year (e.g., “as of 2025”).
- Output tight, skimmable bullet points + a final one-paragraph summary and a compact “Data Appendix” with sources.
"""
system_instructions = st.text_area("System instructions", value=default_system, height=220)

PREFERRED_DOMAINS = [
    "investopedia.com","nar.realtor","bls.gov","zillow.com",
    "cotality.com","corelogic.com","realtor.com","redfin.com",
    "costar.com","altosresearch.com","roofstock.com","mashvisor.com",
    "biggerpockets.com","realwealth.com","attomdata.com","yardimatrix.com",
    "realestateconsulting.com","urban.org","consumerfinance.gov"
]

def build_research_prompt(user_question: str) -> str:
    year = datetime.date.today().year
    domains = ", ".join(PREFERRED_DOMAINS)
    return f"""
Task: {user_question}

Research plan:
- Start with recency (2025–{year}). Prefer data with explicit dates and US scope.
- Prioritize these domains: {domains}.
- Collect: investor share of purchases, # of investor buyers, unit counts, rental demand indicators, income/employment (BLS), affordability/price-to-rent, mortgage/financing trends, and segmentation clues (age, income, life stage, portfolio size, intent).
- Then compute TAM and SAM with clear assumptions and show the math in-line.

Deliverables:
- 5–6 targetable segments (name, who they are, key needs, proxy targeting attributes).
- Channel recommendations per segment (paid search, social, YouTube, podcasts, newsletters, LinkedIn, BiggerPockets, forums, affiliates, lead-gen partners, etc.) with rationale.
- Cite specific pages inline like [Source: NAR 2025 Investor Report] and include a full citation list whereever the stats are quoted.
"""

default_question = (
    "Estimate the current US TAM and SAM for individuals planning to buy a rental property in the next 12–24 months. "
    "Produce 5–6 targetable segments with demographics/psychographics and the best marketing channels to reach each."
    " Include recent data from 2025 onwards and cite sources inline."
    "Identify at least 10 states and metro cities with the highest investor activity in the US."
)
user_question = st.text_area("Research question", value=default_question, height=140)

col1, col2 = st.columns([1,1])
with col1:
    run_btn = st.button("🔎 Run Research", type="primary")
with col2:
    clear_btn = st.button("🧹 Clear Output")

if clear_btn:
    st.session_state.pop("last_output", None)

# ---------------- OpenAI Call Helpers ----------------
def extract_text_from_responses(resp) -> str:
    # Works across SDK shapes
    if getattr(resp, "output_text", None):
        return resp.output_text
    chunks = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", "") == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") == "output_text":
                    chunks.append(c.text)
    return "\n".join(chunks) if chunks else str(resp)

def call_openai(effective_api_key: str, model: str, system_text: str, user_text: str,
                use_web: bool, high_reasoning: bool) -> str:
    client = OpenAI(api_key=effective_api_key)

    system_msg = {"role": "system", "content": [{"type": "text", "text": system_text}]}
    user_msg = {"role": "user", "content": [{"type": "text", "text": user_text}]}

    # First attempt: Responses API with optional web_search and reasoning
    kwargs = {"model": model, "input": [system_msg, user_msg]}
    if use_web:
        kwargs["tools"] = [{"type": "web_search"}]
    if high_reasoning:
        kwargs["reasoning"] = {"effort": "high"}

    try:
        resp = client.responses.create(**kwargs)
        return extract_text_from_responses(resp)
    except TypeError:
        # Remove possibly unsupported keys (older SDK)
        kwargs.pop("tools", None)
        kwargs.pop("reasoning", None)
        resp = client.responses.create(**kwargs)
        return extract_text_from_responses(resp)
    except Exception:
        # Final fallback: classic Chat Completions (no tools)
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        )
        return chat.choices[0].message.content

# ---------------- Run Button ----------------
if run_btn:
    if not effective_api_key:
        st.error("Please paste your OpenAI API key in the sidebar (or set OPENAI_API_KEY in your environment).")
    else:
        with st.spinner("Researching and compiling…"):
            try:
                prompt = build_research_prompt(user_question)
                output = call_openai(
                    effective_api_key,
                    model,
                    system_instructions,
                    prompt,
                    use_web_search,
                    high_effort
                )
                st.session_state["last_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- Output ----------------
if "last_output" in st.session_state and st.session_state["last_output"]:
    st.subheader("📄 Results")
    st.markdown(st.session_state["last_output"])
    st.download_button(
        "⬇️ Download Markdown",
        data=st.session_state["last_output"],
        file_name="real_estate_segments.md",
        mime="text/markdown",
    )

st.markdown("---")
with st.expander("ℹ️ Notes & Tips"):
    st.markdown(
        """
- **Avoid filename collisions**: Don’t name your file `streamlit.py` or `openai.py`.
- **web_search** requires a newer OpenAI SDK. If you see a tools-related error, uncheck it or `pip install -U openai`.
- **Privacy**: Your API key is kept in memory only for this session.
        """
    )
