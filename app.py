import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI

st.set_page_config(page_title="📊 KPI AI Dashboard", layout="wide")
st.title("📊 BackOffice Operations Dashboard with AI Insights")

# ---------------- Load CSV from Google Drive ----------------
file_id = "1mkVXQ_ZQsIXYnh72ysfqo-c2wyMZ7I_1"
file_url = f"https://drive.google.com/uc?export=download&id={file_id}"

try:
    df = pd.read_csv(file_url, dayfirst=True, parse_dates=["Start Date", "End Date", "Target Date"])
except Exception as e:
    st.error(f"❌ Failed to load CSV from Google Drive.\n\n**Error:** `{e}`")
    st.stop()

df["Start Date"] = pd.to_datetime(df["Start Date"], errors='coerce')
df["End Date"] = pd.to_datetime(df["End Date"], errors='coerce')
df["Target Date"] = pd.to_datetime(df["Target Date"], errors='coerce')

# ---------------- FILTER SECTION ----------------
st.sidebar.header("📂 Filters")

# 📅 Date range filter — defaults to full range
start_date, end_date = st.sidebar.date_input(
    "Select Week (Monday to Sunday)",
    [df["Start Date"].min(), df["Start Date"].min() + pd.Timedelta(days=6)]
)

# Ensure proper conversion to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Convert selected week into list of "Report Date" format
selected_labels = pd.date_range(start=start_date, end=end_date).strftime("%d-%b").tolist()

# Filter full data set by selected dates
filtered_df = df[(df["Start Date"] >= start_date) & (df["Start Date"] <= end_date)]

# 📁 Portfolio filter
if "Portfolio" in df.columns:
    portfolios = st.sidebar.multiselect("Filter by Portfolio", sorted(df["Portfolio"].dropna().unique()))
    if portfolios:
        filtered_df = filtered_df[filtered_df["Portfolio"].isin(portfolios)]

# 🎯 Source filter
if "Source" in df.columns:
    sources = st.sidebar.multiselect("Filter by Source", sorted(df["Source"].dropna().unique()))
    if sources:
        filtered_df = filtered_df[filtered_df["Source"].isin(sources)]

# ⚠️ Handle empty data
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters.")
    st.stop()

# ---------------- KPI RE-CALCULATION ----------------
min_date = filtered_df["Start Date"].min()
max_date = max(filtered_df["Start Date"].max(), filtered_df["End Date"].max(), filtered_df["Target Date"].max())
date_range = pd.date_range(start=min_date, end=max_date)

kpi_data = []
pend_rate_values = []
prev_closing_wip = filtered_df[(filtered_df["End Date"].isna()) & (filtered_df["Start Date"] <= min_date)].shape[0]

for report_date in date_range:
    opening_wip = prev_closing_wip
    received_today = filtered_df[filtered_df["Start Date"] == report_date]
    cases_received = received_today.shape[0]
    complete_today = filtered_df[filtered_df["End Date"] == report_date]
    cases_complete = complete_today.shape[0]
    complete_within_sla = complete_today[complete_today["End Date"] < complete_today["Target Date"]].shape[0]
    complete_within_sla_pct = f"{int(round((complete_within_sla / cases_complete * 100)))}%" if cases_complete > 0 else "0%"
    backlog_over_sla = filtered_df[
        (filtered_df["Start Date"] <= report_date) &
        (filtered_df["End Date"].isna()) &
        (filtered_df["Target Date"] < report_date)
    ].shape[0]
    backlog_pct = f"{int(round((backlog_over_sla / prev_closing_wip * 100)))}%" if prev_closing_wip > 0 else "0%"
    wip_in_sla = filtered_df[
        (filtered_df["Start Date"] <= report_date) &
        (filtered_df["End Date"].isna()) &
        (filtered_df["Target Date"] >= report_date)
    ].shape[0]
    closing_wip = opening_wip + cases_received - cases_complete
    wip_in_sla_pct = f"{int(round((wip_in_sla / closing_wip * 100)))}%" if closing_wip > 0 else "0%"

    pend_subset = filtered_df[filtered_df["Start Date"] <= report_date]
    pend_total = pend_subset["Pend Case"].notna().sum()
    pend_yes = pend_subset[pend_subset["Pend Case"].astype(str).str.lower() == "yes"].shape[0]
    pend_rate_val = int(round((pend_yes / pend_total * 100))) if pend_total > 0 else 0
    pend_rate = f"{pend_rate_val}%"

    pend_rate_values.append(pend_rate_val)

    kpi_data.append({
        "Report Date": report_date.strftime("%d-%b"),
        "Opening WIP": opening_wip,
        "Cases Received": cases_received,
        "Cases Complete": cases_complete,
        "Closing WIP": closing_wip,
        "Complete Within SLA": complete_within_sla,
        "Complete Within SLA %": complete_within_sla_pct,
        "Backlog - WIP Over SLA": backlog_over_sla,
        "Backlog %": backlog_pct,
        "WIP in SLA": wip_in_sla,
        "WIP in SLA %": wip_in_sla_pct,
        "Pend Rate": pend_rate
    })

    prev_closing_wip = closing_wip

# ✅ Use this filtered KPI dataset everywhere now
kpi_df = pd.DataFrame(kpi_data)

# ---------------- AI Insights Section ----------------
st.subheader("🧠 AI-Generated Insights")

# 🔍 WIP Analyzer (used for AI insights)
def analyze_wip_spikes(df_kpi, raw_df):
    df_kpi["Closing WIP Num"] = df_kpi["Closing WIP"]
    rolling_avg = df_kpi["Closing WIP Num"].rolling(window=3).mean()
    df_kpi["WIP Spike"] = df_kpi["Closing WIP Num"] > rolling_avg * 1.2

    spike_days = df_kpi[df_kpi["WIP Spike"] == True]["Report Date"].tolist()
    analysis = []

    for day in spike_days:
        day_raw = raw_df[raw_df["Start Date"].dt.strftime("%d-%b") == day]
        day_kpi = df_kpi[df_kpi["Report Date"] == day]

        pend_total = day_raw["Pend Case"].notna().sum()
        pend_yes = day_raw[day_raw["Pend Case"].astype(str).str.lower() == "yes"].shape[0]
        pend_rate = round((pend_yes / pend_total * 100), 1) if pend_total > 0 else 0

        pend_reason_counts = day_raw[day_raw["Pend Case"].astype(str).str.lower() == "yes"] \
            .groupby("Pend Reason").size().sort_values(ascending=False).to_dict()

        breakdown = {
            "Portfolio": day_raw["Portfolio"].value_counts().head(3).to_dict(),
            "Source": day_raw["Source"].value_counts().head(3).to_dict(),
            "Event Type": day_raw["Event Type"].value_counts().head(3).to_dict(),
            "Manual/RPA": day_raw["Manual/RPA"].value_counts().head(3).to_dict()
        }

        analysis.append({
            "date": day,
            "closing_wip": int(day_kpi["Closing WIP"].values[0]),
            "pend_rate": f"{pend_rate}%",
            "top_pend_reasons": pend_reason_counts,
            "breakdown": breakdown
        })

    return analysis


if st.button("Generate Insights with GPT"):
    with st.spinner("Analyzing and generating insights..."):
        deep_dive_insights = analyze_wip_spikes(kpi_df, filtered_df)  # 🔁 Updated here!

        client = OpenAI(api_key=st.secrets["openai_key"])

        story_prompt = f"""
You are a senior operations analyst with deep expertise in back-office performance analysis.

Below is structured data from a recent operational deep dive:

{json.dumps(deep_dive_insights, indent=2)}

Please review the data and return exactly **5 concise bullet points** that:

- Are clear, punchy, and no longer than 2 lines each
- Include relevant **metrics** (e.g. % pend rate, volume counts)
- Highlight key **issues**, **trends**, and **root causes**
- Use **markdown-style emphasis** (e.g. **bold**, emojis like 📉📈🛠️✅) for readability

Output format:
- 📌 **[Bold Insight]** – supporting number(s) and short explanation.
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in back-office operations and root cause analysis."},
                {"role": "user", "content": story_prompt}
            ],
            temperature=0.5
        )
        gpt_bullets = response.choices[0].message.content
        st.markdown(gpt_bullets)
        
# ---------------- CHARTS SECTION ----------------
st.markdown("## 📈 Operational Trends")

# ✅ Use only filtered KPI data within selected week
chart_df = kpi_df[kpi_df["Report Date"].isin(selected_labels)]

labels = list(chart_df["Report Date"])
cases_received = list(chart_df["Cases Received"])
cases_complete = list(chart_df["Cases Complete"])
closing_wip = list(chart_df["Closing WIP"])
wip_sla_pct = [int(x.replace('%', '')) if '%' in x else 0 for x in chart_df["WIP in SLA %"]]
complete_sla_pct = [int(x.replace('%', '')) if '%' in x else 0 for x in chart_df["Complete Within SLA %"]]

# Pend Rate values from filtered KPI df
pend_rate_chart = [
    int(row["Pend Rate"].replace('%', '')) if isinstance(row["Pend Rate"], str) else 0
    for _, row in chart_df.iterrows()
]

# Filter pend reasons for selected range
pend_reasons = filtered_df[filtered_df["Start Date"].dt.strftime("%d-%b").isin(labels)]
pend_reason_counts = pend_reasons["Pend Reason"].value_counts().to_dict()
pend_reason_labels = list(pend_reason_counts.keys())
pend_reason_values = list(pend_reason_counts.values())

# ---------------- Row 1: WIP vs Cases + SLA Trends ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📦 Cases Processed vs WIP")
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Cases Received', x=labels, y=cases_received, marker_color='red'))
    fig.add_trace(go.Bar(name='Cases Complete', x=labels, y=cases_complete, marker_color='blue'))
    fig.add_trace(go.Scatter(name='Closing WIP', x=labels, y=closing_wip, mode='lines+markers', line=dict(color='orange', width=3)))
    fig.update_layout(barmode='group', height=360)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### 🧾 SLA Compliance %")
    sla_fig = go.Figure()
    sla_fig.add_trace(go.Scatter(x=labels, y=wip_sla_pct, name='WIP in SLA %', line=dict(color='teal')))
    sla_fig.add_trace(go.Scatter(x=labels, y=complete_sla_pct, name='Complete SLA %', line=dict(color='purple')))
    sla_fig.update_layout(height=360)
    st.plotly_chart(sla_fig, use_container_width=True)

# ---------------- Row 2: Pie Chart + Pend Rate ----------------
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 🥧 Top Pend Reasons")
    pie_fig = px.pie(
        names=pend_reason_labels,
        values=pend_reason_values,
        title="Pend Reasons Breakdown",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    pie_fig.update_traces(textinfo='percent+label')
    st.plotly_chart(pie_fig, use_container_width=True)

with col4:
    st.markdown("#### 📊 Pend Rate Trend")
    pend_fig = px.bar(
        x=labels,
        y=pend_rate_chart,
        title="Pend Rate (%)",
        labels={'x': 'Date', 'y': 'Pend Rate (%)'},
        color_discrete_sequence=['orange']
    )
    st.plotly_chart(pend_fig, use_container_width=True)

# ---------------- KPI Table ----------------
st.subheader("📋 KPI Table")
st.dataframe(chart_df, use_container_width=True)

# ---------------- AI CHATBOT SECTION ----------------
import textwrap

st.markdown("## 👋✨ Meet Opsi — Your Smart Operations Assistant")

# Load and prepare a clean full version of the dataset (used only by chatbot)
file_id = "1mkVXQ_ZQsIXYnh72ysfqo-c2wyMZ7I_1"
file_url = f"https://drive.google.com/uc?export=download&id={file_id}"
raw_df = pd.read_csv(file_url, dayfirst=True, parse_dates=["Start Date", "End Date", "Target Date"])

# Summarize dataset for chatbot input
summary_text = f"""
📊 Dataset Summary:

- Rows: {raw_df.shape[0]}
- Columns: {raw_df.shape[1]}
- Fields: {', '.join(raw_df.columns)}

📈 Basic Statistics:
{raw_df.describe(include='all').fillna('-').to_string()}
"""

# Input box for user query
user_question = st.text_input("", placeholder="e.g. What’s the average pend rate in Jan?", key="chat_input")

# Enable Enter key to trigger submission
if user_question:
    with st.spinner("Analyzing your question..."):
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["openai_key"])

        prompt = textwrap.dedent(f"""
        You are an expert operational analyst. You will receive:

        1. A summarized dataset with statistics.
        2. A user question about the data.

        Your task is to respond clearly and concisely based on the data provided. Use bullet points if possible and include actual figures when relevant.

        --- DATA SUMMARY ---
        {summary_text}

        --- USER QUESTION ---
        {user_question}

        Answer:
        """)

        try:
            response = client.chat.completions.create(
                model="gpt-4",  # You can switch to "gpt-3.5-turbo" if needed
                messages=[
                    {"role": "system", "content": "You are a helpful analyst trained in data storytelling."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
        except Exception as e:
            st.error(f"❌ Error: {e}")
