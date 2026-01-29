import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data
from utils.standardize import merge_datasets, generate_missingness_report
from utils.qa_rules import QARules
from utils.ai_workflow import AIWorkflow
import uuid
from datetime import datetime

st.set_page_config(page_title="Campaign QA-First AI Workflow", layout="wide")

# Session State for Run Manifest
if 'run_id' not in st.session_state:
    st.session_state['run_id'] = str(uuid.uuid4())[:8]
if 'qa_status' not in st.session_state:
    st.session_state['qa_status'] = "Not Started"

# Sidebar
st.sidebar.title("Configuration")
campaign_name = st.sidebar.text_input("Campaign Name", "Hackathon")
use_dummy = st.sidebar.checkbox("Use dummy data if uploads missing", value=True)


# AI Settings
st.sidebar.markdown("---")
st.sidebar.subheader("AI Settings")
st.sidebar.subheader("AI Settings")
use_real_ai = st.sidebar.toggle("Use Real AI")
ai_provider = "Gemini"
api_key = ""

if use_real_ai:
    ai_provider = st.sidebar.selectbox("Select Provider", ["Gemini", "OpenAI (ChatGPT)"])
    if ai_provider == "Gemini":
        api_key = st.sidebar.text_input("Gemini API Key", type="password")
    else:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Requires an OpenAI API Key")
        
    if not api_key:
        st.sidebar.warning(f"API Key required for {ai_provider}.")

st.sidebar.subheader("Uploads")
uploads = {}
uploads['impressions'] = st.sidebar.file_uploader("Impressions (CSV)", type="csv")
uploads['visits'] = st.sidebar.file_uploader("Visits (CSV)", type="csv")
uploads['search'] = st.sidebar.file_uploader("Search (CSV)", type="csv")
uploads['web'] = st.sidebar.file_uploader("Web Analytics (CSV)", type="csv")


# Logic to load data
# If use_dummy is True and NO uploads, use dummy.
# If use_dummy is True and SOME uploads, mixed? For MVP, simplistic toggling:
# If any upload is present, we try to use uploads. If use_dummy is strictly forced for missing ones, we'd need to mix.
# The prompt says "Use dummy data if uploads missing" implies fallback.
raw_datasets = {}
if use_dummy and all(v is None for v in uploads.values()):
    raw_datasets = load_data(None, use_dummy=True)
else:
    # Load uploads, potentially missing some
    raw_datasets = load_data(uploads, use_dummy=False)
    # If partial dummy support needed, would add here.
    
# Main App
st.title(f"Campaign QA-First AI Workflow: {campaign_name}")

# Run Manifest Panel
with st.expander("Run Manifest", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Run ID", st.session_state['run_id'])
    col2.metric("Date", datetime.now().strftime("%Y-%m-%d"))
    col3.metric("QA Status", st.session_state['qa_status'])

# Processing
unified_df = pd.DataFrame()
qa_results = pd.DataFrame()
qa_summary = {}
missing_rpt = pd.DataFrame()

if raw_datasets:
    unified_df = merge_datasets(raw_datasets)
    
    # Run QA
    qa_engine = QARules(unified_df)
    qa_results, qa_summary = qa_engine.run_all()
    
    # Update status for manifest
    if qa_summary.get('total_fails', 0) > 0:
        st.session_state['qa_status'] = "FAIL"
    elif qa_summary.get('total_warns', 0) > 0:
        st.session_state['qa_status'] = "WARN"
    else:
        st.session_state['qa_status'] = "PASS"

tabs = st.tabs(["Data Preview", "Unified Dataset", "QA Center", "AI Summary"])

with tabs[0]:
    st.subheader("Data Preview")
    if not raw_datasets:
        st.info("No data loaded.")
    else:
        for name, df in raw_datasets.items():
            st.write(f"**{name}** ({len(df)} rows)")
            st.dataframe(df.head())

with tabs[1]:
    st.subheader("Unified Dataset")
    if unified_df.empty:
        st.info("Unified dataset is empty.")
        if raw_datasets:
             st.write("Debug Info:")
             for k, v in raw_datasets.items():
                 st.write(f"{k}: {v.shape} columns: {v.columns.tolist()}")
    else:
        st.dataframe(unified_df)
        
        st.download_button(
            "Download Unified Dataset",
            unified_df.to_csv(index=False),
            "unified_dataset.csv",
            "text/csv"
        )
        
        st.subheader("Missingness Report")
        missing_rpt = generate_missingness_report(unified_df)
        st.dataframe(missing_rpt)
        
        if st.button("Analyze Missingness with AI"):
            from utils.ai_workflow import MockLLMProvider, GeminiLLMProvider, OpenAILLMProvider
            llm_provider = None
            if use_real_ai and api_key:
                if ai_provider == "Gemini":
                    llm_provider = GeminiLLMProvider(api_key=api_key)
                elif ai_provider == "OpenAI (ChatGPT)":
                    llm_provider = OpenAILLMProvider(api_key=api_key)
            else:
                llm_provider = MockLLMProvider()
            
            if llm_provider:
                workflow = AIWorkflow(qa_summary, qa_results, unified_df, missing_rpt, llm_provider)
                with st.spinner("Analyzing missingness..."):
                    summary_text = workflow.summarize_missingness()
                    st.info(summary_text)

with tabs[2]:
    st.subheader("QA Center")
    if qa_results.empty and not unified_df.empty:
         st.success("No QA issues found!")
    elif not unified_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Fails", qa_summary.get('total_fails', 0))
        col2.metric("Total Warnings", qa_summary.get('total_warns', 0))
        col3.metric("Affected Stores", len(qa_summary.get('affected_stores', [])))
        
        filter_status = st.multiselect("Filter by Severity", ["FAIL", "WARN"], default=["FAIL", "WARN"])
        
        display_df = qa_results[qa_results['severity'].isin(filter_status)]
        st.dataframe(display_df)
        
        st.download_button(
            "Download QA Results",
            qa_results.to_csv(index=False),
            "qa_results.csv",
            "text/csv"
        )
        
        st.markdown("---")
        if st.button("Run AI Feasibility Analysis"):
            from utils.ai_workflow import MockLLMProvider, GeminiLLMProvider, OpenAILLMProvider
            llm_provider = None
            if use_real_ai and api_key:
                if ai_provider == "Gemini":
                    llm_provider = GeminiLLMProvider(api_key=api_key)
                elif ai_provider == "OpenAI (ChatGPT)":
                    llm_provider = OpenAILLMProvider(api_key=api_key)
            else:
                llm_provider = MockLLMProvider()
                
            if llm_provider:
                workflow = AIWorkflow(qa_summary, qa_results, unified_df, missing_rpt, llm_provider)
                with st.spinner("Checking feasibility..."):
                    feasibility_text = workflow.run_feasibility_check()
                    st.info(feasibility_text)
                    
        if st.button("Summarize QA Results with AI"):
            from utils.ai_workflow import MockLLMProvider, GeminiLLMProvider, OpenAILLMProvider
            llm_provider = None
            if use_real_ai and api_key:
                if ai_provider == "Gemini":
                    llm_provider = GeminiLLMProvider(api_key=api_key)
                elif ai_provider == "OpenAI (ChatGPT)":
                    llm_provider = OpenAILLMProvider(api_key=api_key)
            else:
                llm_provider = MockLLMProvider()
                
            if llm_provider:
                workflow = AIWorkflow(qa_summary, qa_results, unified_df, missing_rpt, llm_provider)
                with st.spinner("Summarizing QA results..."):
                    qa_summary_text = workflow.summarize_qa()
                    st.success(qa_summary_text)

with tabs[3]:
    st.subheader("AI Summary")
    if st.button("Run AI Workflow"):
        # Select Provider
        from utils.ai_workflow import MockLLMProvider, GeminiLLMProvider, OpenAILLMProvider
        
        llm_provider = None
        if use_real_ai and api_key:
            if ai_provider == "Gemini":
                llm_provider = GeminiLLMProvider(api_key=api_key)
            elif ai_provider == "OpenAI (ChatGPT)":
                llm_provider = OpenAILLMProvider(api_key=api_key)
        elif use_real_ai and not api_key:
            st.error(f"Please enter a valid API Key for {ai_provider}.")
        else:
            llm_provider = MockLLMProvider()
            
        if llm_provider:
            workflow = AIWorkflow(qa_summary, qa_results, unified_df, missing_rpt, llm_provider)
            stages = workflow.run()
            
            final_report = ""
            
            for stage_name, content in stages.items():
                with st.expander(f"Stage {stage_name}", expanded=True):
                    st.markdown(content)
                    final_report += f"\n\n{content}"
                    
            st.download_button(
                "Download Final Report",
                final_report,
                "ai_report.md",
                "text/markdown"
            )
