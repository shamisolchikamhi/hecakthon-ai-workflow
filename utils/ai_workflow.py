import pandas as pd
import os
from abc import ABC, abstractmethod
import google.generativeai as genai
import openai

class LLMProvider(ABC):
    @abstractmethod
    def generate_content(self, section_name: str, context: dict) -> str:
        """Generates content for a specific section given the context."""
        pass

class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o" # or gpt-3.5-turbo

    def generate_content(self, section_name: str, context: dict) -> str:
        prompt = self._build_prompt(section_name, context)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant for marketing analysis."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating content with OpenAI: {e}"

    def _build_prompt(self, section_name: str, context: dict) -> str:
        # Reuse logic from Gemini provider or specific logic? 
        # Ideally we refactor prompt building to a common mixin or helper, 
        # but for now we can duplicate the prompt logic from Gemini provider if it's identical.
        # Let's DRY it up by making a BasePromptBuilder or just copying for speed/MVP.
        
        qa_summary = context.get('qa_summary', {})
        qa_results = context.get('qa_results', pd.DataFrame())
        unified_df = context.get('unified_df', pd.DataFrame())
        missing_rpt = context.get('missingness_report', pd.DataFrame())
        
        # Prepare context string
        data_sample = ""
        if not unified_df.empty:
             summary_stats = unified_df.describe().to_string()
             data_sample = f"Data Summary Statistics:\n{summary_stats}\n\nTop 5 rows:\n{unified_df.head().to_string()}"

        qa_context = f"QA Summary: {qa_summary}\n"
        if not qa_results.empty:
            qa_context += f"QA Detailed Results (First 10):\n{qa_results.head(10).to_string()}"

        if section_name == "Stage A":
            return f"""
            You are a Data Quality Auditor. Write 'Stage A: QA Summary' for a campaign report.
            Base your analysis on the following QA results:
            {qa_context}
            
            Structure:
            1. Overall QA status (GREEN/YELLOW/RED).
            2. List critical failures and warnings.
            3. Provide data-grounded hypotheses for why these issues occurred.
            4. Recommend next checks.
            Keep it professional and concise. Use Markdown.
            """
        elif section_name == "Stage B":
            return f"""
            You are a Marketing Analyst. Write 'Stage B: Performance Contributors' for a campaign report.
            Base your analysis on this data sample:
            {data_sample}
            
            And consider the QA Context:
            {qa_context}
            
            Structure:
            1. Start with a Caveat based on QA status.
            2. Highlight Top 5 and Bottom 5 stores by visits/impressions.
            3. Comment on CTR performance.
            Keep it professional and concise. Use Markdown.
            """
        elif section_name == "Stage C":
             return f"""
            You are a Senior Analyst. Write 'Stage C: Store Drilldown'.
            Analyze patterns in the data:
            {data_sample}
            
            Identify outliers, successful patterns, and suggested further analysis.
            Keep it professional and concise. Use Markdown.
            """

        elif section_name == "Stage E":
             return f"""
            You are an Executive. Write 'Stage E: Final Executive Report'.
            Synthesize the findings from QA, Performance, and Forecast.
            
            QA Context: {qa_context}
            Data Context: {data_sample}
            
            Structure:
            1. Executive Summary.
            2. Key Actions (Address QA issues first if any, then performance).
            Keep it professional and concise. Use Markdown.
            """
        elif section_name == "Stage F":
             return f"""
            You are a Data Scrutinizer. Perform a 'Feasibility Check' on the traffic data.
            
            Data Sample (Store, Visits, Exposed Visits):
            {data_sample}
            
            Task:
            1. Evaluate if the 'Avg Daily Visits' (you can estimate from total / days) seem plausible for the store type (infer from store name).
               - e.g., Grocery stores have high footfall, Boutiques have lower.
            2. Evaluate if 'Exposed Visits' counts are realistic given general digital ad campaign reach (industry standards).
            3. Flag any store where numbers look suspiciously high (bot traffic?) or low.
            
            Keep it professional. Focus on top anomalies. Use Markdown.
            """
        elif section_name == "Missingness Summary":
             return f"""
            You are a Data Engineer. Summarize this Missingness Report for a non-technical user.
            
            Missingness Report (Count of missing values per store per metric):
            {missing_rpt.to_string() if not missing_rpt.empty else "No missing data."}
            
            Task:
            1. Which stores have missing data?
            2. Which metrics are most affected?
            3. is the missingness random or systematic (e.g. entire columns missing)?
            
            Keep it concise. Use Markdown.
            """
        elif section_name == "QA Summary":
             return f"""
            You are a QA Lead. Summarize these QA Check results for a campaign report.
            
            QA Results:
            {qa_context}
            
            Task:
            1. Summarize the major failures and warnings.
            2. What is the impact of these issues on the campaign analysis?
            3. What are the top 3 priorities for data cleaning?
            
            Keep it professional and concise. Use Markdown.
            """
        return "Please generate a report section."

class MockLLMProvider(LLMProvider):
    def generate_content(self, section_name: str, context: dict) -> str:
        qa_summary = context.get('qa_summary', {})
        qa_results = context.get('qa_results', pd.DataFrame())
        unified_df = context.get('unified_df', pd.DataFrame())
        unified_df = context.get('unified_df', pd.DataFrame())

        if section_name == "Stage A":
            status = "GREEN"
            if qa_summary.get('total_fails', 0) > 0:
                status = "RED"
            elif qa_summary.get('total_warns', 0) > 0:
                status = "YELLOW"
                
            markdown = f"### Stage A: QA Summary\n\n"
            markdown += f"**Overall QA status: {status}**\n\n"
            
            if status != "GREEN":
                markdown += f"- **Total Fails:** {qa_summary.get('total_fails', 0)}\n"
                markdown += f"- **Total Warnings:** {qa_summary.get('total_warns', 0)}\n"
                affected = list(qa_summary.get('affected_stores', []))
                markdown += f"- **Affected Stores:** {', '.join(affected[:5])}"
                if len(affected) > 5:
                    markdown += " ...\n"
                else:
                    markdown += "\n"
                    
                if qa_summary.get('total_fails', 0) > 0 and not qa_results.empty:
                    markdown += "\n**Critical Failures:**\n"
                    fails = qa_results[qa_results['severity'] == "FAIL"]
                    for i, row in fails.head(5).iterrows():
                        markdown += f"- {row['store_name']}: {row['rule_name']} ({row['reason']})\n"
                        
                markdown += "\n**Possible Causes:**\n"
                markdown += "- Data ingestion errors or incomplete uploads.\n"
                markdown += "- Potential bot traffic spikes if exposed visits are anomalously high.\n"
                
                markdown += "\n**Recommended Checks:**\n"
                markdown += "- Re-verify raw CSV sources.\n"
                markdown += "- Check date range completeness.\n"
            else:
                markdown += "All checks passed. Data looks consistent for analysis.\n"
            return markdown

        elif section_name == "Stage B":
            markdown = "### Stage B: Performance Contributors\n\n"
            markdown += "> **QA Caveat:** Please review Stage A for data reliability context.\n\n"
            
            if unified_df.empty:
                return markdown + "No data available."
                
            store_stats = unified_df.groupby('store_name')[['total_visits', 'impressions']].sum().sort_values('total_visits', ascending=False)
            top_5 = store_stats.head(5)
            bottom_5 = store_stats.tail(5)
            
            markdown += "**Top 5 Stores (by Visits):**\n"
            for store, row in top_5.iterrows():
                markdown += f"- {store}: {int(row['total_visits'])} visits, {int(row['impressions'])} imps\n" # simplified
                
            markdown += "\n**Bottom 5 Stores (by Visits):**\n"
            for store, row in bottom_5.iterrows():
                markdown += f"- {store}: {int(row['total_visits'])} visits, {int(row['impressions'])} imps\n"
                
            return markdown

        elif section_name == "Stage C":
            markdown = "### Stage C: Store Drilldown\n\n"
            markdown += "Analysing outliers and patterns...\n\n"
            markdown += "- **High Performers:** Stores with high CTR tend to have better conversion rates.\n"
            markdown += "- **Outliers:** Check stores with low visits but high search metrics.\n"
            markdown += "\n*Suggested Analysis:* Correlation between Search Metric and In-Store Visits.\n"
            return markdown



        elif section_name == "Stage E":
            markdown = "### Stage E: Final Executive Report\n\n"
            markdown += "#### Executive Summary\n"
            status = "GREEN"
            if qa_summary.get('total_fails', 0) > 0:
                status = "RED"
            elif qa_summary.get('total_warns', 0) > 0:
                status = "YELLOW"
                
            markdown += f"The campaign data has been processed with a **{status}** QA status. "
            if status == "RED":
                markdown += "Significant data quality issues were detected that may affect analysis reliability. "
            else:
                markdown += "Data quality is generally sufficient for analysis. "
            return markdown
            
        return ""

class GeminiLLMProvider(LLMProvider):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-flash-latest')

    def generate_content(self, section_name: str, context: dict) -> str:
        prompt = self._build_prompt(section_name, context)
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
            except Exception as list_err:
                 available_models = [f"Could not list models: {list_err}"]
                 
            return f"Error with {self.model.model_name}: {e}\n\n**Available Models for your Key:**\n" + "\n".join([f"- `{m}`" for m in available_models])

    def _build_prompt(self, section_name: str, context: dict) -> str:
        qa_summary = context.get('qa_summary', {})
        qa_results = context.get('qa_results', pd.DataFrame())
        unified_df = context.get('unified_df', pd.DataFrame())
        missing_rpt = context.get('missingness_report', pd.DataFrame())
        
        # Prepare context string
        data_sample = ""
        if not unified_df.empty:
             summary_stats = unified_df.describe().to_string()
             data_sample = f"Data Summary Statistics:\n{summary_stats}\n\nTop 5 rows:\n{unified_df.head().to_string()}"

        qa_context = f"QA Summary: {qa_summary}\n"
        if not qa_results.empty:
            qa_context += f"QA Detailed Results (First 10):\n{qa_results.head(10).to_string()}"

        if section_name == "Stage A":
            return f"""
            You are a Data Quality Auditor. Write 'Stage A: QA Summary' for a campaign report.
            Base your analysis on the following QA results:
            {qa_context}
            
            Structure:
            1. Overall QA status (GREEN/YELLOW/RED).
            2. List critical failures and warnings.
            3. Provide data-grounded hypotheses for why these issues occurred.
            4. Recommend next checks.
            Keep it professional and concise. Use Markdown.
            """
        elif section_name == "Stage B":
            return f"""
            You are a Marketing Analyst. Write 'Stage B: Performance Contributors' for a campaign report.
            Base your analysis on this data sample:
            {data_sample}
            
            And consider the QA Context:
            {qa_context}
            
            Structure:
            1. Start with a Caveat based on QA status.
            2. Highlight Top 5 and Bottom 5 stores by visits/impressions.
            3. Comment on CTR performance.
            Keep it professional and concise. Use Markdown.
            """
        elif section_name == "Stage C":
             return f"""
            You are a Senior Analyst. Write 'Stage C: Store Drilldown'.
            Analyze patterns in the data:
            {data_sample}
            
            Identify outliers, successful patterns, and suggested further analysis.
            Keep it professional and concise. Use Markdown.
            """

        elif section_name == "Stage E":
             return f"""
            You are an Executive. Write 'Stage E: Final Executive Report'.
            Synthesize the findings from QA, Performance, and Forecast.
            
            QA Context: {qa_context}
            Data Context: {data_sample}
            
            Structure:
            1. Executive Summary.
            2. Key Actions (Address QA issues first if any, then performance).
            Keep it professional and concise. Use Markdown.
            """
        elif section_name == "Stage F":
             return f"""
            You are a Data Scrutinizer. Perform a 'Feasibility Check' on the traffic data.
            
            Data Sample (Store, Visits, Exposed Visits):
            {data_sample}
            
            Task:
            1. Evaluate if the 'Avg Daily Visits' (you can estimate from total / days) seem plausible for the store type (infer from store name).
               - e.g., Grocery stores have high footfall, Boutiques have lower.
            2. Evaluate if 'Exposed Visits' counts are realistic given general digital ad campaign reach (industry standards).
            3. Flag any store where numbers look suspiciously high (bot traffic?) or low.
            
            Keep it professional. Focus on top anomalies. Use Markdown.
            """
        elif section_name == "Missingness Summary":
             return f"""
            You are a Data Engineer. Summarize this Missingness Report for a non-technical user.
            
            Missingness Report (Count of missing values per store per metric):
            {missing_rpt.to_string() if not missing_rpt.empty else "No missing data."}
            
            Task:
            1. Which stores have missing data?
            2. Which metrics are most affected?
            3. is the missingness random or systematic (e.g. entire columns missing)?
            
            Keep it concise. Use Markdown.
            """
        elif section_name == "QA Summary":
             return f"""
            You are a QA Lead. Summarize these QA Check results for a campaign report.
            
            QA Results:
            {qa_context}
            
            Task:
            1. Summarize the major failures and warnings.
            2. What is the impact of these issues on the campaign analysis?
            3. What are the top 3 priorities for data cleaning?
            
            Keep it professional and concise. Use Markdown.
            """
        return "Please generate a report section."


class AIWorkflow:
    def __init__(self, qa_summary, qa_results, unified_df, missingness_report, llm_provider: LLMProvider):
        self.context = {
            "qa_summary": qa_summary,
            "qa_results": qa_results,
            "unified_df": unified_df,
            "missingness_report": missingness_report
        }
        self.llm_provider = llm_provider
        self.stages = {}

    def run(self):
        """
        Runs the workflow stages using the configured provider.
        """
        self.stages['A'] = self.llm_provider.generate_content('Stage A', self.context)
        self.stages['B'] = self.llm_provider.generate_content('Stage B', self.context)
        self.stages['C'] = self.llm_provider.generate_content('Stage C', self.context)
        
        self.stages['E'] = self.llm_provider.generate_content('Stage E', self.context)
        return self.stages

    def run_feasibility_check(self):
        return self.llm_provider.generate_content('Stage F', self.context)
        
    def summarize_missingness(self):
        return self.llm_provider.generate_content('Missingness Summary', self.context)
        
    def summarize_qa(self):
        return self.llm_provider.generate_content('QA Summary', self.context)
