# llmutil.py - Clean & Optimized for AI Summary + Insights

import os
import re
from datetime import datetime
from typing import Dict, Tuple
import logging

from dotenv import load_dotenv
load_dotenv()

# Try importing the Gemini API
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
#  GEMINI SETUP
# ============================================================
def setup_gemini(api_key: str = None) -> bool:
    """Configure Gemini safely."""
    if genai is None:
        logger.error("❌ google.generativeai module not installed.")
        return False

  
    api_key = st.secrets["API_KEY"] or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        logger.error("❌ Gemini API key not found. Set GOOGLE_API_KEY in .env")
        return False

    try:
        genai.configure(api_key=api_key)
        logger.info("✅ Gemini API configured successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Failed configuring Gemini: {e}")
        return False


# ============================================================
#  AI CHART INSIGHT GENERATOR
# ============================================================
def generate_chart_insight(chart_type: str, chart_data, chart_title: str) -> str:
    """Generate insights for a chart using Gemini."""

    if genai is None:
        return "AI chart insights unavailable — Gemini not configured."

    try:
        # Convert chart data to readable summary
        try:
            data_dict = chart_data.to_dict()
        except:
            data_dict = str(chart_data)[:300]

        prompt = f"""
You are a senior business intelligence analyst.

Analyze this chart and write insights.

Chart Title: {chart_title}
Chart Type: {chart_type}

Chart Data Summary:
{data_dict}

Provide:
1. Key observations
2. Patterns & anomalies
3. Business interpretation
4. Actionable recommendation

Output must be clean and readable.
"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:
        return f"Error generating chart insight: {e}"


# ============================================================
#  AI BUSINESS SUMMARY REPORT
# ============================================================
def generate_summary(analysis_results: Dict, meta: Dict) -> str:
    """Generate a full business report from dataset analysis."""

    if genai is None:
        return "AI summary unavailable — Gemini not configured."

    try:
        prompt = f"""
You are a Business Intelligence expert. 
Generate a full business performance report.

DATASET INFO:
- Rows: {meta.get("rows")}
- Date Range: {meta.get("date_range")}

ANALYSIS SUMMARY:
{analysis_results}

Write a clean report with Markdown formatting:

## Executive Summary
(What overall performance shows)

## KPI Overview
(Sales, units, transactions, growth)

## Product Insights
(Top categories, top products, declines, strengths)

## Customer Insights
(Returning customers, customer value)

## Trend Analysis
(Monthly/seasonal performance)

## Opportunities
(Growth areas)

## Actionable Recommendations
(Short list of 3–5 actions)

Keep paragraphs short and professional.
"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:
        return f"Error generating summary: {e}"


# ============================================================
#  TEST FUNCTION
# ============================================================
def test_api_connection() -> Tuple[bool, str]:
    """Test whether Gemini API is working."""
    if genai is None:
        return False, "Gemini library not installed."

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("Say 'API working' if you can see this message.")

        return True, response.text.strip()

    except Exception as e:
        return False, str(e)
