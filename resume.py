# ================================
# Resume Screening AI Agent
# Local LLM (Ollama + LLaMA3)
# ================================

import fitz  # PyMuPDF
import ollama
import json
import re
import sys

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "llama3"
MAX_RESUME_CHARS = 6000

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")

# -------------------------------
# TEXT TRUNCATION (LLM SAFETY)
# -------------------------------
def truncate_text(text, max_chars=MAX_RESUME_CHARS):
    return text[:max_chars]

# -------------------------------
# LLM RESUME SCREENING
# -------------------------------
def screen_resume(resume_text, job_description):
    resume_text = truncate_text(resume_text)

    prompt = f"""
You are a Senior Technical Recruiter with 20 years of experience.

STRICT RULES:
- Output VALID JSON ONLY
- No markdown
- No explanations
- No extra text

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

Return JSON exactly in this format:
{{
  "candidate_name": "Full name if found, else Unknown",
  "match_score": 0-100,
  "key_strengths": ["strength1", "strength2", "strength3"],
  "missing_critical_skills": ["skill1", "skill2"],
  "recommendation": "Interview" or "Reject",
  "reasoning": "Two concise sentences explaining the decision."
}}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response["message"]["content"]

# -------------------------------
# SAFE JSON PARSER
# -------------------------------
def parse_llm_json(text):
    import json
    import re

    # Remove markdown if present
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

    # Try normal parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Auto-fix common LLM issue: missing closing brace
    if text.count("{") > text.count("}"):
        text = text + "}"

    return json.loads(text)


# -------------------------------
# MAIN EXECUTION
# -------------------------------
def main():

    # ---- JOB DESCRIPTION ----
    job_description = """
We are looking for a Junior Data Scientist.

Must have:
- Python (Pandas, NumPy, Scikit-Learn)
- Experience with SQL
- Basic understanding of Machine Learning algorithms
- Good communication skills

Nice to have:
- Experience with AWS or Cloud deployment
- Knowledge of NLP
"""

    # ---- RESUME PATH ----
    resume_path = r"C:\Users\vivek\Downloads\resumescanner\vivek_Machine Learning Engineer_amazon.pdf"

    # ---- LOAD RESUME ----
    try:
        resume_text = extract_text_from_pdf(resume_path)
        print(f"✅ Resume loaded ({len(resume_text)} characters)")
    except Exception as e:
        print("❌", e)
        sys.exit(1)

    # ---- SCREENING ----
    print("🤖 AI is analyzing the candidate...\n")
    raw_output = screen_resume(resume_text, job_description)

    # ---- PARSE OUTPUT ----
    try:
        result = parse_llm_json(raw_output)

        print("========== SCREENING REPORT ==========")
        print(f"Candidate Name : {result.get('candidate_name')}")
        print(f"Match Score    : {result.get('match_score')}/100")
        print(f"Decision       : {result.get('recommendation')}")
        print(f"Reasoning      : {result.get('reasoning')}")
        print("Key Strengths  :", ", ".join(result.get("key_strengths", [])))
        print("Missing Skills :", ", ".join(result.get("missing_critical_skills", [])))
        print("======================================")

    except Exception:
        print("❌ Failed to parse LLM output")
        print("\nRaw Output:\n", raw_output)

# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    main()
