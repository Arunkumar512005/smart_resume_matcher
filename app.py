import streamlit as st
import os
import re
import pdfplumber
import docx2txt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Text cleaning
def clean_text(txt):
    txt = re.sub(r"[^A-Za-z0-9\s]", " ", txt.lower())
    tokens = [w for w in txt.split() if w not in stopwords]
    return " ".join(tokens)

# Extract text from resume
def extract_resume_text(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        with pdfplumber.open(uploaded_file) as pdf:
            return " ".join([page.extract_text() or "" for page in pdf.pages])
    elif uploaded_file.name.endswith('.docx'):
        return docx2txt.process(uploaded_file)
    return ""

# Feedback: missing keywords
def get_feedback(resume_text, jd_text):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())
    missing = jd_words - resume_words
    return ", ".join(list(missing)[:10])

# Streamlit config
st.set_page_config(page_title="Smart Resume Matcher", layout="wide")
st.title("📄 Smart Resume Matcher")

# Mode selector
st.sidebar.title("Choose User Mode")
mode = st.sidebar.radio("Who are you?", ("Student", "HR/Recruiter"))

# -------------------------------
# 🎓 Student Mode
# -------------------------------
if mode == "Student":
    st.header("🎓 Student Resume Analyzer")
    st.markdown("""
    Upload your resume and paste a job description.
    We'll show your match score and feedback on missing keywords.
    """)

    jd_input = st.text_area("📌 Paste the Job Description", height=200)
    uploaded_file = st.file_uploader("📤 Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

    analyze_btn = st.button("🔍 Analyze Resume (Student)")

    if analyze_btn and jd_input and uploaded_file:
        clean_jd = clean_text(jd_input)
        jd_embed = model.encode([clean_jd])

        resume_text = extract_resume_text(uploaded_file)
        clean_resume = clean_text(resume_text)
        resume_embed = model.encode([clean_resume])

        score = cosine_similarity(resume_embed, jd_embed)[0][0] * 100
        feedback = get_feedback(clean_resume, clean_jd)

        st.success(f"✅ Match Score: {round(score, 2)}%")
        st.markdown(f"**🔎 Missing Keywords:** {feedback}")

    elif analyze_btn:
        st.warning("⚠️ Please upload a resume and enter a job description.")

# -------------------------------
# 🧑‍💼 HR Mode
# -------------------------------
elif mode == "HR/Recruiter":
    st.header("🧑‍💼 HR Resume Matcher")
    st.markdown("""
    Upload multiple resumes and paste a job description.
    We'll rank resumes by match score and give feedback on missing keywords.
    """)

    jd_input = st.text_area("📌 Paste the Job Description", height=200)
    uploaded_files = st.file_uploader("📤 Upload resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    match_btn = st.button("📊 Match Resumes (HR)")

    if match_btn and jd_input and uploaded_files:
        clean_jd = clean_text(jd_input)
        jd_embed = model.encode([clean_jd])
        results = []

        for file in uploaded_files:
            resume_text = extract_resume_text(file)
            clean_resume = clean_text(resume_text)
            resume_embed = model.encode([clean_resume])
            score = cosine_similarity(resume_embed, jd_embed)[0][0] * 100
            feedback = get_feedback(clean_resume, clean_jd)
            results.append((file.name, round(score, 2), feedback))

        top_results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
        df = pd.DataFrame(top_results, columns=["Resume", "Match Score (%)", "Missing Keywords"])

        st.success("🏆 Top 10 Matching Resumes")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name='top_matched_resumes.csv',
            mime='text/csv',
        )

    elif match_btn:
        st.warning("⚠️ Please provide both job description and resumes.")
