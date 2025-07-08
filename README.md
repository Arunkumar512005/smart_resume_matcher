# smart_resume_matcher
# 🧠 Smart Resume Matcher (Streamlit App)

A powerful NLP-based web app that analyzes resumes and ranks them against a given Job Description using semantic similarity (BERT embeddings). Designed for both **Students** and **HR/Recruiters**.

---

## 🔍 Features

- 🎯 Match resumes with job descriptions using Sentence-BERT
- 📈 Get semantic **Match Scores** for each resume
- 🧠 Receive **feedback on missing keywords**
- 🧾 Upload resumes in PDF or DOCX format
- 📤 Export results as a CSV
- 👥 Dual mode:
  - **Student Mode**: Analyze your resume against a JD and get personalized tips
  - **HR Mode**: Upload 100+ resumes and get the top 10 best matches

---


## 📦 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **NLP**: [Sentence-Transformers](https://www.sbert.net/)
- **Backend**: Python (Sklearn, NLTK, Pandas)
- **File Handling**: pdfplumber, docx2txt

---

## 🧰 Installation

```bash
git clone https://github.com/your-username/smart-resume-matcher.git
cd smart-resume-matcher
pip install -r requirements.txt
streamlit run app.py
