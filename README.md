# 📄 AI Resume Analyzer

An AI-powered resume analysis tool built with **Python, Streamlit, scikit-learn, and Docker**.

Upload a resume, compare it with a job description, and identify matching skills and areas for improvement.

## ✨ Features

- PDF resume text extraction
- ATS-style match score estimation
- TF-IDF text similarity
- Matched and missing skill detection
- Required and preferred skill classification
- Personalized improvement suggestions
- Interactive dashboard
- Docker support

## 🛠️ Tech Stack

Python · Streamlit · PyMuPDF · scikit-learn · Docker

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR-USERNAME/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## 🐳 Run with Docker

```bash
docker compose up --build
```

## 📊 Scoring Method

The current ATS-style estimate combines:

- **70%** skill coverage
- **30%** text similarity

These are project-specific weights, not an official ATS formula or a prediction of hiring outcomes.

## ⚠️ Limitations

Skill matching relies on a predefined dictionary. Scanned PDFs may require OCR, and results should be treated as guidance rather than a hiring decision.

## 🔮 Future Improvements

- OCR support
- Expanded skill recognition
- Downloadable PDF reports
- Semantic text matching
- Online deployment
