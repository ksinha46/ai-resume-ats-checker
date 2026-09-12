# ResumeMatch — AI Resume Builder & ATS Score Checker

**Live demo:** https://ai-resume-ats-checker.streamlit.app

ResumeMatch is a local Streamlit app that compares a PDF resume with a pasted
job description. It produces an ATS-style match score, highlights important
keywords that are not present in the resume, and suggests practical edits.

## How it works

1. Upload a text-based PDF resume.
2. Paste the job description.
3. Select **Check ATS Score**.
4. The app extracts the PDF text with `pdfplumber`.
5. It cleans both documents and removes common English stopwords with `nltk`.
6. It compares the documents with TF-IDF vectors and cosine similarity.
7. It ranks high-value job-description keywords and shows those missing from
   the resume.

The first run downloads NLTK's English stopword list automatically. No API key
or paid service is required.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Project files

- `app.py` — Streamlit interface, input validation, and result presentation.
- `utils.py` — PDF extraction, text cleaning, scoring, keyword extraction, and
  suggestions.
- `requirements.txt` — Python dependencies.
- `README.md` — this guide.

## Notes

The score is a useful comparison signal, not a prediction of whether a
recruiter or hiring manager will choose an applicant. Add keywords only when
they accurately reflect real experience.
