# ResumeMatch — AI Resume Builder & ATS Score Checker

**Live demo**: https://ai-resume-ats-checker.streamlit.app

ResumeMatch is a Streamlit dashboard that compares a PDF resume with a pasted
job description, produces a realistic ATS-style match score, and helps you 
improve your resume directly in the website.

## How it works

1. Upload a text-based PDF resume.
2. Paste the job description.
3. Select **Check ATS Score**.
4. The app extracts the PDF text with `pdfplumber`.
5. It cleans both documents and removes common English stopwords with `nltk`.
6. It scores the match using a blend of keyword coverage and TF-IDF cosine 
   similarity, closer to how real ATS systems evaluate resumes than plain
   text similarity alone.
7. It ranks high-value job-description keywords and shows those missing from
   the resume.

The first run downloads NLTK's English stopword list automatically. No API key
or paid service is required.

## Dashboard

Results are organized into four tabs:

1. Score Overview - a color-coded match score (red/yellow/green) with a plain
   language interpretation, plus a before-and-after comparison once you re-check
   an edited resume.
2. Keyword Analysis - matched a missing keywords shown as tags, a word cloud of the
   job description's key terms, and a section-wise match breakdown (summary, skills,
   experience, education) pre-filled with suggested missing keywords, with a
   "re-check score" button and a sample strong resume for reference.
3. Download Report - a PDF report of the score, keywords, and suggestions, plus a text
   download of the edited resume.

A sidebar toggle switches between light and dark themes, and keep a short history of 
scores from earlier checks in the same session.
   
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
