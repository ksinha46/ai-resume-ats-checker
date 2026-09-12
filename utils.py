"""Text extraction and offline resume-to-job-description analysis helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO

import nltk
import pdfplumber
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MAX_KEYWORDS = 12
MIN_KEYWORD_LENGTH = 2


@dataclass(frozen=True)
class AnalysisResult:
    """All values needed to render one resume comparison."""

    score: float
    missing_keywords: list[str]
    suggestions: list[str]


def prepare_nltk_resources() -> None:
    """Ensure the English stopword list exists, downloading it when needed."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            nltk.data.find("corpora/stopwords")
        except Exception as error:
            raise RuntimeError(
                "The app could not download its language resources. "
                "Please try again when an internet connection is available."
            ) from error


def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    """Extract readable text from every page in an uploaded PDF."""
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            page_text = [page.extract_text() or "" for page in pdf.pages]
    except Exception as error:
        raise ValueError(
            "We couldn't read that PDF. Please upload a valid, text-based PDF resume."
        ) from error

    text = "\n".join(page_text).strip()
    if not text:
        raise ValueError(
            "This PDF does not contain readable text. Please upload a text-based "
            "PDF rather than a scanned image."
        )
    return text


def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation, and remove English stopwords."""
    prepare_nltk_resources()
    words = re.findall(r"[a-z0-9][a-z0-9+#.-]*", text.lower())
    english_stopwords = set(stopwords.words("english"))
    cleaned_words = [
        word
        for word in words
        if len(word) >= MIN_KEYWORD_LENGTH and word not in english_stopwords
    ]
    return " ".join(cleaned_words)


def _get_missing_keywords(resume_text: str, job_description: str) -> list[str]:
    """Find important job-description terms that do not occur in the resume."""
    vectorizer = TfidfVectorizer(
        stop_words=list(stopwords.words("english")),
        ngram_range=(1, 2),
        max_features=80,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.-]*\b",
    )
    try:
        job_matrix = vectorizer.fit_transform([job_description])
    except ValueError:
        return []

    terms = vectorizer.get_feature_names_out()
    weights = job_matrix.toarray()[0]
    resume_lower = resume_text.lower()
    ranked_terms = sorted(
        zip(terms, weights),
        key=lambda term_and_weight: term_and_weight[1],
        reverse=True,
    )

    missing: list[str] = []
    for term, weight in ranked_terms:
        if weight <= 0:
            continue
        normalized_term = term.lower()
        if normalized_term not in resume_lower and normalized_term not in missing:
            missing.append(term)
        if len(missing) == MAX_KEYWORDS:
            break
    return missing


def _get_suggestions(score: float, missing_keywords: list[str]) -> list[str]:
    """Create 2–3 concise recommendations based on the match score."""
    suggestions: list[str] = []
    if score < 45:
        suggestions.append(
            "Add the missing skills that honestly match your experience, "
            "using the same wording as the job description where appropriate."
        )
        suggestions.append(
            "Rewrite your summary so the target role and your strongest "
            "qualifications appear in the first few lines."
        )
    elif score < 70:
        suggestions.append(
            "Work the most relevant missing keywords into your experience "
            "bullets or skills section when you can support them."
        )
        suggestions.append(
            "Match the job description's title and terminology, then remove "
            "older or unrelated details that dilute the match."
        )
    else:
        suggestions.append(
            "Keep the strong keyword alignment, and make each experience "
            "bullet more specific with measurable outcomes."
        )
        suggestions.append(
            "Check that the resume uses consistent dates, headings, and "
            "simple formatting so an ATS can parse it reliably."
        )

    if missing_keywords:
        suggestions.append(
            f"Review the top {min(3, len(missing_keywords))} missing terms first: "
            + ", ".join(missing_keywords[:3])
            + "."
        )
    return suggestions[:3]


def analyze_resume(resume_text: str, job_description: str) -> AnalysisResult:
    """Compare a resume to a job description and return ATS-style insights."""
    if not resume_text.strip():
        raise ValueError("The uploaded resume does not contain readable text.")
    if not job_description.strip():
        raise ValueError("Please provide a job description to compare.")

    cleaned_resume = clean_text(resume_text)
    cleaned_job_description = clean_text(job_description)
    if not cleaned_resume or not cleaned_job_description:
        raise ValueError(
            "There is not enough readable text to compare these documents."
        )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        vectors = vectorizer.fit_transform(
            [cleaned_resume, cleaned_job_description]
        )
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except ValueError as error:
        raise ValueError(
            "There is not enough unique text to calculate a match score."
        ) from error

    score = round(float(max(0.0, min(1.0, similarity)) * 100), 1)
    missing_keywords = _get_missing_keywords(cleaned_resume, cleaned_job_description)
    return AnalysisResult(
        score=score,
        missing_keywords=missing_keywords,
        suggestions=_get_suggestions(score, missing_keywords),
    )


def get_score_label(score: float) -> str:
    """Return a plain-language interpretation of the numeric match score."""
    if score < 45:
        return "Needs alignment"
    if score < 70:
        return "Good starting point"
    return "Strong alignment"