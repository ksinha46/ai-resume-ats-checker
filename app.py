"""Streamlit user interface for the AI Resume Builder & ATS Score Checker."""

from __future__ import annotations

import streamlit as st

from utils import (
    AnalysisResult,
    analyze_resume,
    extract_text_from_pdf,
    get_score_label,
    prepare_nltk_resources,
)


def configure_page() -> None:
    """Set the browser tab title and the overall page layout."""
    st.set_page_config(
        page_title="AI Resume Builder & ATS Score Checker",
        page_icon="📄",
        layout="centered",
    )


def render_header() -> None:
    """Render the app introduction shown before the input controls."""
    st.title("AI Resume Builder & ATS Score Checker")
    st.markdown(
        "### AI Resume Builder & ATS Score Checker\n"
        "Compare your resume with a job description, see your ATS match score, "
        "and get practical next steps before you apply."
    )
    st.caption(
        "Your resume stays in this session. Analysis runs locally with "
        "PDF text extraction and TF-IDF matching—no API key required."
    )


def render_score(result: AnalysisResult) -> None:
    """Display the score, its interpretation, and a progress bar."""
    st.subheader("ATS Match Score")
    score_columns = st.columns([1, 2])
    with score_columns[0]:
        st.metric("Match score", f"{result.score:.0f}%")
    with score_columns[1]:
        st.progress(result.score / 100)
        st.markdown(f"**{get_score_label(result.score)}**")
        st.caption(
            "This score reflects wording overlap between the two documents. "
            "It is a guide, not a hiring decision."
        )


def render_missing_keywords(result: AnalysisResult) -> None:
    """Show the highest-value job-description terms not found in the resume."""
    st.subheader("Keywords Missing From Your Resume")
    if not result.missing_keywords:
        st.success(
            "Great coverage. No high-priority keywords were found to be missing."
        )
        return

    st.markdown(
        "Consider adding the terms that genuinely describe your experience. "
        "Avoid copying keywords without evidence."
    )
    for keyword in result.missing_keywords:
        st.markdown(f"- **{keyword}**")


def render_suggestions(result: AnalysisResult) -> None:
    """Display a short list of score-aware resume improvement suggestions."""
    st.subheader("Suggestions to Improve Your Match")
    for suggestion in result.suggestions:
        st.warning(suggestion)


def main() -> None:
    """Build the complete Streamlit page and handle the analysis action."""
    configure_page()
    render_header()

    # Download the small NLTK stopword resource automatically on first run.
    try:
        prepare_nltk_resources()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    st.divider()
    st.subheader("1. Upload your resume")
    resume_file = st.file_uploader(
        "Choose a PDF resume",
        type=["pdf"],
        help="Only PDF files are accepted. Text-based PDFs work best.",
    )

    st.subheader("2. Paste the job description")
    job_description = st.text_area(
        "Job description",
        height=240,
        placeholder=(
            "Paste the full job description here, including responsibilities "
            "and required skills..."
        ),
        label_visibility="collapsed",
    )

    check_score = st.button(
        "Check ATS Score",
        type="primary",
        use_container_width=True,
    )

    if not check_score:
        st.info(
            "Upload a resume, paste a job description, and select "
            "**Check ATS Score** to see your results."
        )
        return

    if resume_file is None:
        st.error("Please upload your resume as a PDF before checking the score.")
        return

    if not job_description.strip():
        st.error("Please paste a job description before checking the score.")
        return

    try:
        with st.spinner("Reading your resume and comparing both documents..."):
            resume_text = extract_text_from_pdf(resume_file)
            result = analyze_resume(resume_text, job_description)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception:
        # Do not expose a traceback or internal details to a user.
        st.error(
            "We couldn't analyze these files. Please try a text-based PDF "
            "and make sure the job description contains readable text."
        )
        return

    st.divider()
    render_score(result)
    render_missing_keywords(result)
    render_suggestions(result)

    with st.expander("How this score is calculated"):
        st.markdown(
            "The app cleans both documents, removes common English stopwords, "
            "converts the remaining terms into TF-IDF vectors, and measures "
            "their cosine similarity. The missing-keyword list comes from the "
            "highest-weight terms in the job description that do not appear "
            "in your resume."
        )


if __name__ == "__main__":
    main()