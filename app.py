"""Professional dashboard for resume and job-description alignment."""

from __future__ import annotations

import html
import io
import logging
import time

import streamlit as st
import pdfplumber
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from wordcloud import WordCloud

from utils import (
    AnalysisResult,
    MAX_OCR_FILE_BYTES,
    MAX_OCR_PAGES,
    analyze_resume,
    extract_text_from_pdf,
    get_score_label,
    prepare_nltk_resources,
    split_resume_sections,
)

logger = logging.getLogger(__name__)


STRONG_EXAMPLE = """SUMMARY
Results-focused operations professional with 6+ years of experience improving
workflows, leading cross-functional projects, and turning data into decisions.

SKILLS
Project management, stakeholder communication, process improvement, data
analysis, reporting, Excel, SQL, team leadership

EXPERIENCE
Operations Manager | Example Company | 2021–Present
- Reduced order-processing time by 28% by mapping and redesigning workflows.
- Led a five-person team and delivered 12 projects on schedule.
- Built weekly performance reports used by senior leaders for planning.

EDUCATION
Bachelor of Business Administration | Example University
"""


def configure_page() -> None:
    """Set page metadata before drawing any interface elements."""
    st.set_page_config(
        page_title="ResumeMatch | ATS Alignment Dashboard",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def initialize_state() -> None:
    """Create session values once so edits survive Streamlit reruns."""
    defaults = {
        "result": None,
        "resume_text": "",
        "analyzed_job_description": "",
        "comparison_before": None,
        "history": [],
        "editor_ready": False,
        "dark_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme() -> None:
    """Apply a lightweight light or dark dashboard theme with CSS."""
    dark = st.session_state.dark_mode
    background = "#0b1220" if dark else "#f6f8fb"
    panel = "#111c2e" if dark else "#ffffff"
    text = "#e8eef8" if dark else "#152033"
    muted = "#a9b7ca" if dark else "#5d6b7e"
    border = "#26364d" if dark else "#dde3ec"
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {background}; color: {text}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: {panel}; }}
        h1, h2, h3, p, label {{ color: {text}; }}
        .muted {{ color: {muted}; }}
        .hero {{
            background: linear-gradient(135deg, #4f46e5, #2563eb);
            border-radius: 20px; color: white; padding: 2rem 2.2rem;
            margin-bottom: 1.5rem; box-shadow: 0 12px 30px rgba(37,99,235,.18);
        }}
        .hero h1, .hero p {{ color: white; margin: 0; }}
        .hero p {{ margin-top: .55rem; opacity: .9; }}
        .score-card {{
            background: {panel}; border: 1px solid {border}; border-radius: 16px;
            padding: 1.25rem; min-height: 138px;
        }}
        .score-number {{ font-size: 3rem; font-weight: 750; line-height: 1; }}
        .pill {{
            display: inline-block; padding: .38rem .72rem; margin: .22rem;
            border-radius: 999px; font-size: .88rem; font-weight: 600;
        }}
        .pill-match {{ background: #dcfce7; color: #166534; }}
        .pill-missing {{ background: #ffedd5; color: #9a3412; }}
        div[data-testid="stTab"] button {{ font-weight: 650; }}
        div[data-testid="stMetric"] {{
            background: {panel}; border: 1px solid {border}; border-radius: 14px;
            padding: .8rem 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Show appearance controls and progress from this browser session."""
    st.sidebar.header("Dashboard settings")
    st.sidebar.toggle("Dark theme", key="dark_mode")
    st.sidebar.divider()
    st.sidebar.subheader("Session History")
    history = st.session_state.history
    if not history:
        st.sidebar.caption("Your completed checks will appear here.")
    else:
        for item in reversed(history[-6:]):
            st.sidebar.markdown(f"**{item['label']}**  \n{item['score']:.0f}% match")
    st.sidebar.divider()
    st.sidebar.caption("Files are processed only for the current session.")


def render_header() -> None:
    """Introduce the dashboard and explain its purpose."""
    st.markdown(
        """
        <div class="hero">
          <h1>ResumeMatch</h1>
          <p>Measure job-specific keyword alignment, uncover gaps, and improve
          your resume before you apply.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_color(score: float) -> str:
    """Return the result color for a score band."""
    if score < 45:
        return "#dc2626"
    if score < 70:
        return "#d97706"
    return "#16a34a"


def history_label(job_description: str) -> str:
    """Create a short, useful label from the job description."""
    compact = " ".join(job_description.split())
    return compact[:42] + ("…" if len(compact) > 42 else "")


def add_history(score: float, job_description: str) -> None:
    """Store a concise record of each successful score check."""
    st.session_state.history.append(
        {"label": history_label(job_description), "score": score}
    )


def populate_editor(resume_text: str, missing_keywords: list[str]) -> None:
    """Pre-fill editable sections from the uploaded resume."""
    sections = split_resume_sections(resume_text)
    suggestions = ", ".join(missing_keywords[:8])
    skills = sections["skills"]
    if suggestions:
        skills = (
            skills
            + ("\n\n" if skills else "")
            + "Suggested terms to consider (only keep those you can support): "
            + suggestions
        )
    for name, content in sections.items():
        st.session_state[f"edit_{name}"] = skills if name == "skills" else content
    st.session_state.editor_ready = True


def analyze_upload(resume_file, job_description: str) -> None:
    """Validate inputs, extract the PDF, and update dashboard state."""
    if resume_file is None:
        st.error("Please upload a PDF resume before checking your score.")
        return
    if not job_description.strip():
        st.error("Please paste a job description before checking your score.")
        return
    try:
        with st.status("Reading your resume...", expanded=True) as extraction_status:
            def show_ocr_progress(page: int, total: int) -> None:
                extraction_status.update(
                    label=f"Scanning page {page} of {total} with local OCR...",
                    state="running",
                )

            resume_text = extract_text_from_pdf(resume_file, show_ocr_progress)
            extraction_status.update(label="Resume text is ready.", state="complete")
        with st.spinner("Comparing your resume with the job description..."):
            result = analyze_resume(resume_text, job_description)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
        return
    except Exception:
        st.error(
            "We couldn't complete the analysis. Try a text-based PDF and a "
            "job description with clear responsibilities and skills."
        )
        return

    st.session_state.resume_text = resume_text
    st.session_state.result = result
    st.session_state.analyzed_job_description = job_description
    st.session_state.comparison_before = None
    populate_editor(resume_text, result.missing_keywords)
    add_history(result.score, job_description)


def render_score_overview(result: AnalysisResult) -> None:
    """Render the primary score, component metrics, and recommendations."""
    color = score_color(result.score)
    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"""
            <div class="score-card">
              <div class="muted">ATS alignment score</div>
              <div class="score-number" style="color:{color}">{result.score:.0f}%</div>
              <div style="color:{color};font-weight:700;margin-top:.55rem">
                {get_score_label(result.score)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        progress = st.progress(0, text="Calculating alignment")
        for value in range(0, int(result.score) + 1, max(1, int(result.score) // 12 or 1)):
            progress.progress(min(value, 100), text=f"{min(value, int(result.score))}% aligned")
            time.sleep(0.012)
        progress.progress(int(result.score), text=f"{result.score:.0f}% aligned")
        st.info(
            "This score reflects keyword alignment with this job description, "
            "not your overall resume quality or hiring potential."
        )

    before = st.session_state.comparison_before
    if before is not None:
        delta = result.score - before
        st.subheader("Before → After")
        first, arrow, second = st.columns([1, 0.35, 1])
        first.metric("Previous score", f"{before:.0f}%")
        arrow.markdown("<h2 style='text-align:center'>→</h2>", unsafe_allow_html=True)
        second.metric("Edited score", f"{result.score:.0f}%", f"{delta:+.1f} points")

    st.subheader("How the score was built")
    metric_one, metric_two = st.columns(2)
    metric_one.metric("Keyword coverage · 70% weight", f"{result.keyword_coverage:.0f}%")
    metric_two.metric("Text similarity · 30% weight", f"{result.similarity:.0f}%")
    st.subheader("Recommended next steps")
    for suggestion in result.suggestions:
        st.markdown(f"- {suggestion}")


def render_pills(items: list[str], class_name: str, empty_message: str) -> None:
    """Render safe HTML badges instead of a plain keyword table."""
    if not items:
        st.success(empty_message)
        return
    pills = "".join(
        f'<span class="pill {class_name}">{html.escape(item)}</span>' for item in items
    )
    st.markdown(pills, unsafe_allow_html=True)


def render_keyword_analysis(result: AnalysisResult) -> None:
    """Show matched terms, missing terms, word cloud, and section estimates."""
    st.subheader("Matched keywords")
    render_pills(result.matched_keywords, "pill-match", "No priority terms matched yet.")
    st.subheader("Missing keywords")
    render_pills(
        result.missing_keywords,
        "pill-missing",
        "Excellent coverage — no priority terms are missing.",
    )
    st.caption("Only add missing terms when they accurately describe your experience.")

    st.subheader("Job description word cloud")
    try:
        cloud = WordCloud(
            width=1000,
            height=360,
            background_color="white",
            colormap="viridis",
            prefer_horizontal=0.9,
        ).generate_from_frequencies(result.keyword_weights)
        image = io.BytesIO()
        cloud.to_image().save(image, format="PNG")
        st.image(image.getvalue(), width="stretch")
    except Exception:
        st.info("The word cloud is unavailable, but the keyword lists are complete.")

    st.subheader("Estimated section alignment")
    st.caption(
        "Section scores estimate how many priority job terms appear in each detected area."
    )
    for section, score in result.section_scores.items():
        st.markdown(f"**{section} · {score:.0f}%**")
        st.progress(int(score))


def combined_editor_text() -> str:
    """Combine the four editable fields into one ATS-readable resume."""
    blocks = []
    for section in ("summary", "skills", "experience", "education"):
        content = st.session_state.get(f"edit_{section}", "").strip()
        if content:
            blocks.append(f"{section.upper()}\n{content}")
    return "\n\n".join(blocks)


def render_resume_editor(result: AnalysisResult | None) -> None:
    """Offer a structured editor and re-score the user's revised text."""
    needs_focus = result is not None and result.score < 60
    if needs_focus:
        st.warning(
            "Your score is below 60%. Start with the highlighted missing keywords, "
            "but only add terms that match your real experience."
        )
    with st.expander("Resume Editor", expanded=needs_focus or result is None):
        if not st.session_state.editor_ready:
            st.info("Upload and analyze a resume to pre-fill the editor.")
        for section, height in (
            ("summary", 150),
            ("skills", 150),
            ("experience", 260),
            ("education", 130),
        ):
            st.text_area(section.title(), key=f"edit_{section}", height=height)

        if st.button(
            "Re-check Score",
            type="primary",
            width="stretch",
            disabled=not st.session_state.editor_ready,
        ):
            edited_text = combined_editor_text()
            job_description = st.session_state.analyzed_job_description
            if not edited_text.strip():
                st.error("Add some resume content before re-checking the score.")
            else:
                try:
                    with st.spinner("Re-checking your edited resume..."):
                        new_result = analyze_resume(edited_text, job_description)
                    previous = st.session_state.result
                    st.session_state.comparison_before = previous.score
                    st.session_state.result = new_result
                    st.session_state.resume_text = edited_text
                    add_history(new_result.score, job_description)
                    st.success(
                        "Score updated. Open Score Overview to see the comparison."
                    )
                except (ValueError, RuntimeError) as error:
                    st.error(str(error))
                except Exception:
                    st.error("We couldn't re-check the edited text. Please try again.")

    with st.expander("Show a strong example resume"):
        st.code(STRONG_EXAMPLE, language=None)


def plain_report(result: AnalysisResult) -> str:
    """Build a portable plain-text version of the analysis report."""
    matched = ", ".join(result.matched_keywords) or "None"
    missing = ", ".join(result.missing_keywords) or "None"
    suggestions = "\n".join(f"- {item}" for item in result.suggestions)
    return (
        "RESUMEMATCH ATS ALIGNMENT REPORT\n\n"
        f"Overall score: {result.score:.1f}%\n"
        f"Keyword coverage: {result.keyword_coverage:.1f}%\n"
        f"Text similarity: {result.similarity:.1f}%\n\n"
        f"MATCHED KEYWORDS\n{matched}\n\n"
        f"MISSING KEYWORDS\n{missing}\n\n"
        f"SUGGESTIONS\n{suggestions}\n"
    )


def pdf_report(result: AnalysisResult) -> bytes:
    """Create a simple PDF report that works with built-in fonts."""
    report = plain_report(result)
    safe_report = report.encode("latin-1", "replace").decode("latin-1")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(
        0,
        12,
        "ResumeMatch ATS Alignment Report",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, safe_report.split("\n\n", 1)[1])
    output = pdf.output()
    return output.encode("latin-1") if isinstance(output, str) else bytes(output)


def validate_pdf_report(report_data: bytes) -> None:
    """Raise when generated report bytes cannot be read as a non-empty PDF."""
    with pdfplumber.open(io.BytesIO(report_data)) as parsed_pdf:
        if not parsed_pdf.pages:
            raise ValueError("Generated PDF has no pages.")


def render_downloads(result: AnalysisResult | None) -> None:
    """Provide the report and edited resume without allowing export errors to crash."""
    if result is None:
        st.info("Analyze a resume to unlock report downloads.")
        return

    st.subheader("Download your analysis")
    text_report = plain_report(result)
    export_stage = "generation"
    report_data: bytes | None = None
    try:
        report_data = pdf_report(result)
        export_stage = "validation"
        validate_pdf_report(report_data)
        st.download_button(
            "Download PDF report",
            report_data,
            file_name="resumematch-report.pdf",
            mime="application/pdf",
            width="stretch",
        )
    except Exception:
        logger.exception(
            "PDF export failed during %s (generated_bytes=%s)",
            export_stage,
            len(report_data) if report_data is not None else None,
        )
        st.warning("PDF export is unavailable, so a text report is ready instead.")
        st.download_button(
            "Download text report",
            text_report,
            file_name="resumematch-report.txt",
            mime="text/plain",
            width="stretch",
        )

    edited_resume = (
        combined_editor_text()
        if st.session_state.editor_ready
        else st.session_state.resume_text
    )
    st.download_button(
        "Download edited resume text",
        edited_resume,
        file_name="edited-resume.txt",
        mime="text/plain",
        width="stretch",
    )


def main() -> None:
    """Build the complete dashboard and handle scoring actions."""
    configure_page()
    initialize_state()
    render_sidebar()
    apply_theme()
    render_header()

    try:
        prepare_nltk_resources()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    upload_column, description_column = st.columns([1, 1.5])
    with upload_column:
        st.subheader("1. Upload your resume")
        resume_file = st.file_uploader(
            "PDF resume",
            type=["pdf"],
            help=(
                "Text-based PDFs are supported. Scanned PDFs can be up to "
                f"{MAX_OCR_PAGES} pages and "
                f"{MAX_OCR_FILE_BYTES // (1024 * 1024)} MB."
            ),
        )
    with description_column:
        st.subheader("2. Add the job description")
        job_description = st.text_area(
            "Job description",
            height=190,
            placeholder="Paste responsibilities, requirements, and preferred skills...",
            label_visibility="collapsed",
        )

    if st.button("Analyze Resume", type="primary", width="stretch"):
        analyze_upload(resume_file, job_description)

    if st.session_state.result is None:
        st.info(
            "Upload a resume and paste a job description once. Your analysis, "
            "editor, and downloads will appear below."
        )

    st.divider()
    overview_tab, keywords_tab, editor_tab, download_tab = st.tabs(
        ["Score Overview", "Keyword Analysis", "Resume Editor", "Download Report"]
    )
    result = st.session_state.result
    with overview_tab:
        if result is None:
            st.caption("Your score overview will appear after the first analysis.")
        else:
            render_score_overview(result)
    with keywords_tab:
        if result is None:
            st.caption("Your keyword analysis will appear after the first analysis.")
        else:
            render_keyword_analysis(result)
    with editor_tab:
        render_resume_editor(result)
    with download_tab:
        render_downloads(result)


if __name__ == "__main__":
    main()