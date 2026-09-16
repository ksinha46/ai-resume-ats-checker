"""Text extraction and offline resume-to-job-description analysis helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO, Callable

import nltk
import pdfplumber
import pymupdf
import pytesseract
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MAX_KEYWORDS = 18
MIN_KEYWORD_LENGTH = 2
KEYWORD_WEIGHT = 0.7
SIMILARITY_WEIGHT = 0.3
MAX_OCR_PAGES = 20
MAX_OCR_FILE_BYTES = 15 * 1024 * 1024
MIN_EMBEDDED_TEXT_WORDS = 2
MIN_SCANNED_IMAGE_COVERAGE = 0.5
SECTION_NAMES = ("summary", "skills", "experience", "education")
_STEMMER = PorterStemmer()


@dataclass(frozen=True)
class AnalysisResult:
    """All values needed to render one resume comparison."""

    score: float
    keyword_coverage: float
    similarity: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    keyword_weights: dict[str, float]
    section_scores: dict[str, float]
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


def _ocr_pdf_pages(
    pdf_data: bytes,
    progress_callback: Callable[[int, int], None] | None = None,
    page_indexes: list[int] | None = None,
) -> list[str]:
    """Return page-aligned OCR text for selected pages, including blank results."""
    if len(pdf_data) > MAX_OCR_FILE_BYTES:
        max_megabytes = MAX_OCR_FILE_BYTES // (1024 * 1024)
        raise ValueError(
            f"This scanned PDF is too large to process safely (limit: "
            f"{max_megabytes} MB). Compress the PDF, split it into smaller files, "
            "or upload a text-based PDF."
        )

    try:
        document = pymupdf.open(stream=pdf_data, filetype="pdf")
    except Exception as error:
        raise ValueError("We couldn't open that PDF for scanning.") from error

    page_text: list[str] = []
    try:
        indexes = (
            list(range(document.page_count))
            if page_indexes is None
            else page_indexes
        )
        total_ocr_pages = len(indexes)
        if total_ocr_pages > MAX_OCR_PAGES:
            raise ValueError(
                f"This scanned PDF has too many pages to process safely (limit: "
                f"{MAX_OCR_PAGES} pages). Upload only the resume pages, split the "
                "PDF into smaller files, or upload a text-based PDF."
            )
        for ocr_number, page_index in enumerate(indexes, start=1):
            if progress_callback:
                progress_callback(ocr_number, total_ocr_pages)
            page = document[page_index]
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            page_text.append(pytesseract.image_to_string(image))
    except ValueError:
        raise
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Scanning is temporarily unavailable because the local OCR tool "
            "could not be started."
        ) from error
    except Exception as error:
        raise RuntimeError(
            "We couldn't scan this PDF. Try a clearer scan or a text-based PDF."
        ) from error
    finally:
        document.close()

    return page_text


def _ocr_pdf(
    pdf_data: bytes,
    progress_callback: Callable[[int, int], None] | None = None,
    page_indexes: list[int] | None = None,
) -> str:
    """Render and OCR selected PDF pages, or every page when none are selected."""
    text = "\n".join(
        _ocr_pdf_pages(pdf_data, progress_callback, page_indexes)
    ).strip()
    if not text:
        raise ValueError(
            "We scanned this PDF but could not find readable text. Try a clearer "
            "scan with upright, high-contrast pages."
        )
    return text


def _has_large_image(page: object) -> bool:
    """Return whether one image covers enough of the page to plausibly be a scan."""
    page_width = float(getattr(page, "width", 0) or 0)
    page_height = float(getattr(page, "height", 0) or 0)
    page_area = page_width * page_height
    if page_area <= 0:
        return False

    for image in getattr(page, "images", ()):
        try:
            width = max(0.0, float(image["x1"]) - float(image["x0"]))
            height = max(0.0, float(image["bottom"]) - float(image["top"]))
        except (KeyError, TypeError, ValueError):
            continue
        if (width * height) / page_area >= MIN_SCANNED_IMAGE_COVERAGE:
            return True
    return False


def _has_substantive_embedded_text(text: str, page: object | None = None) -> bool:
    """Return whether embedded text is substantial enough to represent a page."""
    normalized_text = " ".join(text.split())
    if re.search(
        r"\b(?:scanned\s+(?:by|with|copy)|camscanner|scanbot)\b",
        normalized_text,
        flags=re.IGNORECASE,
    ):
        return False
    words = re.findall(r"\b[^\W\d_]+\b", normalized_text, flags=re.UNICODE)
    if len(words) >= MIN_EMBEDDED_TEXT_WORDS:
        return True
    return len(words) == 1 and page is not None and not _has_large_image(page)


def extract_text_from_pdf(
    pdf_file: BinaryIO,
    ocr_progress: Callable[[int, int], None] | None = None,
) -> str:
    """Extract embedded text and OCR pages without substantive embedded text."""
    try:
        pdf_file.seek(0)
        pdf_data = pdf_file.read()
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            pages = list(pdf.pages)
            page_text = [page.extract_text() or "" for page in pages]
    except Exception as error:
        raise ValueError(
            "We couldn't read that PDF. Please upload a valid PDF resume."
        ) from error

    image_only_page_indexes = [
        index
        for index, (text, page) in enumerate(zip(page_text, pages))
        if not _has_substantive_embedded_text(text, page)
    ]
    if image_only_page_indexes:
        ocr_text = _ocr_pdf_pages(
            pdf_data,
            ocr_progress,
            page_indexes=image_only_page_indexes,
        )
        for page_index, scanned_text in zip(image_only_page_indexes, ocr_text):
            page_text[page_index] = scanned_text
    text = "\n".join(page_text).strip()
    if not text:
        raise ValueError(
            "We scanned this PDF but could not find readable text. Try a clearer "
            "scan with upright, high-contrast pages."
        )
    return text


def _tokens(text: str) -> list[str]:
    """Return normalized words while preserving common technical terms."""
    return re.findall(r"[a-z0-9][a-z0-9+#.-]*", text.lower())


def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation, and remove English stopwords."""
    prepare_nltk_resources()
    english_stopwords = set(stopwords.words("english"))
    cleaned_words = [
        word
        for word in _tokens(text)
        if len(word) >= MIN_KEYWORD_LENGTH and word not in english_stopwords
    ]
    return " ".join(cleaned_words)


def _extract_ranked_keywords(job_description: str) -> list[tuple[str, float]]:
    """Rank meaningful job-description terms with TF-IDF importance."""
    vectorizer = TfidfVectorizer(
        stop_words=list(stopwords.words("english")),
        ngram_range=(1, 2),
        max_features=100,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.-]*\b",
    )
    try:
        matrix = vectorizer.fit_transform([job_description])
    except ValueError:
        return []

    ranked = sorted(
        zip(vectorizer.get_feature_names_out(), matrix.toarray()[0]),
        key=lambda item: item[1],
        reverse=True,
    )

    # Prefer specific phrases, while avoiding a list dominated by near-duplicates.
    selected: list[tuple[str, float]] = []
    selected_tokens: set[str] = set()
    for term, weight in ranked:
        term_tokens = set(term.split())
        if weight <= 0 or len(term) < MIN_KEYWORD_LENGTH:
            continue
        if len(term_tokens) == 1 and term in selected_tokens:
            continue
        selected.append((term, float(weight)))
        selected_tokens.update(term_tokens)
        if len(selected) == MAX_KEYWORDS:
            break
    return selected


def _term_matches(term: str, resume_text: str) -> bool:
    """Match exact phrases first, then allow conservative stem matching."""
    term_lower = term.lower()
    resume_lower = resume_text.lower()
    if re.search(rf"(?<!\w){re.escape(term_lower)}(?!\w)", resume_lower):
        return True

    resume_stems = {_STEMMER.stem(token) for token in _tokens(resume_lower)}
    term_stems = [_STEMMER.stem(token) for token in _tokens(term_lower)]
    return bool(term_stems) and all(stem in resume_stems for stem in term_stems)


def split_resume_sections(resume_text: str) -> dict[str, str]:
    """Split common resume headings into editable sections when detectable."""
    sections = {name: "" for name in SECTION_NAMES}
    heading_pattern = re.compile(
        r"(?im)^\s*(professional\s+summary|profile|objective|summary|"
        r"technical\s+skills|core\s+competencies|skills|"
        r"work\s+experience|professional\s+experience|employment|experience|"
        r"academic\s+background|education)\s*:?\s*$"
    )
    matches = list(heading_pattern.finditer(resume_text))
    if not matches:
        sections["summary"] = resume_text.strip()
        return sections

    prefix = resume_text[: matches[0].start()].strip()
    if prefix:
        sections["summary"] = prefix

    for index, match in enumerate(matches):
        heading = match.group(1).lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(resume_text)
        content = resume_text[match.end() : end].strip()
        if "skill" in heading or "competenc" in heading:
            section = "skills"
        elif "experience" in heading or "employment" in heading:
            section = "experience"
        elif "education" in heading or "academic" in heading:
            section = "education"
        else:
            section = "summary"
        sections[section] = "\n".join(filter(None, [sections[section], content]))
    return sections


def _coverage_for_text(text: str, keywords: list[str]) -> float:
    """Calculate unweighted keyword coverage for a block of resume text."""
    if not text.strip() or not keywords:
        return 0.0
    matched = sum(_term_matches(keyword, text) for keyword in keywords)
    return round(matched / len(keywords) * 100, 1)


def _get_suggestions(score: float, missing_keywords: list[str]) -> list[str]:
    """Create concise recommendations based on score and missing terms."""
    if score < 45:
        suggestions = [
            "Tailor your summary to the target role and lead with your most relevant strengths.",
            "Add relevant missing skills to your experience bullets with evidence of how you used them.",
        ]
    elif score < 70:
        suggestions = [
            "Use the job description's terminology where it accurately reflects your experience.",
            "Strengthen experience bullets with measurable outcomes and role-specific skills.",
        ]
    else:
        suggestions = [
            "Keep the strong alignment and verify that every keyword is supported by real experience.",
            "Use simple headings, consistent dates, and measurable results to keep the resume ATS-friendly.",
        ]
    if missing_keywords:
        suggestions.append(
            "Prioritize these relevant missing terms: "
            + ", ".join(missing_keywords[:3])
            + "."
        )
    return suggestions[:3]


def analyze_resume(resume_text: str, job_description: str) -> AnalysisResult:
    """Calculate a keyword-led ATS alignment score and supporting details."""
    if not resume_text.strip():
        raise ValueError("The resume does not contain readable text.")
    if not job_description.strip():
        raise ValueError("Please provide a job description to compare.")

    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_description)
    if not cleaned_resume or not cleaned_job:
        raise ValueError("There is not enough readable text to compare these documents.")

    ranked_keywords = _extract_ranked_keywords(cleaned_job)
    if not ranked_keywords:
        raise ValueError("The job description needs more specific readable details.")

    matched_keywords = [
        term for term, _ in ranked_keywords if _term_matches(term, cleaned_resume)
    ]
    missing_keywords = [
        term for term, _ in ranked_keywords if term not in matched_keywords
    ]
    total_weight = sum(weight for _, weight in ranked_keywords)
    matched_weight = sum(
        weight for term, weight in ranked_keywords if term in matched_keywords
    )
    coverage = matched_weight / total_weight if total_weight else 0.0

    try:
        vectors = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(
            [cleaned_resume, cleaned_job]
        )
        similarity = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    except ValueError as error:
        raise ValueError(
            "There is not enough unique text to calculate a match score."
        ) from error

    raw_score = (coverage * KEYWORD_WEIGHT) + (similarity * SIMILARITY_WEIGHT)
    score = round(max(0.0, min(1.0, raw_score)) * 100, 1)
    sections = split_resume_sections(resume_text)
    keyword_names = [term for term, _ in ranked_keywords]
    section_scores = {
        name.title(): _coverage_for_text(text, keyword_names)
        for name, text in sections.items()
        if text.strip()
    }
    if not section_scores:
        section_scores = {"Overall resume": round(coverage * 100, 1)}

    return AnalysisResult(
        score=score,
        keyword_coverage=round(coverage * 100, 1),
        similarity=round(similarity * 100, 1),
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
        keyword_weights={term: weight for term, weight in ranked_keywords},
        section_scores=section_scores,
        suggestions=_get_suggestions(score, missing_keywords),
    )


def get_score_label(score: float) -> str:
    """Return a plain-language interpretation of the numeric match score."""
    if score < 45:
        return "Needs more alignment"
    if score < 70:
        return "A solid starting point"
    return "Strong alignment"