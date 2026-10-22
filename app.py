
import re

import fitz
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ResumeIQ | AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# COLOR PALETTE
# Navy: #172B4D | Teal: #0F9D8A
# Background: #F6F8FC | Cards: #FFFFFF
# =========================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --navy: #172B4D;
        --teal: #0F9D8A;
        --teal-dark: #0B7F70;
        --teal-light: #CCFBF1;
        --background: #F6F8FC;
        --white: #FFFFFF;
        --text: #334155;
        --muted: #64748B;
        --line: #E2E8F0;
        --red-bg: #FFF1F2;
        --red-text: #9F1239;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    .block-container {
        max-width: 1240px;
        padding: 2rem 2rem 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--navy) !important;
        letter-spacing: -0.4px;
    }

    h1 {
        font-weight: 800 !important;
    }

    .hero {
        background: linear-gradient(120deg, #172B4D 0%, #263F60 100%);
        padding: 2.5rem;
        border-radius: 22px;
        margin-bottom: 1.8rem;
        color: #FFFFFF;
        border: 1px solid #344D6C;
        box-shadow: 0 10px 30px rgba(23, 43, 77, 0.10);
    }

    .hero-label {
        color: #99F6E4;
        text-transform: uppercase;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        margin-bottom: 0.8rem;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(2rem, 4vw, 2.8rem);
        line-height: 1.2;
        margin: 0 0 0.8rem;
    }

    .hero p {
        color: #DCE6F1;
        font-size: 1rem;
        line-height: 1.8;
        max-width: 700px;
        margin-bottom: 0;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: var(--navy);
        font-size: 1.25rem;
        font-weight: 800;
        margin: 1.6rem 0 0.5rem;
    }

    .subtle {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.7;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem;
        min-height: 140px;
        box-shadow: 0 4px 15px rgba(23, 43, 77, 0.04);
        height: 100%;
    }

    .metric-label {
        color: #64748B;
        font-size: 0.83rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .metric-value {
        font-family: 'Manrope', sans-serif;
        color: #172B4D;
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-note {
        color: #64748B;
        font-size: 0.77rem;
        line-height: 1.5;
        margin-top: 0.6rem;
    }

    .panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        font-family: 'Manrope', sans-serif;
        color: #172B4D;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }

    .skill-chip {
        display: inline-block;
        padding: 0.4rem 0.7rem;
        margin: 0.2rem 0.2rem 0.2rem 0;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .chip-match {
        background: #CCFBF1;
        color: #115E59;
        border: 1px solid #99F6E4;
    }

    .chip-missing {
        background: #FFF1F2;
        color: #9F1239;
        border: 1px solid #FECDD3;
    }

    .chip-neutral {
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
    }

    .status-pill {
        display: inline-block;
        background: #CCFBF1;
        color: #115E59;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .tip {
        background: #F0FDFA;
        border: 1px solid #99F6E4;
        border-left: 4px solid #0F9D8A;
        padding: 1rem;
        border-radius: 10px;
        color: #134E4A;
        font-size: 0.9rem;
        line-height: 1.7;
        margin: 0.6rem 0 0.8rem;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.8;
        padding-top: 2rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 2.5rem;
    }

    div.stButton > button[kind="primary"] {
        background: #0F9D8A;
        border: 1px solid #0F9D8A;
        border-radius: 11px;
        color: #FFFFFF;
        font-weight: 700;
        min-height: 3.1rem;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #0B7F70;
        border-color: #0B7F70;
        color: #FFFFFF;
        box-shadow: 0 5px 15px rgba(15, 157, 138, 0.18);
    }

    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #94A3B8;
        border-radius: 12px;
        padding: 0.8rem;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        background: #FFFFFF;
        color: #172B4D;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #0F9D8A;
        box-shadow: 0 0 0 1px #0F9D8A;
    }

    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: #0F9D8A;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.8rem 2rem;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 16px;
        }

        .metric-card {
            padding: 0.9rem;
            min-height: 120px;
        }

        .metric-value {
            font-size: 1.65rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SKILL DICTIONARY
# =========================================================

SKILL_ALIASES = {
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++"],
    "C#": ["c#"],
    "SQL": ["sql", "structured query language"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript"],
    "HTML": ["html"],
    "CSS": ["css"],
    "React": ["react", "react.js"],
    "Node.js": ["node.js", "nodejs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "Streamlit": ["streamlit"],
    "REST API": ["rest api", "restful api"],
    "Git": ["git"],
    "GitHub": ["github"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["microsoft azure", "azure"],
    "Google Cloud": ["google cloud", "gcp"],
    "Linux": ["linux"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Artificial Intelligence": ["artificial intelligence", "ai"],
    "NLP": ["natural language processing", "nlp"],
    "Computer Vision": ["computer vision"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Visualization": ["data visualization"],
    "Power BI": ["power bi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "microsoft excel"],
    "Statistics": ["statistics", "statistical analysis"],
    "Problem Solving": ["problem solving", "problem-solving"],
    "Communication": ["communication", "communication skills"],
    "Project Management": ["project management"],
    "Agile": ["agile"],
    "Scrum": ["scrum"],
    "Testing": ["software testing", "testing"],
    "CI/CD": [
        "ci/cd",
        "continuous integration",
        "continuous deployment",
    ],
}


# =========================================================
# ANALYSIS FUNCTIONS
# =========================================================

def extract_pdf_text(uploaded_file):
    """Extract selectable text from an uploaded PDF."""
    with fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf",
    ) as pdf:
        return "\n".join(
            page.get_text() for page in pdf
        ).strip()


def contains_phrase(text, phrase):
    """Find phrases without matching inside unrelated words."""
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def detect_skills(text):
    """Return canonical skill names detected in the text."""
    found = set()

    for skill, aliases in SKILL_ALIASES.items():
        if any(
            contains_phrase(text, alias)
            for alias in aliases
        ):
            found.add(skill)

    return found


def classify_job_skills(job_description):
    """Classify skills under explicitly labeled sections."""
    required = set()
    preferred = set()
    all_skills = detect_skills(job_description)

    section = "other"

    required_headings = [
        "required qualifications",
        "required skills",
        "requirements",
        "must have",
        "minimum qualifications",
        "essential skills",
    ]

    preferred_headings = [
        "preferred qualifications",
        "preferred skills",
        "nice to have",
        "good to have",
        "desirable skills",
        "bonus skills",
    ]

    for line in job_description.splitlines():
        text = line.strip().lower()

        if not text:
            continue

        if any(h in text for h in preferred_headings):
            section = "preferred"

        elif any(h in text for h in required_headings):
            section = "required"

        elif text.endswith(":"):
            section = "other"

        skills_on_line = detect_skills(line)

        if section == "required":
            required.update(skills_on_line)

        elif section == "preferred":
            preferred.update(skills_on_line)

    preferred -= required
    unclassified = all_skills - required - preferred

    return required, preferred, unclassified


def calculate_similarity(resume_text, job_description):
    """Calculate TF-IDF cosine similarity as a percentage."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )

        vectors = vectorizer.fit_transform(
            [resume_text, job_description]
        )

        score = cosine_similarity(
            vectors[0],
            vectors[1],
        )[0][0]

        return round(float(score) * 100, 1)

    except ValueError:
        return 0.0


def create_suggestions(missing_skills, missing_required):
    """Create practical, evidence-based suggestions."""
    suggestions = []

    if missing_required:
        suggestions.append(
            "Review these required skills not detected: "
            + ", ".join(missing_required)
            + ". If you have relevant experience, make it clearer "
            "in your skills, projects, or work-history sections."
        )

    if missing_skills:
        suggestions.append(
            "Review other job skills not detected: "
            + ", ".join(missing_skills)
            + ". Include them only if they accurately reflect "
            "your skills or experience."
        )

    if not suggestions:
        suggestions.append(
            "All recognized job skills were detected. Strengthen "
            "your resume with specific project examples and "
            "measurable results."
        )

    suggestions.append(
        "Where possible, describe outcomes with numbers, such as "
        "time saved, accuracy improved, costs reduced, or users supported."
    )

    suggestions.append(
        "Use clear headings, consistent dates, readable formatting, "
        "and accurate contact information."
    )

    return suggestions


def render_skill_chips(skills, css_class):
    """Display skills as visual badges."""
    if not skills:
        st.caption("None detected.")
        return

    # Escape skill names before inserting them into HTML.
    from html import escape

    chips = "".join(
        f'<span class="skill-chip {css_class}">'
        f'{escape(skill)}</span>'
        for skill in skills
    )

    st.markdown(
        chips,
        unsafe_allow_html=True,
    )


def render_metric(label, value, note):
    """Render a consistent score card."""
    from html import escape

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# MAIN HEADER — NO NAVIGATION SIDEBAR
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">RESUME INTELLIGENCE PLATFORM</div>
        <h1>Make your next application stronger.</h1>
        <p>
            Understand how your resume aligns with a job description.
            Discover matching skills, identify gaps, and find practical
            ways to improve your application.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading">Start your analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtle">Add your resume and the job description you are targeting.</p>',
    unsafe_allow_html=True,
)


# =========================================================
# INPUT AREA
# =========================================================

left, right = st.columns(
    [0.9, 1.1],
    gap="large",
)

with left:
    st.markdown("#### 📄 Your resume")

    uploaded_file = st.file_uploader(
        "Upload resume PDF",
        type=["pdf"],
        help="Choose a text-based PDF resume.",
    )

    if uploaded_file:
        st.success(
            f"Ready to analyze: {uploaded_file.name}"
        )

        st.caption(
            f"File size: {uploaded_file.size / 1024:.1f} KB"
        )

    else:
        st.markdown(
            '<div class="tip">Upload a PDF resume to begin. '
            'Text-based PDFs work best. Scanned documents may '
            'require OCR.</div>',
            unsafe_allow_html=True,
        )


with right:
    st.markdown("#### 🎯 Target job")

    job_description = st.text_area(
        "Job description",
        height=220,
        placeholder=(
            "Paste the complete job description here...\n\n"
            "Required Skills:\n"
            "Python\n"
            "SQL\n"
            "Machine Learning\n\n"
            "Preferred Skills:\n"
            "Docker\n"
            "AWS"
        ),
    )


st.markdown("")

analyze_clicked = st.button(
    "✨  Analyze my resume",
    type="primary",
    use_container_width=True,
)

st.caption(
    "The analysis is an estimate based on recognized skills and "
    "text similarity. It is not an official ATS score or a hiring prediction."
)


# =========================================================
# RUN ANALYSIS
# =========================================================

if analyze_clicked:

    if uploaded_file is None:
        st.error("Please upload your resume PDF first.")
        st.stop()

    if not job_description.strip():
        st.error("Please paste the target job description first.")
        st.stop()

    try:
        with st.spinner(
            "Reading your resume and comparing skills..."
        ):
            resume_text = extract_pdf_text(uploaded_file)

            if len(resume_text) < 30:
                st.error(
                    "We could not extract enough text from this PDF. "
                    "It may be scanned or image-based. Try a text-based PDF."
                )
                st.stop()

            resume_skills = detect_skills(resume_text)
            job_skills = detect_skills(job_description)

            (
                required_skills,
                preferred_skills,
                unclassified_skills,
            ) = classify_job_skills(job_description)

            matched_skills = sorted(
                resume_skills & job_skills
            )

            missing_skills = sorted(
                job_skills - resume_skills
            )

            matched_required = sorted(
                resume_skills & required_skills
            )

            missing_required = sorted(
                required_skills - resume_skills
            )

            matched_preferred = sorted(
                resume_skills & preferred_skills
            )

            missing_preferred = sorted(
                preferred_skills - resume_skills
            )

            similarity = calculate_similarity(
                resume_text,
                job_description,
            )

            if job_skills:
                skill_coverage = (
                    len(matched_skills) / len(job_skills)
                ) * 100

                ats_score = round(
                    0.70 * skill_coverage
                    + 0.30 * similarity,
                    1,
                )

                score_method = (
                    "70% skill coverage + 30% text similarity"
                )

            else:
                skill_coverage = None
                ats_score = similarity

                score_method = (
                    "Text similarity only: no recognized job skills"
                )

            suggestions = create_suggestions(
                missing_skills,
                missing_required,
            )

    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.stop()


    # =====================================================
    # RESULTS HEADER
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">Your resume insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="status-pill">✓ Analysis complete</span>',
        unsafe_allow_html=True,
    )

    st.markdown("")


    # =====================================================
    # SCORE CARDS
    # =====================================================

    c1, c2, c3 = st.columns(
        3,
        gap="medium",
    )

    with c1:
        render_metric(
            "ATS-style estimate",
            f"{ats_score:.1f}%",
            "Project-specific estimate",
        )

    with c2:
        render_metric(
            "Skill coverage",
            (
                f"{skill_coverage:.1f}%"
                if skill_coverage is not None
                else "N/A"
            ),
            (
                f"{len(matched_skills)} of "
                f"{len(job_skills)} skills detected"
                if job_skills
                else "No recognized job skills"
            ),
        )

    with c3:
        render_metric(
            "Text similarity",
            f"{similarity:.1f}%",
            "TF-IDF cosine similarity",
        )

    st.markdown(
        f'<p class="subtle">Scoring method: {score_method}</p>',
        unsafe_allow_html=True,
    )

    if not job_skills:
        st.warning(
            "No skills from the current dictionary were recognized "
            "in the job description. The estimate uses text similarity "
            "only and may not reflect actual job requirements."
        )


    # =====================================================
    # COVERAGE BAR
    # =====================================================

    if skill_coverage is not None:
        st.markdown(
            '<div class="section-heading">Skill coverage overview</div>',
            unsafe_allow_html=True,
        )

        st.progress(
            int(round(skill_coverage)),
            text=(
                f"{skill_coverage:.1f}% of recognized "
                "job skills detected"
            ),
        )


    # =====================================================
    # SKILL MATCHING
    # =====================================================

    st.markdown(
        '<div class="section-heading">Skill match breakdown</div>',
        unsafe_allow_html=True,
    )

    match_col, gap_col = st.columns(
        2,
        gap="medium",
    )

    with match_col:
        with st.container(border=True):
            st.markdown("#### ✅ Matching skills")

            render_skill_chips(
                matched_skills,
                "chip-match",
            )

            st.caption(
                "These recognized skills appear in both "
                "the resume and the job description."
            )

    with gap_col:
        with st.container(border=True):
            st.markdown("#### 🔎 Skills not detected")

            render_skill_chips(
                missing_skills,
                "chip-missing",
            )

            st.caption(
                "These job-description skills were not detected. "
                "That does not necessarily mean you lack them."
            )


    # =====================================================
    # REQUIRED AND PREFERRED SKILLS
    # =====================================================

    st.markdown(
        '<div class="section-heading">Required vs. preferred</div>',
        unsafe_allow_html=True,
    )

    req_col, pref_col = st.columns(
        2,
        gap="medium",
    )

    with req_col:
        with st.container(border=True):
            st.markdown("#### 🔴 Required skills")

            if required_skills:
                st.caption(
                    f"{len(matched_required)} of "
                    f"{len(required_skills)} detected"
                )

                render_skill_chips(
                    matched_required,
                    "chip-match",
                )

                if missing_required:
                    st.caption("Not detected:")

                    render_skill_chips(
                        missing_required,
                        "chip-missing",
                    )
                else:
                    st.success(
                        "All recognized required skills detected."
                    )

            else:
                st.info(
                    "No explicitly labeled required-skills section found."
                )

    with pref_col:
        with st.container(border=True):
            st.markdown("#### 🟠 Preferred skills")

            if preferred_skills:
                st.caption(
                    f"{len(matched_preferred)} of "
                    f"{len(preferred_skills)} detected"
                )

                render_skill_chips(
                    matched_preferred,
                    "chip-match",
                )

                if missing_preferred:
                    st.caption("Not detected:")

                    render_skill_chips(
                        missing_preferred,
                        "chip-missing",
                    )
                else:
                    st.success(
                        "All recognized preferred skills detected."
                    )

            else:
                st.info(
                    "No explicitly labeled preferred-skills section found."
                )


    if unclassified_skills:
        with st.expander("Other recognized job skills"):
            render_skill_chips(
                sorted(unclassified_skills),
                "chip-neutral",
            )


    # =====================================================
    # IMPROVEMENT SUGGESTIONS
    # =====================================================

    st.markdown(
        '<div class="section-heading">Your improvement plan</div>',
        unsafe_allow_html=True,
    )

    for index, suggestion in enumerate(
        suggestions,
        start=1,
    ):
        from html import escape

        st.markdown(
            f"""
            <div class="panel">
                <div style="color:#0F9D8A;font-weight:800;
                            font-size:0.78rem;margin-bottom:0.4rem;">
                    SUGGESTION {index:02d}
                </div>
                <div style="color:#334155;line-height:1.7;">
                    {escape(suggestion)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # =====================================================
    # DETAIL SECTIONS
    # =====================================================

    with st.expander("View all skills detected in your resume"):
        render_skill_chips(
            sorted(resume_skills),
            "chip-neutral",
        )

    with st.expander("View all detected job-description skills"):
        render_skill_chips(
            sorted(job_skills),
            "chip-neutral",
        )

    with st.expander("View extracted resume text"):
        st.text(resume_text)

    with st.expander("How is the score calculated?"):
        st.markdown(
            """
            **Skill coverage**

            The number of recognized job skills found in the resume,
            divided by the number of recognized job skills.

            **Text similarity**

            TF-IDF cosine similarity compares the wording in the
            resume with the job description.

            **ATS-style estimate**

            When recognized job skills are present:

            - 70% skill coverage
            - 30% text similarity

            These weights are illustrative and have not been validated
            against commercial ATS results. The score does not predict
            whether an employer will interview or hire a candidate.
            """
        )

    st.markdown(
        """
        <div class="tip">
            <b>Pro tip:</b> Tailor your resume truthfully for each role.
            Prioritize relevant experience, project evidence, and
            measurable achievements rather than adding keywords alone.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <b style="color:#172B4D;">ResumeIQ</b><br>
        AI Resume Analyzer · Built with Python and Streamlit<br>
        For career guidance, not hiring decisions
    </div>
    """,
    unsafe_allow_html=True,
)
