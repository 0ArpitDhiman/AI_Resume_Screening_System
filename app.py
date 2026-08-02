"""
app.py
------
Streamlit web app for the AI Resume Screening System.

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from core.text_extraction import extract_text, get_candidate_name, UnsupportedFileTypeError
from core.ranking_engine import rank_resumes
from core.skills_database import skill_category

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 AI Resume Screening System")
st.caption(
    "NLP-based system that ranks resumes against a job description using "
    "TF-IDF semantic similarity + skill matching."
)

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown(
        """
        **Domain:** Human Resources
        **Problem:** HR teams spend significant time screening resumes manually.
        **Solution:** This system uses NLP to automatically parse resumes and job
        descriptions, extract skills, compute semantic similarity, and rank
        candidates — cutting screening time from hours to seconds.

        **Tech used:** Python, scikit-learn (TF-IDF + cosine similarity),
        regex-based skill extraction, pdfplumber (PDF parsing),
        python-docx (DOCX parsing), Streamlit (UI), Pandas.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
col_jd, col_settings = st.columns([2, 1])

with col_jd:
    st.subheader("1. Job Description")
    jd_input_mode = st.radio(
        "Provide the job description via:",
        ["Paste text", "Upload file (.pdf/.docx/.txt)"],
        horizontal=True,
    )

    job_description = ""
    if jd_input_mode == "Paste text":
        job_description = st.text_area(
            "Paste the job description here",
            height=220,
            placeholder="e.g. We are looking for a Machine Learning Engineer with NLP experience...",
        )
    else:
        jd_file = st.file_uploader(
            "Upload job description file", type=["pdf", "docx", "txt"], key="jd_file"
        )
        if jd_file is not None:
            try:
                job_description = extract_text(jd_file)
                with st.expander("Preview extracted JD text"):
                    st.text(job_description[:2000])
            except UnsupportedFileTypeError as e:
                st.error(str(e))

with col_settings:
    st.subheader("2. Scoring Weights")
    sim_weight = st.slider(
        "Semantic similarity weight", 0.0, 1.0, 0.6, 0.05,
        help="How much weight to give overall contextual/semantic match (TF-IDF cosine similarity)."
    )
    skill_weight = round(1.0 - sim_weight, 2)
    st.write(f"Skill match weight: **{skill_weight}**")
    top_n = st.number_input("Show top N candidates", min_value=1, max_value=100, value=10)

st.subheader("3. Upload Resumes")
resume_files = st.file_uploader(
    "Upload one or more resumes (.pdf, .docx, .txt)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

st.divider()

run_btn = st.button("🔍 Rank Resumes", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_btn:
    if not job_description or not job_description.strip():
        st.error("Please provide a job description (paste text or upload a file).")
    elif not resume_files:
        st.error("Please upload at least one resume.")
    else:
        MIN_WORDS = 15  # below this, extraction almost certainly failed/garbled

        resumes = {}
        skipped = []
        low_text_warnings = []
        diagnostics_rows = []  # (label, word_count) — built once, reused below (no re-reading files)

        for idx, f in enumerate(resume_files):
            try:
                text = extract_text(f)  # read exactly once per file
                word_count = len(text.split())
                diagnostics_rows.append((f.name, word_count))
                if word_count >= MIN_WORDS:
                    name = get_candidate_name(f, fallback_index=idx)
                    # De-duplicate names if two files share a filename
                    original_name = name
                    suffix = 1
                    while name in resumes:
                        suffix += 1
                        name = f"{original_name} ({suffix})"
                    resumes[name] = text
                elif text.strip():
                    low_text_warnings.append((f.name, word_count))
                else:
                    skipped.append(f.name)
            except UnsupportedFileTypeError:
                skipped.append(f.name)
                diagnostics_rows.append((f.name, "unsupported file type"))

        if skipped:
            st.warning(f"Skipped unreadable file(s): {', '.join(skipped)}")

        if low_text_warnings:
            for fname, wc in low_text_warnings:
                st.warning(
                    f"⚠️ '{fname}' only yielded **{wc} word(s)** of extractable text — "
                    "it was excluded to avoid a misleading zero score. This usually means "
                    "the file is a scanned/image-based PDF (no real text layer) rather than "
                    "a text-based document. Try re-saving/exporting it as a text-based PDF "
                    "or DOCX, or paste the content as plain text instead."
                )

        jd_word_count = len(job_description.split())
        if jd_word_count < MIN_WORDS:
            st.warning(
                f"⚠️ The job description only has **{jd_word_count} word(s)** of extractable "
                "text. If you uploaded a file, it may be a scanned/image-based PDF. Scores "
                "will be unreliable until this has real text."
            )

        with st.expander("🔧 Diagnostics — extracted text stats", expanded=bool(low_text_warnings or jd_word_count < MIN_WORDS)):
            st.write(f"**Job description:** {jd_word_count} words extracted")
            for fname, wc in diagnostics_rows:
                st.write(f"**{fname}:** {wc} words extracted" if isinstance(wc, int) else f"**{fname}:** {wc}")

        if not resumes:
            st.error(
                "No resume had enough extractable text to score. See the diagnostics panel "
                "above — this is almost always a scanned/image PDF issue, not a bug in the "
                "scoring logic."
            )
        else:
            with st.spinner("Running NLP pipeline (TF-IDF + skill matching)..."):
                results = rank_resumes(
                    job_description, resumes,
                    similarity_weight=sim_weight, skill_weight=skill_weight,
                )

            results = results[: int(top_n)]

            st.success(f"Ranked {len(resumes)} resume(s) against the job description.")

            # ---- Summary table ----
            df = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Candidate": r.candidate,
                    "Final Score (%)": r.final_score_pct,
                    "Semantic Similarity (%)": r.similarity_score_pct,
                    "Skill Match (%)": r.skill_match_score_pct,
                    "Matched Skills": len(r.matched_skills),
                    "Missing Skills": len(r.missing_skills),
                }
                for i, r in enumerate(results)
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # ---- Bar chart ----
            st.bar_chart(df.set_index("Candidate")[["Final Score (%)"]])

            # ---- Download CSV ----
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download ranked results as CSV",
                data=csv_bytes,
                file_name="ranked_resumes.csv",
                mime="text/csv",
            )

            st.divider()
            st.subheader("Candidate Details")

            for i, r in enumerate(results):
                with st.expander(f"#{i + 1} — {r.candidate}  ·  Final Score: {r.final_score_pct}%"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Final Score", f"{r.final_score_pct}%")
                    c2.metric("Semantic Similarity", f"{r.similarity_score_pct}%")
                    c3.metric("Skill Match", f"{r.skill_match_score_pct}%")

                    st.markdown("**✅ Matched Skills**")
                    st.write(", ".join(sorted(r.matched_skills)) or "None")

                    st.markdown("**❌ Missing Skills (required by JD, not found in resume)**")
                    st.write(", ".join(sorted(r.missing_skills)) or "None")

                    if r.extra_skills:
                        st.markdown("**➕ Additional Skills (in resume, not in JD)**")
                        st.write(", ".join(sorted(r.extra_skills)))

                    st.markdown("**Resume preview**")
                    st.caption(r.raw_text_preview + "...")

st.divider()
st.caption(
    "MP Online AIML Internship — Capstone Project · Domain: Human Resources · "
    "AI Resume Screening System"
)
