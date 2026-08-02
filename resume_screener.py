#!/usr/bin/env python3
"""
resume_screener.py
-------------------
Command-line entry point for the AI Resume Screening System.

Usage
-----
    python resume_screener.py --jd data/job_description_sample.txt \
                               --resumes data/resumes \
                               --out output/ranked_resumes.csv

Accepts resumes in .pdf, .docx, or .txt format (any mix, in one folder).
"""

import argparse
import os
import sys

import pandas as pd

from core.text_extraction import extract_text, get_candidate_name, UnsupportedFileTypeError
from core.ranking_engine import rank_resumes

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_resumes_from_folder(folder_path: str) -> dict:
    """Read every supported resume file in a folder into {name: text}."""
    resumes = {}
    for filename in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        full_path = os.path.join(folder_path, filename)
        try:
            text = extract_text(full_path)
            if text.strip():
                candidate_name = get_candidate_name(full_path)
                resumes[candidate_name] = text
            else:
                print(f"  [!] Skipped '{filename}': no extractable text found.")
        except UnsupportedFileTypeError as e:
            print(f"  [!] Skipped '{filename}': {e}")
    return resumes


def main():
    parser = argparse.ArgumentParser(
        description="AI-based NLP Resume Screening System — ranks resumes against a job description."
    )
    parser.add_argument("--jd", required=True, help="Path to job description file (.txt, .pdf, or .docx)")
    parser.add_argument("--resumes", required=True, help="Path to folder containing resume files")
    parser.add_argument("--out", default="output/ranked_resumes.csv", help="Path to write ranked results CSV")
    parser.add_argument("--sim-weight", type=float, default=0.6, help="Weight for semantic similarity (default 0.6)")
    parser.add_argument("--skill-weight", type=float, default=0.4, help="Weight for skill match (default 0.4)")
    parser.add_argument("--top", type=int, default=None, help="Only show/save top N candidates")
    args = parser.parse_args()

    if not os.path.exists(args.jd):
        sys.exit(f"Error: job description file not found: {args.jd}")
    if not os.path.isdir(args.resumes):
        sys.exit(f"Error: resumes folder not found: {args.resumes}")

    print(f"Reading job description from: {args.jd}")
    job_description = extract_text(args.jd)
    if not job_description.strip():
        sys.exit("Error: could not extract any text from the job description file.")

    print(f"Loading resumes from: {args.resumes}")
    resumes = load_resumes_from_folder(args.resumes)
    if not resumes:
        sys.exit("Error: no valid resumes found in the given folder.")
    print(f"  Loaded {len(resumes)} resume(s).\n")

    print("Ranking resumes against job description (TF-IDF + skill matching)...")
    results = rank_resumes(
        job_description,
        resumes,
        similarity_weight=args.sim_weight,
        skill_weight=args.skill_weight,
    )

    if args.top:
        results = results[: args.top]

    # ---- Print summary table to console ----
    print("\n" + "=" * 90)
    print(f"{'Rank':<5}{'Candidate':<30}{'Final':<10}{'Similarity':<12}{'Skill Match':<12}{'Matched Skills'}")
    print("=" * 90)
    for rank, r in enumerate(results, start=1):
        matched_preview = ", ".join(sorted(r.matched_skills)) or "-"
        print(
            f"{rank:<5}{r.candidate[:28]:<30}{r.final_score_pct:<10}"
            f"{r.similarity_score_pct:<12}{r.skill_match_score_pct:<12}{matched_preview}"
        )
    print("=" * 90 + "\n")

    # ---- Write CSV ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for rank, r in enumerate(results, start=1):
        rows.append({
            "Rank": rank,
            "Candidate": r.candidate,
            "Final Score (%)": r.final_score_pct,
            "Semantic Similarity (%)": r.similarity_score_pct,
            "Skill Match (%)": r.skill_match_score_pct,
            "Matched Skills": ", ".join(sorted(r.matched_skills)),
            "Missing Skills": ", ".join(sorted(r.missing_skills)),
            "Additional Skills": ", ".join(sorted(r.extra_skills)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Ranked results saved to: {args.out}")


if __name__ == "__main__":
    main()
