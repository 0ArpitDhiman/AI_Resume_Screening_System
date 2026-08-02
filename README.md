# AI Resume Screening System

**Domain:** Human Resources
**Problem Statement:** HR teams spend significant time screening resumes manually. This project develops an NLP-based AI system that ranks resumes based on job descriptions and candidate skills.


---

## What it does

1. Takes a **job description** (pasted text or uploaded `.pdf` / `.docx` / `.txt`).
2. Takes a batch of **resumes** (`.pdf`, `.docx`, `.txt`, any mix).
3. Cleans and preprocesses all text with NLP techniques (tokenization, stopword removal, normalization).
4. Vectorizes the job description and every resume using **TF-IDF** and computes **cosine similarity** to measure semantic match.
5. Extracts known **skills** from both the JD and each resume using a curated skills taxonomy, and computes a skill-overlap score.
6. Combines both signals into a weighted **Final Score** and ranks candidates from best to worst fit.
7. Shows, for every candidate: matched skills, missing (required) skills, and extra skills not mentioned in the JD.

## Tech stack

| Purpose | Tool / Library |
|---|---|
| Language | Python 3 |
| NLP / text preprocessing | Custom regex-based cleaning + tokenization, scikit-learn stopword list |
| Feature extraction | TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`) |
| Similarity scoring | Cosine similarity (`sklearn.metrics.pairwise`) |
| Skill extraction | Regex pattern matching against a curated skills taxonomy |
| Resume/JD parsing | `pdfplumber` (PDF), `python-docx` (DOCX), built-in file I/O (TXT) |
| Web app / UI | Streamlit |
| Data handling | Pandas |

No NLTK/spaCy model downloads are required — preprocessing is done with lightweight, dependency-free regex logic, so the project runs on any machine immediately after `pip install -r requirements.txt`, with no extra download step.

## Project structure

```
resume_screening_project/
├── app.py                     # Streamlit web app
├── resume_screener.py         # Command-line version
├── core/
│   ├── text_extraction.py     # PDF / DOCX / TXT parsing
│   ├── preprocessing.py       # NLP cleaning, tokenization, stopwords
│   ├── skills_database.py     # Skill taxonomy + extraction logic
│   └── ranking_engine.py      # TF-IDF + cosine similarity + skill scoring
├── data/
│   ├── job_description_sample.txt
│   └── resumes/                # 5 sample resumes (.txt, .docx, .pdf mix)
├── output/                     # Ranked CSV results are written here
├── requirements.txt
└── README.md
```

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Option A — Web app (recommended for demo)
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`), paste/upload a job description, upload resumes, and click **Rank Resumes**.

### 2. Option B — Command line
```bash
python resume_screener.py --jd data/job_description_sample.txt --resumes data/resumes --out output/ranked_resumes.csv
```
This prints a ranked table to the console and saves a CSV to `output/ranked_resumes.csv`.

Optional flags:
```
--sim-weight 0.6      # weight given to semantic similarity (default 0.6)
--skill-weight 0.4    # weight given to skill match (default 0.4)
--top 5               # only show/save the top N candidates
```

## Sample data included

- `data/job_description_sample.txt` — an ML Engineer (NLP focus) job description
- `data/resumes/` — 5 sample resumes with varying degrees of fit, covering all 3 supported formats:
  - `Ananya_Sharma.txt` — strong match
  - `Rohit_Verma.txt` — moderate match
  - `Priya_Nair.txt` — weak match / different domain (marketing)
  - `Karan_Mehta.docx` — strong match
  - `Sneha_Iyer.pdf` — strong match

Run the CLI command above out of the box to see it working end-to-end.

## How scoring works

```
Final Score = (similarity_weight × Semantic Similarity) + (skill_weight × Skill Match Ratio)
```

- **Semantic Similarity** — TF-IDF cosine similarity between the whole JD and the whole resume (0–1). Captures overall contextual relevance, not just exact keyword hits.
- **Skill Match Ratio** — fraction of skills required by the JD (from the skills taxonomy) that are also found in the resume (0–1). Captures hard, named requirements that a pure similarity score might dilute.

Default weighting is 60% similarity / 40% skill match, both adjustable via the UI slider or CLI flags.

## Possible extensions (future work)

- Swap the static skills taxonomy for a larger external one (ESCO / O*NET / LinkedIn Skills API).
- Add named-entity recognition (spaCy) to auto-extract candidate name, education, and years of experience.
- Use sentence embeddings (e.g. Sentence-BERT) instead of / alongside TF-IDF for deeper semantic matching.
- Add a persistent database to store screened candidates across sessions.
- Add authentication and role-based access for HR teams.
