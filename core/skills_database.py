"""
skills_database.py
-------------------
A curated taxonomy of technical and soft skills used to extract and match
skills mentioned in job descriptions and resumes.

For a production system this could be swapped for a larger external
taxonomy (e.g. ESCO, LinkedIn Skills API, O*NET) without changing any
other part of the pipeline -- everything downstream just consumes the
Python set returned by SKILL_DB.
"""

import re

# ---------------------------------------------------------------------------
# Skill taxonomy grouped by category (purely for readability / future UI use)
# ---------------------------------------------------------------------------
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "javascript", "typescript", "sql",
        "r", "go", "golang", "scala", "kotlin", "php", "ruby", "swift",
        "matlab", "bash", "shell scripting"
    ],
    "AI / ML / Data Science": [
        "machine learning", "deep learning", "natural language processing",
        "nlp", "computer vision", "data science", "data analysis",
        "data visualization", "statistics", "predictive modeling",
        "neural networks", "reinforcement learning", "generative ai",
        "large language models", "llm", "feature engineering",
        "model deployment", "mlops", "time series analysis"
    ],
    "ML/NLP Frameworks & Libraries": [
        "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras",
        "nltk", "spacy", "huggingface", "transformers", "opencv",
        "pandas", "numpy", "matplotlib", "seaborn", "xgboost", "lightgbm",
        "gensim", "word2vec", "bert", "tf-idf", "langchain"
    ],
    "Web / Backend": [
        "html", "css", "react", "angular", "vue", "node.js", "django",
        "flask", "fastapi", "streamlit", "rest api", "graphql",
        "microservices"
    ],
    "Databases": [
        "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis",
        "elasticsearch", "cassandra", "firebase"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "ci/cd", "jenkins", "git", "github", "gitlab", "linux",
        "terraform", "airflow"
    ],
    "Analytics / BI Tools": [
        "excel", "power bi", "tableau", "looker", "google analytics",
        "spss", "sas"
    ],
    "Soft Skills": [
        "communication", "teamwork", "leadership", "problem solving",
        "critical thinking", "project management", "collaboration",
        "adaptability", "time management", "agile", "scrum"
    ],
}

# Flattened, lower-cased set of all known skills for fast lookup
SKILL_DB = sorted({
    skill.lower()
    for skills in SKILL_TAXONOMY.values()
    for skill in skills
})

# Pre-build regex patterns once (word-boundary safe, handles multi-word and
# symbol-containing skills like "c++" / "c#" / "ci/cd")
def _build_pattern(skill: str) -> re.Pattern:
    escaped = re.escape(skill)
    # Use lookaround word boundaries that tolerate leading/trailing punctuation
    pattern = r"(?<![a-zA-Z0-9]){}(?![a-zA-Z0-9])".format(escaped)
    return re.compile(pattern, re.IGNORECASE)


_SKILL_PATTERNS = {skill: _build_pattern(skill) for skill in SKILL_DB}


def extract_skills(text: str) -> set:
    """
    Scan `text` (raw, un-preprocessed is fine) and return the set of known
    skills found in it, matched against SKILL_DB.
    """
    if not text:
        return set()

    found = set()
    for skill, pattern in _SKILL_PATTERNS.items():
        if pattern.search(text):
            found.add(skill)
    return found


def skill_category(skill: str) -> str:
    """Return the taxonomy category a given skill belongs to."""
    skill = skill.lower()
    for category, skills in SKILL_TAXONOMY.items():
        if skill in [s.lower() for s in skills]:
            return category
    return "Other"
