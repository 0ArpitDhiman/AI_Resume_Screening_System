"""
ranking_engine.py
------------------
Core NLP/ML engine of the AI Resume Screening System.

Approach
--------
1. Preprocess the job description (JD) and every resume.
2. Vectorize all documents together with TF-IDF (so the vocabulary and
   IDF weights are shared/comparable across JD + resumes).
3. Score each resume's semantic similarity to the JD using cosine
   similarity between TF-IDF vectors.
4. Independently extract known skills (see core/skills_database.py) from
   both the JD and each resume, and compute a skill-overlap ratio.
5. Combine both signals into a single weighted Final Score, which is what
   resumes are ranked on. Weighting is configurable (defaults: 60% semantic
   similarity, 40% skill match) because pure keyword overlap can miss
   synonyms/context, and pure TF-IDF similarity can miss the fact that a
   resume is simply missing a hard, named requirement (e.g. "AWS").
"""

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.preprocessing import preprocess
from core.skills_database import extract_skills


@dataclass
class ResumeResult:
    candidate: str
    similarity_score: float          # TF-IDF cosine similarity (0-1)
    skill_match_score: float         # matched / required skills (0-1)
    final_score: float               # weighted combination (0-1)
    matched_skills: set = field(default_factory=set)
    missing_skills: set = field(default_factory=set)
    extra_skills: set = field(default_factory=set)
    raw_text_preview: str = ""

    @property
    def final_score_pct(self):
        return round(self.final_score * 100, 2)

    @property
    def similarity_score_pct(self):
        return round(self.similarity_score * 100, 2)

    @property
    def skill_match_score_pct(self):
        return round(self.skill_match_score * 100, 2)


def rank_resumes(
    job_description: str,
    resumes: dict,
    similarity_weight: float = 0.6,
    skill_weight: float = 0.4,
):
    """
    Parameters
    ----------
    job_description : str
        Raw job description text.
    resumes : dict[str, str]
        Mapping of {candidate_name: raw_resume_text}.
    similarity_weight, skill_weight : float
        Must sum to 1.0. Controls how much each signal contributes to the
        final ranking score.

    Returns
    -------
    list[ResumeResult], sorted by final_score descending.
    """
    assert abs((similarity_weight + skill_weight) - 1.0) < 1e-6, \
        "similarity_weight + skill_weight must equal 1.0"

    if not resumes:
        return []

    # ---- 1. Preprocess ----
    jd_clean = preprocess(job_description)
    candidate_names = list(resumes.keys())
    resumes_clean = [preprocess(resumes[name]) for name in candidate_names]

    # ---- 2. TF-IDF vectorization (JD + all resumes share one vocabulary) ----
    corpus = [jd_clean] + resumes_clean
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    # ---- 3. Cosine similarity ----
    similarities = cosine_similarity(jd_vector, resume_vectors).flatten()

    # ---- 4. Skill extraction & overlap ----
    jd_skills = extract_skills(job_description)

    results = []
    for i, name in enumerate(candidate_names):
        resume_text = resumes[name]
        resume_skills = extract_skills(resume_text)

        matched = jd_skills & resume_skills
        missing = jd_skills - resume_skills
        extra = resume_skills - jd_skills

        skill_match_score = (len(matched) / len(jd_skills)) if jd_skills else 0.0
        similarity_score = float(similarities[i])

        final_score = (
            similarity_weight * similarity_score
            + skill_weight * skill_match_score
        )

        results.append(
            ResumeResult(
                candidate=name,
                similarity_score=similarity_score,
                skill_match_score=skill_match_score,
                final_score=final_score,
                matched_skills=matched,
                missing_skills=missing,
                extra_skills=extra,
                raw_text_preview=resume_text[:300].replace("\n", " ").strip(),
            )
        )

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results
