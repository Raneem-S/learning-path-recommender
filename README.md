# Learning Path Recommender

**A content-based course recommendation engine that balances relevance, learner readiness, and exploration.**

Most course recommenders focus on what looks similar to a learner's background. This project takes a more practical approach: it ranks Coursera courses using a free-text learner profile, removes courses the learner has already completed, adjusts for experience level, and can reserve one recommendation for a related-but-different direction.

> **Current scope:** This repository implements **course recommendation and ranking**, not yet prerequisite-aware or automatically ordered learning paths. Sequencing courses into a learning path is a planned next step.

## At a glance

| | |
|---|---|
| **Data** | [Coursera Course Dataset](https://huggingface.co/datasets/azrai99/coursera-course-dataset) |
| **Catalog** | 6,645 source records; **6,635 usable courses** after cleaning in the tested dataset snapshot |
| **Approach** | Two separate TF-IDF representations + cosine similarity |
| **Main ranking** | 60% skills similarity + 40% title/description similarity, optionally adjusted by course level |
| **Personalization** | Free-text profile, completed-course exclusions, learner level |
| **Exploration** | Optional adjacent recommendation, separate from the highest-ranked core matches |
| **Stack** | Python, pandas, NumPy, scikit-learn, Hugging Face Datasets |

## Why I built it

A learner who has completed several machine-learning and data-analytics courses does not need another list of introductory content. But simply searching for matching course titles misses related opportunities in new tools and domains.

I wanted to answer three questions:

1. **What courses match this learner's existing technical background?**
2. **Which suggestions are duplicates of work they have already completed or a poor difficulty fit?**
3. **Can the system surface one useful adjacent direction without drifting into unrelated topics?**

The model accepts a narrative learner profile rather than requiring an exact course title or predefined job role. That makes it possible to represent skills accumulated across courses and providers.

## How it works

```text
Free-text learner profile
         |
         +--------------------------+
         |                          |
         v                          v
Skills TF-IDF vectorizer    Content TF-IDF vectorizer
         |                   (title + description)
         v                          v
Cosine skills similarity    Cosine content similarity
         |                          |
         +------------+-------------+
                      |
             Weighted similarity
            0.60 skills + 0.40 content
                      |
          Exclude completed courses
          (title, or title + provider)
                      |
        Optional level-aware reranking
                      |
             Core recommendations
                      |
       Optional adjacent exploration
            (reserve one slot)
                      |
            Ranked course results
```

### 1. Build two course representations

- **Skills signal:** TF-IDF fitted on the course's skill tags.
- **Content signal:** A separate TF-IDF model fitted on the course title and description.

Each representation is independently compared with the free-text learner profile using cosine similarity:

```text
base_score = 0.60 × skills_similarity + 0.40 × content_similarity
```

These are **heuristic weights**, not coefficients learned from labeled relevance judgments.

### 2. Remove completed learning

Exclusions support either a course title or a **(title, organization)** pair. Matching is case-insensitive and ignores extra whitespace. The catalog and all similarity arrays are filtered identically so their positions remain aligned. Unmatched exclusions are reported in the result metadata.

### 3. Account for difficulty

When the learner provides a level, the combined score is multiplied by a level-based factor:

| Relative course level | Example multiplier for an intermediate learner |
|---|---:|
| Beginner (below learner level) | 0.85 |
| Intermediate (same level) | 1.00 |
| Advanced (above learner level) | 1.15 |
| Unspecified | 1.00 |

```text
final_score = base_score × level_multiplier
```

The adjustment encourages progression but does **not** ban lower-level courses: learning a genuinely new tool may still justify a beginner course.

### 4. Reserve an optional exploration slot

With `include_adjacent=True` and `n >= 2`, the engine first selects the top `n - 1` core matches. It then looks for one **adjacent-but-different** course that:

- Has skills similarity at or above the **75th percentile**.
- Has content similarity at or above the **60th percentile**.
- Has skills similarity greater than content similarity.
- Is not already included in the core recommendations.

Eligible candidates are ranked using:

```text
exploration_score = (skills_similarity - content_similarity) × level_multiplier
```

If no candidate qualifies, the engine fills all `n` positions with core recommendations. For `n = 1`, it always returns the best core match. The returned DataFrame marks the exploratory course with `is_adjacent_pick=True` and includes its separate `exploration_score`.

## Quickstart

Use Python 3.10+ from the repository root. The first run downloads the dataset from Hugging Face and fits the TF-IDF vectorizers.

```bash
git clone https://github.com/Raneem-S/learning-path-recommender.git
cd learning-path-recommender
python -m pip install numpy pandas scikit-learn datasets
```

```python
from datasets import load_dataset
from src.recommender import build_recommender, recommend

# Load the public Coursera catalog and fit the recommender.
raw_df = load_dataset("azrai99/coursera-course-dataset")["train"].to_pandas()
model = build_recommender(raw_df)

# Describe what the learner knows and wants to explore.
profile = (
    "Experienced with Python, machine learning, regression, classification, "
    "data analysis, SQL, visualization, deep learning, and generative AI. "
    "Interested in data engineering, deploying AI systems, and new analytics tools."
)

# Exclude already completed courses.
completed = [
    ("Exploratory Data Analysis for Machine Learning", "IBM"),
    ("Foundations: Data, Data, Everywhere", "Google"),
]

recommendations = recommend(
    model,
    profile,
    learner_level="Intermediate level",
    exclude_titles=completed,
    n=10,
    include_adjacent=True,
)

print(recommendations.to_string(index=False))
```

The output includes the title, `Organization`, `Level`, `similarity_score`, `level_multiplier`, `score`, `is_adjacent_pick`, and `exploration_score` (empty for core recommendations). Scores are **ranking signals, not calibrated probabilities**.

## Example from testing

In a notebook test using a profile built from completed ML, generative-AI engineering, and data-analytics training, the top results included:

| Result | Provider | Final score | Selection |
|---|---|---:|---|
| Generative AI for Data Scientists Specialization | IBM | 0.4125 | Core |
| Accounting Data Analytics Specialization | University of Illinois Urbana-Champaign | 0.4070 | Core |
| IBM Introduction to Machine Learning Specialization | IBM | 0.4037 | Core |
| Arranging and Visualizing Data in R | University of Michigan | 0.2410 | **Adjacent** |

The adjacent R course was added as the tenth recommendation, **not because it had the tenth-highest overall score**, but because it represented an exploratory application of familiar analytics skills in a different tool ecosystem. Its separate exploration score was **0.2928**.

*This table is an excerpt from one saved notebook run, not a benchmark or a guarantee that the quickstart profile will generate identical results.*

## A failure that improved the design

An early exploration rule simply chose courses with a large difference between skills similarity and content similarity. It surfaced **Introduction to Archaeology Specialization**, despite the learner focusing on ML and AI.

Investigation revealed why:

- The course had just one skill tag: **Data Analysis**.
- Its skills similarity was **0.3925**, but content similarity was only **0.0152**.
- TF-IDF normalization and a generic skill tag made the skills signal appear much stronger than its topical relevance.

I added a **60th-percentile minimum content-relevance floor** alongside the existing **75th-percentile skills threshold**. This excluded that unrelated example while preserving relevant, adjacent candidates such as R-based visualization, Snowflake, and production-AI topics.

**Lesson:** A high similarity score is not automatically a meaningful learning recommendation. Diagnostics on individual failures were more useful here than blindly tuning the blend weights.

## Design decisions and trade-offs

| Decision | Rationale / trade-off |
|---|---|
| Free-text learner profile | Captures cross-course experience, but depends on how completely the learner describes their skills |
| Separate skills and content TF-IDF models | Makes the two signals inspectable; their cosine values are not necessarily perfectly calibrated against each other |
| Completed-course exclusion by title and provider | Avoids recommending known completions without removing unrelated courses that share a title |
| Soft difficulty multipliers | Encourages stretch while retaining potentially useful beginner courses |
| One exploration slot | Preserves core recommendation quality while adding controlled diversity |
| Deterministic, interpretable retrieval | Straightforward to audit; lacks semantic understanding beyond the observed text |

## Validation

The Colab integration tests checked that:

- `n=1` returns the **same best core recommendation** whether exploration is enabled or not.
- `n=10` with exploration produces **nine core matches plus one adjacent pick** when a candidate qualifies.
- Excluded courses do not enter the ranking; filtered similarity arrays stay aligned.
- Only the exploration row has a non-null `exploration_score`.

Two further properties hold **by construction** rather than being separately tested: the exploration pick only ever fills the single reserved slot, so the ordering of the other `n - 1` recommendations is unaffected by whether exploration is enabled; and because each result row maps to a distinct positional index in the catalog, the same **(title, provider)** combination cannot appear twice in one result.

These are **functional and regression checks**, not a ground-truth ranking-quality evaluation.

## Current limitations

- **Not yet a path planner:** courses and specializations are ranked, but not sequenced by prerequisites or dependencies.
- **Prerequisites are not enforced:** some highly ranked courses belong to multi-course specializations.
- **Skill novelty is approximate:** shared tags and unfamiliar description words do not prove that a learner will gain a new skill.
- **No labeled relevance benchmark yet:** weights, percentile thresholds, and level multipliers were selected heuristically and tested through inspection, not optimized against human judgments.
- **Profile completeness matters:** skills omitted from the profile may be incorrectly treated as unfamiliar.
- **Dataset-specific:** output depends on the current Coursera catalog snapshot and metadata quality.

## Project modules

```text
src/
├── data_cleaning.py   # Validate/clean course records and normalize skill lists
└── recommender.py     # Fit TF-IDF models, exclude courses, rerank, explore
```

The research and validation were developed in Google Colab notebooks. The source modules can be imported independently of the notebook.

## Next steps

1. **Demo experience:** a streamlined Colab example or small Streamlit interface so visitors can try a learner profile without reading the source.
2. **Prerequisite-aware sequencing:** differentiate standalone courses from full specializations and construct an ordered learning path.
3. **Recommendation evaluation:** gather relevance judgments and evaluate ranking quality, novelty, and diversity.

---

*Built as an applied NLP / recommender-systems project. Current emphasis: transparent, testable course selection, not claims of production-ready educational guidance.*
