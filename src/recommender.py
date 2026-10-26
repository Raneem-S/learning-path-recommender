"""
Content-based course recommender: ranks a course catalog against a
free-text learner profile, with optional exclusion of completed courses
optional level-awareness, and one optional adjacent exploration pick.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_cleaning import clean_courses


LEVEL_ORDER = {
    "beginner level": 0,
    "intermediate level": 1,
    "advanced level": 2,
}


def build_text_columns(df):
    """Create separate text fields for skills and course content."""
    result = df.copy()
    result["skills_text"] = result["Skills"].apply(
        lambda skills: " ".join(skills)
    )
    result["content_text"] = (
        result["title"].fillna("").str.strip()
        + " "
        + result["Description"].fillna("").str.strip()
    ).str.strip()
    return result


def normalize_title(title):
    if not isinstance(title, str):
        return ""
    return " ".join(title.split()).casefold()


def normalize_org(org):
    if not isinstance(org, str):
        return ""
    return " ".join(org.split()).casefold()


def normalize_entry(entry):
    """Normalize a title string or (title, organization) pair."""
    if isinstance(entry, str):
        title = normalize_title(entry)
        if not title:
            raise ValueError("Course title cannot be empty.")
        return (title, None)
    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
        title, org = entry
        title = normalize_title(title)
        org = normalize_org(org)
        if not title or not org:
            raise ValueError("Both title and organization must be valid strings.")
        return (title, org)
    else:
        raise TypeError(
            "Each exclusion must be a title string or a (title, organization) pair."
        )


def filter_excluded(catalog_df, final_scores, exclude_titles=None):
    """Remove completed courses, preserving positional score alignment."""
    if exclude_titles is None:
        exclude_titles = []
    exclude_titles = list(exclude_titles)
    normalized_entries = [normalize_entry(entry) for entry in exclude_titles]

    title_only = {
        title for title, org in normalized_entries if org is None
    }
    title_and_org = {
        (title, org) for title, org in normalized_entries if org is not None
    }

    catalog_titles = catalog_df["title"].map(normalize_title)
    catalog_orgs = catalog_df["Organization"].map(normalize_org)
    catalog_pairs = list(zip(catalog_titles, catalog_orgs))

    title_mask = catalog_titles.isin(title_only).to_numpy()
    pair_mask = np.array(
        [pair in title_and_org for pair in catalog_pairs], dtype=bool
    )
    exclude_mask = title_mask | pair_mask

    available_titles = set(catalog_titles)
    available_pairs = set(catalog_pairs)
    unmatched_exclusions = []
    for original, (title, org) in zip(exclude_titles, normalized_entries):
        if org is None:
            matched = title in available_titles
        else:
            matched = (title, org) in available_pairs
        if not matched:
            unmatched_exclusions.append(original)

    scores = np.asarray(final_scores)
    if scores.ndim != 1 or len(scores) != len(catalog_df):
        raise ValueError("final_scores must contain one score per catalog row.")

    keep_mask = ~exclude_mask
    filtered_catalog = catalog_df.iloc[
        np.flatnonzero(keep_mask)
    ].reset_index(drop=True)
    filtered_scores = scores[keep_mask]

    return {
        "catalog": filtered_catalog,
        "scores": filtered_scores,
        "unmatched_exclusions": unmatched_exclusions,
    }


def level_to_ordinal(level):
    """Convert known course level into 0/1/2, else None."""
    if not isinstance(level, str):
        return None
    normalized_level = " ".join(level.split()).casefold()
    return LEVEL_ORDER.get(normalized_level)


def compute_level_gaps(catalog_df, learner_level):
    """Return course minus learner level, NaN for unknown course level."""
    learner_ordinal = level_to_ordinal(learner_level)
    if learner_ordinal is None:
        raise ValueError(
            f"Unrecognized learner level: {learner_level!r}. "
            f"Expected one of: {list(LEVEL_ORDER)}"
        )
    course_ordinals = catalog_df["Level"].map(
        level_to_ordinal
    ).to_numpy(dtype=float)
    return course_ordinals - learner_ordinal


def compute_level_multipliers(
    level_gaps, stretch_bonus=0.15, retreat_penalty=0.15
):
    """Boost stretch, penalize retreat; unknown/same level stays neutral."""
    gaps = np.asarray(level_gaps, dtype=float)
    if stretch_bonus < 0 or retreat_penalty < 0:
        raise ValueError("Bonus and penalty must be non-negative.")
    multipliers = np.ones_like(gaps, dtype=float)
    above_mask = gaps > 0
    multipliers[above_mask] = (
        1.0 + stretch_bonus * gaps[above_mask]
    )
    below_mask = gaps < 0
    multipliers[below_mask] = np.maximum(
        0.1, 1.0 + retreat_penalty * gaps[below_mask]
    )
    return multipliers


def build_recommender(raw_df, skills_weight=0.6, content_weight=0.4):
    """Fit both TF-IDF vectorizers once and bundle the catalog/model."""
    if skills_weight < 0 or content_weight < 0:
        raise ValueError("Weights must be non-negative.")
    if not np.isclose(skills_weight + content_weight, 1.0):
        raise ValueError("skills_weight and content_weight must sum to 1.")

    catalog = build_text_columns(clean_courses(raw_df))
    if catalog.empty:
        raise ValueError("No courses remain after cleaning.")

    skills_vectorizer = TfidfVectorizer(stop_words="english")
    content_vectorizer = TfidfVectorizer(stop_words="english")
    skills_matrix = skills_vectorizer.fit_transform(catalog["skills_text"])
    content_matrix = content_vectorizer.fit_transform(catalog["content_text"])

    return {
        "catalog": catalog,
        "skills_vectorizer": skills_vectorizer,
        "content_vectorizer": content_vectorizer,
        "skills_matrix": skills_matrix,
        "content_matrix": content_matrix,
        "skills_weight": skills_weight,
        "content_weight": content_weight,
    }


def pick_adjacent_candidate(
    live_catalog,
    skills_sim,
    content_sim,
    exclude_indices=(),
    skills_percentile=75,
    content_percentile=60,
    level_multipliers=None,
):
    """Choose one adjacent course by its level-adjusted skills/content gap.

    Inputs are POSITIONAL arrays aligned with live_catalog, after filtering
    completed courses. Return an integer positional index, or None.
    """
    skills_sim = np.asarray(skills_sim, dtype=float)
    content_sim = np.asarray(content_sim, dtype=float)

    if (
        skills_sim.ndim != 1
        or content_sim.ndim != 1
        or len(skills_sim) != len(live_catalog)
        or len(content_sim) != len(live_catalog)
    ):
        raise ValueError("Similarity arrays must align with live_catalog.")
    if not 0 <= skills_percentile <= 100:
        raise ValueError("skills_percentile must be between 0 and 100.")
    if not 0 <= content_percentile <= 100:
        raise ValueError("content_percentile must be between 0 and 100.")

    if level_multipliers is None:
        multipliers = np.ones(len(live_catalog), dtype=float)
    else:
        multipliers = np.asarray(level_multipliers, dtype=float)
        if multipliers.shape != skills_sim.shape:
            raise ValueError("Level multipliers must align with live_catalog.")
        if not np.all(np.isfinite(multipliers)) or np.any(multipliers <= 0):
            raise ValueError("Level multipliers must be finite and positive.")

    valid = np.isfinite(skills_sim) & np.isfinite(content_sim)
    if not valid.any():
        return None

    skills_threshold = np.percentile(skills_sim[valid], skills_percentile)
    content_threshold = np.percentile(content_sim[valid], content_percentile)

    eligible = (
        valid
        & (skills_sim >= skills_threshold)
        & (content_sim >= content_threshold)
        & (skills_sim > content_sim)
        & (skills_sim > 0)
        & (content_sim > 0)
    )

    for index in exclude_indices:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(eligible):
            raise ValueError("exclude_indices must contain valid positional indices.")
        eligible[index] = False

    candidate_indices = np.flatnonzero(eligible)
    if candidate_indices.size == 0:
        return None

    exploration_scores = (skills_sim - content_sim) * multipliers
    best_index = candidate_indices[np.argmax(exploration_scores[candidate_indices])]
    return int(best_index)


def recommend(
    recommender,
    profile_text,
    learner_level=None,
    exclude_titles=None,
    n=5,
    stretch_bonus=0.15,
    retreat_penalty=0.15,
    include_adjacent=False,
):
    """Return top-n eligible courses, optionally reserving one adjacent slot.

    Standard recommendations are ranked by level-adjusted combined similarity.
    The optional adjacent pick is selected by level-adjusted skills/content gap.
    If no adjacent candidate qualifies, fall back to n standard picks.
    For n=1, return the highest-ranked standard course even when
    include_adjacent=True. Output columns have a consistent schema.
    """
    if not isinstance(profile_text, str) or not profile_text.strip():
        raise ValueError("profile_text must be a non-empty string.")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")

    catalog = recommender["catalog"]
    skills_vec = recommender["skills_vectorizer"].transform([profile_text])
    content_vec = recommender["content_vectorizer"].transform([profile_text])

    if skills_vec.nnz == 0 and content_vec.nnz == 0:
        raise ValueError("Profile has no terms recognized by either vectorizer.")

    skills_sim = cosine_similarity(skills_vec, recommender["skills_matrix"]).ravel()
    content_sim = cosine_similarity(content_vec, recommender["content_matrix"]).ravel()
    combined_scores = (
        recommender["skills_weight"] * skills_sim
        + recommender["content_weight"] * content_sim
    )

    # Filter every similarity signal using identical title/organization exclusions.
    live_catalog = catalog
    base_scores = combined_scores
    live_skills_sim = skills_sim
    live_content_sim = content_sim
    unmatched_exclusions = []

    if exclude_titles is not None:
        filtered = filter_excluded(catalog, combined_scores, exclude_titles)
        live_catalog = filtered["catalog"]
        base_scores = filtered["scores"]
        unmatched_exclusions = filtered["unmatched_exclusions"]
        live_skills_sim = filter_excluded(catalog, skills_sim, exclude_titles)["scores"]
        live_content_sim = filter_excluded(catalog, content_sim, exclude_titles)["scores"]

    # Check positional alignment without changing either similarity metric.
    if not (
        len(live_catalog)
        == len(base_scores)
        == len(live_skills_sim)
        == len(live_content_sim)
    ):
        raise ValueError("Filtered catalog and similarity arrays are misaligned.")

    reconstructed_scores = (
        recommender["skills_weight"] * live_skills_sim
        + recommender["content_weight"] * live_content_sim
    )
    if not np.allclose(reconstructed_scores, base_scores):
        raise ValueError("Filtered similarity scores do not reconstruct base scores.")

    level_multipliers = np.ones(len(live_catalog), dtype=float)
    if learner_level is not None:
        gaps = compute_level_gaps(live_catalog, learner_level)
        level_multipliers = compute_level_multipliers(
            gaps, stretch_bonus=stretch_bonus, retreat_penalty=retreat_penalty
        )
    adjusted_scores = base_scores * level_multipliers

    ranked_indices = np.argsort(adjusted_scores)[::-1]
    adjacent_index = None

    if include_adjacent and n >= 2 and len(ranked_indices) > 0:
        main_indices = ranked_indices[: n - 1].tolist()
        adjacent_index = pick_adjacent_candidate(
            live_catalog=live_catalog,
            skills_sim=live_skills_sim,
            content_sim=live_content_sim,
            exclude_indices=set(main_indices),
            skills_percentile=75,
            content_percentile=60,
            level_multipliers=level_multipliers,
        )
        if adjacent_index is not None:
            top_indices = main_indices + [adjacent_index]
        else:
            top_indices = ranked_indices[:n].tolist()
    else:
        top_indices = ranked_indices[:n].tolist()

    results = live_catalog.iloc[top_indices][
        ["title", "Organization", "Level"]
    ].copy().reset_index(drop=True)
    results["similarity_score"] = base_scores[top_indices]
    results["level_multiplier"] = level_multipliers[top_indices]
    results["score"] = adjusted_scores[top_indices]
    results["is_adjacent_pick"] = [
        adjacent_index is not None and i == adjacent_index for i in top_indices
    ]
    results["exploration_score"] = np.nan
    if adjacent_index is not None:
        results.loc[results["is_adjacent_pick"], "exploration_score"] = (
            (live_skills_sim[adjacent_index] - live_content_sim[adjacent_index])
            * level_multipliers[adjacent_index]
        )

    results.attrs["unmatched_exclusions"] = unmatched_exclusions
    return results
