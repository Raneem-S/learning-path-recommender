# paste your is_missing_content, parse_skills, and clean_courses functions here
# (the ones you already wrote and tested)
import ast
import pandas as pd

def is_missing_content(value):
  """Treat NaN, blanks, and empty list-like fields as missing text."""
  if value is None:
    return True
  if isinstance(value, (list, tuple, set)):
    return not value or all(is_missing_content(item) for item in value)
  if isinstance(value, str):
    text = value.strip()
    if text.casefold() in {"", "nan", "none", "null", "[]"}:
      return True
    if text.startswith("[") and text.endswith("]"):
      try:
        parsed = ast.literal_eval(text)
      except (SyntaxError, ValueError):
        return False
      if isinstance(parsed, (list, tuple)):
        return not parsed or all(is_missing_content(item) for item in parsed)
    return False
  try:
    return bool(pd.isna(value))
  except (TypeError, ValueError):
    return False
def parse_skills(value):
  """Turn a serialized skill list into a deduplicated list, in original order.

  Deduplication ignores case and extra whitespace but preserves the first spelling.
  Example: "['Python', 'python', 'SQL']" -> "['Python', 'SQL']"
  """
  if is_missing_content(value):
    return []
  if isinstance(value, str):
    raw = value.strip()
    if raw.startswith("["):
      try:
        items = ast.literal_eval(raw)
      except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Malformed serialized Skills list: {raw!r}") from exc
      if not isinstance(items, (list, tuple)):
        raise ValueError(f"Expected a Skills list, got {type(items).__name__}")
    else:
      items = [raw]
  elif isinstance(value, (list, tuple)):
    items = value
  else:
    raise TypeError(f"Unexpected Skills type: {type(value).__name__}")
  unique_skills = []
  seen = set()
  for item in items:
    if is_missing_content(item):
      continue
    skill = " ".join(str(item).split())
    key = skill.casefold()
    if key not in seen:
      unique_skills.append(skill)
      seen.add(key)
  return unique_skills

def clean_courses(raw_df):
  """Return an independent, cleaned copy without discarding legitimate courses."""
  required = {"title", "Organization", "Skills", "Description", "Modules/Courses", "Level"}
  missing_columns = required.difference(raw_df.columns)  # fixed: .difference(), not .differences()
  if missing_columns:
    raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")
  cleaned = raw_df.copy(deep=True)

  missing_description = cleaned["Description"].map(is_missing_content)
  missing_modules = cleaned["Modules/Courses"].map(is_missing_content)

  # fixed: actually drop the ~10 broken rows (missing both Description and Modules/Courses),
  # instead of just blanking their fields and letting them survive
  drop_mask = missing_description & missing_modules
  cleaned = cleaned.loc[~drop_mask].reset_index(drop=True)

  cleaned["Skills"] = cleaned["Skills"].map(parse_skills)

  for column in ["Description", "Modules/Courses"]:
    cleaned[column] = cleaned[column].map(
        lambda value: "" if is_missing_content(value) else str(value).strip()
    )

  cleaned["Level"] = cleaned["Level"].map(
      lambda value: "Unspecified" if is_missing_content(value) else str(value).strip()
  )
  cleaned = cleaned.drop(columns=["Satisfaction Rate", "Unnamed: 0"], errors="ignore")

  return cleaned
  
