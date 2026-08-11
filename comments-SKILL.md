---
name: concise-comments
description: Generate Python code with minimal, intentional comments only. Use this skill whenever you're writing or generating Python files, modules, or scripts. The skill enforces a "comments should be rare" philosophy — code should be self-documenting through clear naming and structure. Only add comments for non-obvious logic, design decisions, warnings, or external requirements. Avoid redundant docstrings that repeat function signatures, unnecessary section dividers, or explaining what obviously clear code already shows. Trigger this skill for any Python code generation, refactoring, or review task in production codebases, ML pipelines, data tools, or application code where maintainability depends on signal-to-noise ratio in comments.
compatibility: Python 3.7+
---

# Concise Comments for Code

## Philosophy

Comments should earn their place in code. Excessive comments create noise that obscures intent and makes diffs harder to review. The right approach:

- **Self-documenting code first**: Clear variable names, function names, and logical flow make most code self-explanatory.
- **Comments for the "why"**: Explain non-obvious decisions, performance trade-offs, or external constraints.
- **Avoid the obvious**: Don't comment code that's already clear from reading it.
- **Minimal docstrings**: Use only where external tools (IDE hints, documentation generators) require them.

## When to Comment

### ✅ Good comments (keep these)

```python
# Map attack types to severity levels (required for NIDS compliance)
severity_map = {"DoS": 3, "Probe": 1, "R2L": 2}

# O(n log n) sort is acceptable here; we need stable ordering for joins
data = sorted(data, key=lambda x: x["timestamp"])

# Avoid SimpleImputer(strategy='mean') for security metrics — 
# zero-imputation biases false positives. Use forward-fill instead.
```

### ❌ Redundant comments (remove these)

```python
# Don't do this:
# Load the dataset
df = load_dataset(path)

# Loop through rows
for row in df.iterrows():
    process(row)

# Check if result is None
if result is None:
    continue

# Get the label column name
label_col = detect_label_column(df, user_col)
```

## Specific Rules for This Skill

1. **No section dividers** unless absolutely necessary for visual separation of major phases (e.g., "Training" vs "Inference"). Use blank lines instead.

2. **No docstring noise**:
   - Only include module docstrings for **public APIs**.
   - Skip docstrings for private methods (prefixed with `_`).
   - For public methods, include **parameter types + return type** if not obvious from code; examples only if the behavior is non-standard.

3. **No inline comments for iteration/conditionals**: 
   - If you need to explain why a loop or if statement exists, that's a sign the code should be refactored into a named function.

4. **WARNING and TODO only for critical issues**:
   - Use `# WARNING:` if code has dangerous behavior or side effects that aren't obvious.
   - Use `# TODO:` sparingly, only for high-priority blockers or known limitations.

5. **Performance or algorithm notes only**:
   - Comment complexity (`O(n²)` behavior) only if it's **surprising or problematic** given context.
   - Explain optimizations that trade off readability for speed.

## Example: Before & After

### Before (Too Many Comments)

```python
# Encapsulates all data cleaning and label encoding for the NIDS pipeline.
class DataPreprocessor:
    """
    Encapsulates all data cleaning and label encoding for the NIDS pipeline.

    Responsibilities
    ----------------
    * Optional stratified row sampling (``sample_frac``).
    * Replace ``±Inf`` values with ``NaN`` (handled downstream by the
      ``SimpleImputer`` in each model pipeline).
    * Optional binary label collapsing: anything that is not "benign" becomes
      "Attack".
    * Fit and apply a ``sklearn.preprocessing.LabelEncoder`` so that class
      strings are mapped to contiguous integers.

    Attributes
    ----------
    encoder : LabelEncoder
        Fitted after ``fit_transform`` is called.  Must be serialised with the
        model bundle so that ``evaluate.py`` and ``predict.py`` can decode
        integer predictions back to human-readable labels.
    """

    def __init__(self) -> None:
        # Initialize the encoder
        self.encoder: LabelEncoder = LabelEncoder()

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame, label_col: str, ...) -> Tuple[...]:
        """
        Prepare the dataset for model training.
        [full docstring repeated...]
        """
        # Validate sample_frac parameter
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1].")

        # Sample rows if sample_frac < 1.0
        if sample_frac < 1.0:
            df = (
                df.sample(frac=sample_frac, random_state=random_state)
                .reset_index(drop=True)
            )

        # Separate features from label
        y_raw: pd.Series = df[label_col].astype(str)
        X: pd.DataFrame = df.drop(columns=[label_col]).copy()
        # Replace infinite values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # Optional binary collapsing
        if binary:
            y_raw = pd.Series(
                np.where(y_raw.str.lower() == "benign", "Benign", "Attack")
            )

        # Fit and transform labels to integers
        y: np.ndarray = self.encoder.fit_transform(y_raw)

        # Prepare metadata dictionary
        metadata: Dict[str, Any] = {...}
        return X, y, metadata
```

### After (Minimal & Intentional)

```python
class DataPreprocessor:
    """Data cleaning and label encoding for NIDS pipeline."""

    def __init__(self) -> None:
        self.encoder: LabelEncoder = LabelEncoder()

    def fit_transform(
        self,
        df: pd.DataFrame,
        label_col: str,
        binary: bool = False,
        sample_frac: float = 1.0,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """Prepare dataset for training. Returns (X, y, metadata)."""
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1].")

        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

        y_raw: pd.Series = df[label_col].astype(str)
        X: pd.DataFrame = df.drop(columns=[label_col]).copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        if binary:
            # Map all non-Benign labels to Attack (binary classification mode)
            y_raw = pd.Series(
                np.where(y_raw.str.lower() == "benign", "Benign", "Attack")
            )

        y: np.ndarray = self.encoder.fit_transform(y_raw)

        metadata: Dict[str, Any] = {
            "label_column": label_col,
            "binary_mode": binary,
            "class_names": self.encoder.classes_.tolist(),
            "n_samples": int(len(df)),
            "n_features": int(X.shape[1]),
        }
        return X, y, metadata
```

**Changes made:**
- Removed redundant class docstring (function name is clear).
- Removed section dividers (`# -----`).
- Removed "Responsibilities" and "Attributes" docstrings (code is self-explanatory).
- Removed inline comments explaining obvious steps (loop, conditional, assignment).
- Kept **one** comment explaining the non-obvious binary collapsing logic.

## How to Apply This Skill

When writing or refactoring Python code:

1. **Generate code with clear naming first** — let the code speak.
2. **Add comments only where logic isn't obvious** from the code itself.
3. **For public APIs**, include minimal docstrings that answer "what does this return?" not "what does each line do?"
4. **Review and prune** — if you wrote a comment that just restates the code, delete it.

## Testing Your Work

Ask yourself for each comment:
- "Would a developer understand this line without the comment?" → Remove it.
- "Is this comment explaining *why* or just *what*?" → If *what*, delete it.
- "Is this a design decision, warning, or external constraint?" → Keep it.

