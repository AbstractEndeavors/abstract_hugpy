## Summary Comparison and Judging Module (/src/abstract_summary_judge.py)
"""Compare new and existing text or summaries using DeepZero’s scoring ability."""

from ..imports import *

class SummaryJudge(metaclass=SingletonMeta):
    """
    Uses DeepZero to compare text quality or relevance without generating new text.
    Returns normalized scores (0–1) and chooses the better version.
    """

    def __init__(self):
        self.pattern = re.compile(r"(\d+(?:\.\d+)?)")  # to extract numeric score

    # -------------------------------------------------------------
    def _ask_model(self, text: str, summary: str) -> float:
        """Ask DeepZero for a numeric score (0–1)."""
        prompt = (
            f"Rate the following summary from 0 to 100 for clarity, accuracy, "
            f"and completeness relative to the source text.\n\n"
            f"--- TEXT ---\n{text[:1200]}\n\n"
            f"--- SUMMARY ---\n{summary[:800]}\n\nScore:"
        )
        try:
            result = deep_zero_generate(prompt, max_new_tokens=8, temperature=0.0)
            match = self.pattern.search(result)
            if match:
                score = float(match.group(1))
                return max(0.0, min(score / 100.0, 1.0))
        except Exception as e:
            print(f"[SummaryJudge] Error scoring summary: {e}")
        return 0.5  # neutral fallback

    # -------------------------------------------------------------
    def compare(self, text: str, new_summary: str, old_summary: Optional[str] = None) -> Tuple[str, float, float]:
        """
        Compare summaries and return (best_summary, best_score, other_score).
        If no old summary, returns (new_summary, new_score, 0.0).
        """
        new_score = self._ask_model(text, new_summary)
        if not old_summary:
            return new_summary, new_score, 0.0

        old_score = self._ask_model(text, old_summary)
        if new_score >= old_score:
            return new_summary, new_score, old_score
        else:
            return old_summary, old_score, new_score

    # -------------------------------------------------------------
    def compare_files(self, text_file: str, new_summary: str, existing_json: Optional[str] = None) -> Tuple[str, float, float]:
        """
        Compare summaries using on-disk .txt or .json files.
        """
        text = open(text_file, encoding="utf-8").read()
        old_summary = None
        if existing_json and os.path.exists(existing_json):
            try:
                with open(existing_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    old_summary = data.get("summary") or data.get("text")
            except Exception:
                pass

        return self.compare(text, new_summary, old_summary)

# -------------------------------------------------------------
def compare_and_keep(text: str, new_summary: str, old_summary: Optional[str] = None):
    """
    Convenience wrapper returning the better summary and metadata.
    """
    judge = SummaryJudge()
    chosen, score_a, score_b = judge.compare(text, new_summary, old_summary)
    replaced = (chosen == new_summary) and old_summary is not None
    return {
        "chosen": chosen,
        "new_score": score_a if replaced else score_b,
        "old_score": score_b if replaced else score_a,
        "replaced": replaced
    }
