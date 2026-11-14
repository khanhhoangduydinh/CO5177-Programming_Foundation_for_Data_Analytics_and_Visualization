from pathlib import Path
import pandas as pd
from pyvi import ViTokenizer

ROOT = Path(__file__).resolve().parent.parent

def count_words(text):
    """Count segmented tokens containing letters or digits; ignore punctuation and emoji."""
    tokens = ViTokenizer.tokenize(str(text)).split()
    return sum(any(char.isalnum() for char in token) for token in tokens)


def main():
    distributions = []
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(ROOT / "data" / f"victsd_{split}.csv")
        if frame["Comment"].isna().any():
            raise ValueError(f"Missing comments in {split}; inspect before counting")
        counts = frame["Comment"].map(count_words).value_counts().sort_index()
        distributions.append(counts.rename(split))
        print(f"{split}: {len(frame)} comments processed", flush=True)
    result = pd.concat(distributions, axis=1).fillna(0).astype(int)
    result = result.reindex(range(int(result.index.max()) + 1), fill_value=0)
    result.index.name = "word_count"
    result["all"] = result.sum(axis=1)
    path = ROOT / "reports" / "victsd_word_count_distribution.csv"
    result.to_csv(path)
    print(f"Saved {path}; 10 words: {result.loc[10, 'all']}; 20 words: {result.loc[20, 'all']}")

if __name__ == "__main__":
    main()
