# scripts/01_clean_preprocess.py
import argparse
import os
import re

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\S+")
MULTI_PUNCT_RE = re.compile(r"([!?。，、；：,.])\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")

# Simple spam heuristics (lightweight)
SPAM_PATTERNS = [
    re.compile(r"免费|領取|点击|點擊|加微信|加vx|whatsapp|telegram", re.IGNORECASE),
    re.compile(r"赚钱|賺錢|投资|投資|博彩|賭博", re.IGNORECASE),
]


def looks_like_spam(text: str) -> bool:
    t = text.strip()
    if len(t) < 2:
        return True
    hit = 0
    for p in SPAM_PATTERNS:
        if p.search(t):
            hit += 1
    return hit >= 2


def normalize_text(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    collapse_repeated_punct: bool = True,
    to_simplified: bool = False,
) -> str:
    t = str(text)

    if remove_urls:
        t = URL_RE.sub(" ", t)

    if remove_mentions:
        t = MENTION_RE.sub(" ", t)

    if collapse_repeated_punct:
        t = MULTI_PUNCT_RE.sub(r"\1", t)

    # Optional: Traditional -> Simplified
    # We keep this optional because it requires an extra library.
    if to_simplified:
        try:
            from opencc import OpenCC  # type: ignore

            cc = OpenCC("t2s")
            t = cc.convert(t)
        except Exception:
            # If opencc is not installed, keep original text.
            pass

    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess Chinese-language YouTube comments conservatively.")
    parser.add_argument("--input", required=True, help="Input CSV with a 'text' column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--text_col", default="text", help="Text column name (default: text)")
    parser.add_argument("--to_simplified", action="store_true", help="Convert Traditional to Simplified (requires opencc)")
    parser.add_argument("--drop_spam", action="store_true", help="Drop spam-like comments using simple heuristics")
    parser.add_argument("--min_len", type=int, default=2, help="Minimum length of cleaned text to keep")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available columns: {list(df.columns)}")

    cleaned = []
    kept_rows = 0

    for _, row in df.iterrows():
        raw = row[args.text_col]
        t = normalize_text(raw, to_simplified=args.to_simplified)

        if len(t) < args.min_len:
            continue

        if args.drop_spam and looks_like_spam(t):
            continue

        cleaned.append(t)
        kept_rows += 1

    out = pd.DataFrame({"text": cleaned})
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved cleaned comments: {len(out)} to {args.output}")


if __name__ == "__main__":
    main()
