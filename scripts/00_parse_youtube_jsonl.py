# scripts/00_parse_youtube_jsonl.py
import argparse
import json
import os
from glob import glob

import pandas as pd


def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def parse_one_file(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Many youtube comment downloaders store text in different keys.
            # Try a few common patterns.
            text = (
                obj.get("text")
                or obj.get("content")
                or safe_get(obj, ["comment", "text"])
                or safe_get(obj, ["snippet", "textDisplay"])
                or ""
            )

            text = str(text).strip()
            if not text:
                continue

            # Keep minimal, non-identifying metadata
            video_id = obj.get("video_id") or obj.get("videoId") or ""
            comment_id = obj.get("comment_id") or obj.get("commentId") or obj.get("cid") or ""

            rows.append(
                {
                    "text": text,
                    "video_id": str(video_id),
                    "comment_id": str(comment_id),
                    "source_file": os.path.basename(path),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Parse YouTube comments .jsonl files into a single CSV with a 'text' column."
    )
    parser.add_argument("--input", required=True, help="Input .jsonl file OR a folder containing .jsonl files")
    parser.add_argument("--output", required=True, help="Output CSV path, e.g., data/interim/comments_parsed.csv")
    args = parser.parse_args()

    input_path = args.input
    files = []
    if os.path.isdir(input_path):
        files = sorted(glob(os.path.join(input_path, "*.jsonl")))
    else:
        files = [input_path]

    if not files:
        raise FileNotFoundError("No .jsonl files found.")

    dfs = []
    for fp in files:
        df_part = parse_one_file(fp)
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["text"])
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} comments to: {args.output}")


if __name__ == "__main__":
    main()
