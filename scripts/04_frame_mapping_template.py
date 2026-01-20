# scripts/04_frame_mapping_template.py
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Create a template to map BERTopic topics to information frames.")
    parser.add_argument("--topic_info", default="outputs/topic_info.csv", help="Path to topic_info.csv")
    parser.add_argument("--topic_keywords", default="outputs/topic_keywords.csv", help="Path to topic_keywords.csv")
    parser.add_argument("--output", default="outputs/frame_mapping_template.csv", help="Output CSV path")
    args = parser.parse_args()

    topic_info = pd.read_csv(args.topic_info)
    topic_info = topic_info.rename(columns={"Topic": "topic", "Count": "count"})

    kw = pd.read_csv(args.topic_keywords)

    df = topic_info.merge(kw, on="topic", how="left")
    df = df[df["topic"] != -1].copy()

    # Add coding columns
    df["frame"] = ""  # Frame 1, Frame 2, etc.
    df["frame_label"] = ""  # Cultural authority, evidence negotiation, etc.
    df["topic_label_final"] = ""  # Your final human-assigned topic label
    df["notes"] = ""  # Short rationale

    keep_cols = ["topic", "count", "Name", "top_words", "frame", "frame_label", "topic_label_final", "notes"]
    df = df[keep_cols].sort_values("count", ascending=False)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved frame mapping template to: {args.output}")


if __name__ == "__main__":
    main()
