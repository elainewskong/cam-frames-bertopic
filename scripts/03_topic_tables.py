# scripts/03_topic_tables.py
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Create paper-friendly topic tables from BERTopic outputs.")
    parser.add_argument("--output_dir", default="outputs", help="Directory created by 02_fit_bertopic.py")
    parser.add_argument("--save_dir", default="outputs", help="Where to save formatted tables")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    topic_info_path = os.path.join(args.output_dir, "topic_info.csv")
    keywords_path = os.path.join(args.output_dir, "topic_keywords.csv")

    topic_info = pd.read_csv(topic_info_path)
    keywords = pd.read_csv(keywords_path)

    # Clean columns and merge
    topic_info = topic_info.rename(columns={"Topic": "topic", "Count": "count"})
    merged = topic_info.merge(keywords, on="topic", how="left")

    # Remove outlier topic -1 for main tables
    merged_main = merged[merged["topic"] != -1].copy()

    # Table for paper: topic id, count, short label placeholder, top words
    paper_table = merged_main[["topic", "count", "Name", "top_words"]].copy()
    paper_table = paper_table.sort_values("count", ascending=False)

    paper_table.to_csv(os.path.join(args.save_dir, "table_topics_for_paper.csv"), index=False, encoding="utf-8-sig")

    # Optional: top N topics
    top_n = paper_table.head(15)
    top_n.to_csv(os.path.join(args.save_dir, "table_topics_top15.csv"), index=False, encoding="utf-8-sig")

    print("Saved:")
    print(os.path.join(args.save_dir, "table_topics_for_paper.csv"))
    print(os.path.join(args.save_dir, "table_topics_top15.csv"))


if __name__ == "__main__":
    main()
