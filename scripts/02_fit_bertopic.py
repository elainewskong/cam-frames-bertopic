# scripts/02_fit_bertopic.py
import argparse
import json
import os
import re
import sys
from datetime import datetime
from collections import Counter
from importlib.metadata import version as pkg_version

import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

import umap
import hdbscan
import jieba

# -----------------------------
# Token filtering utilities
# -----------------------------
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)


def is_emoji_token(tok: str) -> bool:
    return bool(EMOJI_RE.fullmatch(tok))


def build_tokenizer(keep_emoji: bool = False, drop_punct_only: bool = True):
    """
    Jieba tokenizer for Chinese + mixed Latin tokens.
    Filters:
      - optional drop punctuation/symbol-only tokens
      - optional drop emoji-only tokens
    """
    def jieba_tokenizer(text: str):
        toks = []
        for tok in jieba.lcut(str(text)):
            tok = tok.strip()
            if not tok:
                continue

            if drop_punct_only and PUNCT_ONLY_RE.match(tok):
                continue

            if not keep_emoji and is_emoji_token(tok):
                continue

            toks.append(tok)
        return toks

    return jieba_tokenizer


def write_manifest(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_pkg_version(name: str) -> str:
    try:
        return pkg_version(name)
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Fit BERTopic on Chinese-language comments (IPM-ready).")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--text_col", default="text", help="Text column name")

    parser.add_argument("--output_dir", default="outputs", help="Directory to store outputs")
    parser.add_argument("--embedding_model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="SentenceTransformer model")

    parser.add_argument("--min_topic_size", type=int, default=30, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=10, help="HDBSCAN min_samples")

    parser.add_argument("--ngram_max", type=int, default=2, help="Max ngram length")
    parser.add_argument("--min_df", type=int, default=3, help="Vectorizer min_df")
    parser.add_argument("--max_df", type=float, default=0.90, help="Vectorizer max_df")
    parser.add_argument("--max_features", type=int, default=5000, help="Vectorizer max_features")

    # UMAP parameters
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--min_dist", type=float, default=0.0)
    parser.add_argument("--umap_metric", default="cosine")
    parser.add_argument("--random_state", type=int, default=42)

    # Tokenization options
    parser.add_argument("--keep_emoji", action="store_true", help="Keep emoji-only tokens in modeling")
    parser.add_argument("--drop_punct_only", action="store_true", help="Drop punctuation-only tokens (recommended)")

    # Pipeline stability
    parser.add_argument("--testing_mode", action="store_true",
                        help="Force stable vectorizer settings (min_df=1, max_df=1.0) for small/test runs")

    # Topic reduction
    parser.add_argument("--reduce_to", type=int, default=40, help="Reduce topics to this number (incl. -1 outlier topic)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {args.input}")

    documents = df[args.text_col].dropna().astype(str).tolist()

    # -----------------------------
    # Vectorizer parameters (stable + reproducible)
    # -----------------------------
    if args.testing_mode:
        min_df, max_df = 1, 1.0
    else:
        min_df, max_df = args.min_df, args.max_df

    # -----------------------------
    # Embedding model
    # -----------------------------
    embedding_model = SentenceTransformer(args.embedding_model)

    # -----------------------------
    # Vectorizer
    # -----------------------------
    tokenizer = build_tokenizer(
        keep_emoji=args.keep_emoji,
        drop_punct_only=args.drop_punct_only
    )

    vectorizer_model = CountVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        ngram_range=(1, args.ngram_max),
        min_df=min_df,
        max_df=max_df,
        max_features=args.max_features,
    )

    # -----------------------------
    # UMAP + HDBSCAN
    # -----------------------------
    umap_model = umap.UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        min_dist=args.min_dist,
        metric=args.umap_metric,
        random_state=args.random_state,
    )

    # Euclidean is typical in UMAP-reduced space
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_topic_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )

    # -----------------------------
    # Fit (pre-reduction)
    # -----------------------------
    topics_pre, _ = topic_model.fit_transform(documents)

    cnt_pre = Counter(topics_pre)
    outliers_pre = cnt_pre.get(-1, 0)
    outlier_rate_pre = outliers_pre / max(len(topics_pre), 1)
    print("Pre-reduction topic counts (top 10):", cnt_pre.most_common(10))

    # Save pre-reduction outputs
    pd.DataFrame({"text": documents, "topic": topics_pre}).to_csv(
        os.path.join(args.output_dir, "doc_topics_pre_reduction.csv"),
        index=False, encoding="utf-8-sig"
    )
    topic_model.get_topic_info().to_csv(
        os.path.join(args.output_dir, "topic_info_pre_reduction.csv"),
        index=False, encoding="utf-8-sig"
    )

    # -----------------------------
    # Reduce topics (post-reduction)
    # -----------------------------
    topic_model.reduce_topics(documents, nr_topics=args.reduce_to)
    topics_post = topic_model.topics_

    cnt_post = Counter(topics_post)
    outliers_post = cnt_post.get(-1, 0)
    outlier_rate_post = outliers_post / max(len(topics_post), 1)
    print("Post-reduction topic counts (top 10):", cnt_post.most_common(10))

    # Save post-reduction outputs
    pd.DataFrame({"text": documents, "topic": topics_post}).to_csv(
        os.path.join(args.output_dir, "doc_topics.csv"),
        index=False, encoding="utf-8-sig"
    )

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(args.output_dir, "topic_info.csv"), index=False, encoding="utf-8-sig")

    # Top keywords per topic
    rows = []
    for t in sorted(set(topics_post)):
        if t == -1:
            continue
        words = topic_model.get_topic(t)
        if not words:
            continue
        rows.append({"topic": t, "top_words": ", ".join([w for w, _ in words[:15]])})
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "topic_keywords.csv"), index=False, encoding="utf-8-sig")

    # Representative docs if available
    if "Representative_Docs" in topic_info.columns:
        topic_info[["Topic", "Representative_Docs"]].to_csv(
            os.path.join(args.output_dir, "representative_docs.csv"),
            index=False, encoding="utf-8-sig"
        )

    # Save model (optional)
    try:
        topic_model.save(os.path.join(args.output_dir, "bertopic_model"), serialization="pickle")
    except Exception:
        pass

    # -----------------------------
    # Manifest
    # -----------------------------
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "n_documents": len(documents),

        "pre_reduction": {
            "n_unique_topics": len(set(topics_pre)),
            "outliers": outliers_pre,
            "outlier_rate": outlier_rate_pre,
        },
        "post_reduction": {
            "reduce_to": args.reduce_to,
            "n_unique_topics": len(set(topics_post)),
            "outliers": outliers_post,
            "outlier_rate": outlier_rate_post,
        },

        "embedding_model": args.embedding_model,
        "tokenizer": {
            "jieba": True,
            "keep_emoji": bool(args.keep_emoji),
            "drop_punct_only": bool(args.drop_punct_only),
        },
        "vectorizer": {
            "ngram_range": [1, args.ngram_max],
            "min_df": min_df,
            "max_df": max_df,
            "max_features": args.max_features,
            "testing_mode": bool(args.testing_mode),
        },
        "umap": {
            "n_neighbors": args.n_neighbors,
            "n_components": args.n_components,
            "min_dist": args.min_dist,
            "metric": args.umap_metric,
            "random_state": args.random_state,
        },
        "hdbscan": {
            "min_cluster_size": args.min_topic_size,
            "min_samples": args.min_samples,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        },
        "package_versions": {
            "bertopic": safe_pkg_version("bertopic"),
            "umap-learn": safe_pkg_version("umap-learn"),
            "hdbscan": safe_pkg_version("hdbscan"),
            "jieba": safe_pkg_version("jieba"),
            "scikit-learn": safe_pkg_version("scikit-learn"),
            "sentence-transformers": safe_pkg_version("sentence-transformers"),
            "pandas": safe_pkg_version("pandas"),
        },
    }

    write_manifest(os.path.join(args.output_dir, "run_manifest.json"), manifest)

    print(f"Documents: {len(documents)}")
    print(f"Saved outputs to: {args.output_dir}")
    print(f"Saved manifest: {os.path.join(args.output_dir, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
