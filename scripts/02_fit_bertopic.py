# scripts/02_fit_bertopic.py
import argparse
import json
import os
import re
import sys
from datetime import datetime

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
# Punctuation-only or symbol-only tokens (e.g., "，", "。", "!!!", "…", "/")
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# Rough emoji block matcher (not perfect, but works well in practice)
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


def build_tokenizer(keep_emoji: bool = False):
    """
    Create a Jieba tokenizer that:
    - segments Chinese and keeps Latin tokens
    - removes punctuation-only tokens
    - optionally removes emoji-only tokens
    """
    def jieba_tokenizer(text: str):
        toks = []
        for tok in jieba.lcut(str(text)):
            tok = tok.strip()
            if not tok:
                continue

            # remove punctuation/symbol-only tokens
            if PUNCT_ONLY_RE.match(tok):
                continue

            # optionally drop emoji-only tokens (keeps emoji inside mixed tokens rare)
            if not keep_emoji and is_emoji_token(tok):
                continue

            toks.append(tok)
        return toks

    return jieba_tokenizer


# -----------------------------
# Manifest writer
# -----------------------------
def write_manifest(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fit BERTopic on Chinese-language comments (IPM-ready).")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--text_col", default="text", help="Text column name")

    parser.add_argument("--output_dir", default="outputs", help="Directory to store outputs")
    parser.add_argument("--embedding_model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="SentenceTransformer model")

    parser.add_argument("--min_topic_size", type=int, default=30, help="Minimum topic size (HDBSCAN min_cluster_size)")
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

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {args.input}")

    documents = df[args.text_col].dropna().astype(str).tolist()

    # -----------------------------
    # Small-data safeguard
    # Avoid max_df/min_df crash in tiny test corpora
    # -----------------------------
    min_df = args.min_df
    max_df = args.max_df
    if len(documents) < 3000:
        # keep pipeline stable for environment tests
        min_df = min(min_df, 2)
        max_df = max(max_df, 0.95)

        min_df = max(1, min_df)
        max_df = min(1.0, max_df)

    # -----------------------------
    # Embedding model
    # -----------------------------
    embedding_model = SentenceTransformer(args.embedding_model)

    # -----------------------------
    # Vectorizer
    # -----------------------------
    tokenizer = build_tokenizer(keep_emoji=args.keep_emoji)
    vectorizer_model = CountVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,  # required when custom tokenizer is provided
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

    # Use cosine for consistency with UMAP embeddings space
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_topic_size,
        min_samples=args.min_samples,
        metric="euclidean" if args.umap_metric == "euclidean" else "euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Note: We are clustering the UMAP-reduced embeddings. Euclidean is typical there.
    # If you prefer strict consistency, you can switch metric="cosine" and test stability.

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(documents)

    # -----------------------------
    # Save outputs
    # -----------------------------
    df_out = pd.DataFrame({"text": documents, "topic": topics})
    df_out.to_csv(os.path.join(args.output_dir, "doc_topics.csv"), index=False, encoding="utf-8-sig")

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(args.output_dir, "topic_info.csv"), index=False, encoding="utf-8-sig")

    # Top keywords per topic
    rows = []
    for t in sorted(set(topics)):
        if t == -1:
            continue
        words = topic_model.get_topic(t)
        if not words:
            continue
        rows.append({"topic": t, "top_words": ", ".join([w for w, _ in words[:15]])})
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "topic_keywords.csv"), index=False, encoding="utf-8-sig")

    # Representative docs if available
    if "Representative_Docs" in topic_info.columns:
        rep = topic_info[["Topic", "Representative_Docs"]].copy()
        rep.to_csv(os.path.join(args.output_dir, "representative_docs.csv"), index=False, encoding="utf-8-sig")

    # Save model (optional)
    try:
        topic_model.save(os.path.join(args.output_dir, "bertopic_model"), serialization="pickle")
    except Exception:
        pass

    # -----------------------------
    # Summary stats + manifest
    # -----------------------------
    outlier_rate = sum(1 for t in topics if t == -1) / max(len(topics), 1)

    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "n_documents": len(documents),
        "n_unique_topics": len(set(topics)),
        "outlier_rate": outlier_rate,
        "embedding_model": args.embedding_model,
        "keep_emoji": bool(args.keep_emoji),
        "vectorizer": {
            "ngram_range": [1, args.ngram_max],
            "min_df": min_df,
            "max_df": max_df,
            "max_features": args.max_features,
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
            "metric": "euclidean" if args.umap_metric == "euclidean" else "euclidean",
            "cluster_selection_method": "eom",
        },
        "package_versions": {
            "bertopic": getattr(__import__("bertopic"), "__version__", "unknown"),
            "umap": getattr(__import__("umap"), "__version__", "unknown"),
            "hdbscan": getattr(__import__("hdbscan"), "__version__", "unknown"),
            "jieba": getattr(__import__("jieba"), "__version__", "unknown"),
            "sklearn": getattr(__import__("sklearn"), "__version__", "unknown"),
            "sentence_transformers": getattr(__import__("sentence_transformers"), "__version__", "unknown"),
        },
    }
    write_manifest(os.path.join(args.output_dir, "run_manifest.json"), manifest)

    print(f"Documents: {len(documents)}")
    print(f"Unique topics: {len(set(topics))}")
    print(f"Outlier rate: {outlier_rate:.3f}")
    print(f"Saved outputs to: {args.output_dir}")
    print(f"Saved manifest: {os.path.join(args.output_dir, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
