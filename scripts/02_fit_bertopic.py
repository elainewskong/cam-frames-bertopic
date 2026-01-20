# scripts/02_fit_bertopic.py
import argparse
import os

import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

import umap
import hdbscan
import jieba


def jieba_tokenizer(text: str):
    # Use jieba to segment Chinese; keep Latin tokens (PSA, chemo) as is
    return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]


def main():
    parser = argparse.ArgumentParser(description="Fit BERTopic on Chinese-language comments.")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--text_col", default="text", help="Text column name")
    parser.add_argument("--output_dir", default="outputs", help="Directory to store outputs")
    parser.add_argument("--embedding_model", default="paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer model")
    parser.add_argument("--min_topic_size", type=int, default=30, help="Minimum topic size")
    parser.add_argument("--ngram_max", type=int, default=2, help="Max ngram length")
    parser.add_argument("--min_df", type=int, default=3, help="Vectorizer min_df")
    parser.add_argument("--max_df", type=float, default=0.90, help="Vectorizer max_df")
    parser.add_argument("--max_features", type=int, default=5000, help="Vectorizer max_features")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {args.input}")

    documents = df[args.text_col].dropna().astype(str).tolist()

    # Embeddings
    embedding_model = SentenceTransformer(args.embedding_model)

    # Vectorizer with jieba tokenizer
    # Use token_pattern=None when passing a custom tokenizer.
    vectorizer_model = CountVectorizer(
        tokenizer=jieba_tokenizer,
        token_pattern=None,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )

    # UMAP + HDBSCAN
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_topic_size,
        min_samples=10,
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

    topics, probs = topic_model.fit_transform(documents)

    # Save core outputs
    df_out = pd.DataFrame({"text": documents, "topic": topics})
    df_out.to_csv(os.path.join(args.output_dir, "doc_topics.csv"), index=False, encoding="utf-8-sig")

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(args.output_dir, "topic_info.csv"), index=False, encoding="utf-8-sig")

    # Save top keywords per topic
    rows = []
    for t in sorted(set(topics)):
        if t == -1:
            continue
        words = topic_model.get_topic(t)
        if not words:
            continue
        rows.append(
            {
                "topic": t,
                "top_words": ", ".join([w for w, _ in words[:15]]),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "topic_keywords.csv"), index=False, encoding="utf-8-sig")

    # Save representative docs
    # Representative docs are stored in topic_info under Representative_Docs for BERTopic versions that support it.
    if "Representative_Docs" in topic_info.columns:
        rep = topic_info[["Topic", "Representative_Docs"]].copy()
        rep.to_csv(os.path.join(args.output_dir, "representative_docs.csv"), index=False, encoding="utf-8-sig")

    # Save model (optional; can be large)
    try:
        topic_model.save(os.path.join(args.output_dir, "bertopic_model"), serialization="pickle")
    except Exception:
        pass

    outlier_rate = sum(1 for t in topics if t == -1) / max(len(topics), 1)
    print(f"Documents: {len(documents)}")
    print(f"Unique topics: {len(set(topics))}")
    print(f"Outlier rate: {outlier_rate:.3f}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
