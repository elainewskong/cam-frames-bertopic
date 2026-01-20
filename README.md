# Information Frames in Chinese-Language Online Discourse on Cancer-Related CAM

**Status:** Research Code / Manuscript in Preparation  
**Methods:** BERTopic, Sentence-Transformers, Interpretive Frame Analysis

This repository contains the replication code and documentation for the study:

> **"Cultural and Linguistic Frames in Online Cancer-Related CAM Discourse: A Topic Modeling Study of Chinese Language Communities"** > *Target Journal: Information Processing & Management (IPM)*

This project analyzes Chinese-language YouTube comments related to cancer and Complementary and Alternative Medicine (CAM). It employs a **human-in-the-loop topic modeling pipeline** (BERTopic) to identify how health information is processed, evaluated, and framed across cultural boundaries.

## Configuration

Model parameters and preprocessing settings are documented in
`configs/bertopic_config.yaml`.
This file records embedding models, clustering parameters,
and language-specific preprocessing choices used in the analysis.

## Repository Structure

```text
├── data/
│   ├── sample_data.csv       # Synthetic/anonymized sample for testing pipeline
│   ├── stopwords_custom.txt  # Culturally-sensitive stopword list
│   └── medical_dict.txt      # Custom dictionary for Jieba segmentation (TCM + Biomed terms)
├── notebooks/
│   ├── 01_preprocessing.ipynb   # Cleaning, segmentation, and deduplication
│   ├── 02_bertopic_modeling.ipynb # Embedding generation and initial clustering
│   └── 03_visualization.ipynb   # Topic distance maps and hierarchy plots
├── src/
│   └── utils.py              # Helper functions for text normalization
├── requirements.txt          # Dependencies
└── README.md
