# Information Frames in Chinese-Language Online Discourse on Cancer-Related CAM

**Status:** Research Code / Manuscript in Preparation  
**Methods:** BERTopic, Sentence-Transformers, Interpretive Frame Analysis

This repository contains the replication code and documentation for the study:

> **"Epistemic validation in Chinese-language cancer-related CAM discourse on YouTube: Identifying information frames"** > *Target Journal: Aslib Journal of Information Management*

This project analyzes Chinese-language YouTube comments related to cancer and Complementary and Alternative Medicine (CAM). It employs a **human-in-the-loop topic modeling pipeline** (BERTopic) to identify how health information is processed, evaluated, and framed across cultural boundaries.

## Configuration

Model parameters and preprocessing settings are documented in
`configs/bertopic_config.yaml`.
This file records embedding models, clustering parameters,
and language-specific preprocessing choices used in the analysis.

## Repository Structure

```text
cam-frames-bertopic/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── bertopic_config.yaml        # Documented model and preprocessing parameters
│
├── scripts/
│   ├── 00_parse_youtube_jsonl.py    # Parse YouTube comment JSONL files
│   ├── 01_clean_preprocess.py       # Culturally sensitive text preprocessing
│   ├── 02_fit_bertopic.py           # BERTopic modeling pipeline
│   ├── 03_topic_tables.py           # Topic summary tables for analysis and reporting
│   └── 04_frame_mapping_template.py # Human coding template for information frames
│
└── data_sample/
    └── README_sample_data.md        # Description of sample data usage
