# Crowdsourced parking perceptions from Google Maps reviews

## Overview

This project analyzes parking perceptions across the United States using crowdsourced online reviews from Google Maps. Overall, we employ multiple natural language processing techniques and regression analysis to investigate public perceptions of parking and its relationship with various socio-spatial factors.

## Abstract

Due to increased reliance on private vehicles and growing travel demand, parking remains a longstanding urban challenge globally. This study introduces a cost-effective and widely accessible data source, crowdsourced online reviews, to investigate public perceptions of parking across the U.S. We examine 4,987,483 parking-related reviews for 1,129,460 points of interest (POIs) across 911 core-based statistical areas (CBSAs) sourced from Google Maps.

We employ the Bidirectional Encoder Representations from Transformers (BERT) model to classify parking sentiment and conduct regression analyses to explore its relationships with socio-spatial factors. Our findings reveal significant variations in parking sentiment across POI types and CBSAs, with insights into the relationships between urban density, demographics, socioeconomic status, and parking experiences.

## Key Components

1. **Data Processing**: Scripts for filtering and preparing the dataset.
   - `data_filtering.py`: Filters and cleans raw data from Google Maps reviews published by UCSD (https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/).

2. **Modeling**: BERT classifier implementation, sentiment classification using various methods, model performance evaluation, and text processing utilities.
   - `bert_classifier.ipynb`: Implements and trains the BERT model for perception classification.
   - `sentiment_classifiers.ipynb`: Explores various sentiment classification techniques, including RoBERTa-based sentiment and Vader sentiment.
   - `tfidf_classifiers.ipynb`: Implements TF-IDF based classifiers for comparison.
   - `model_performance.ipynb`: Evaluates and compares performance of different models.
   - `process_text.py`: Contains utility functions for text preprocessing.

3. **Regression Analysis**: Feature building scripts, regression modeling, and results analysis.
   - `Feature_build_CBSA.py`: Builds features at the CBSA level.
   - `Feature_build_byPOI.py`: Constructs features at the POI level.
   - `Feature_build_total.py`: Generates overall features for the entire dataset.
   - `Model_Regression.R`: R script for running regression models.
   - `Results_analysis.py`: Analyzes and interprets regression results with local socioeconomic factors.

4. **Results and Analysis**: Text cleaning, POI analysis, and textual analysis.
   - `clean_text.py`: Cleans and preprocesses text data for LSVA textual analysis.
   - `poi_analysis.ipynb`: Analyzes patterns and trends across different POI types.
   - `textual_analysis.ipynb`: Performs in-depth analysis of textual content in reviews.
   - `stop_words.py`: Defines and manages stop words for LSVA textual analysis.

## Getting Started

1. Clone the repository
2. Install required dependencies:
   - Python 3.7+
   - R 4.0+
   - Python Libraries: pandas, numpy, nltk, torch, scikit-learn, transformers
3. Run data processing scripts to prepare the dataset
4. Execute modeling notebooks to train and evaluate the classifiers
5. Perform regression analysis using the R script and Python analysis scripts
6. Analyze results using the provided notebooks in the results folder


## Data Availability

The original data used in this study comes from: 
1. **Google Maps Reviews**: 
   - Source: "Google local review data" published by researchers from UCSD [1, 2]
   - Available for download from the UCSD website (link to be provided)

Please note that use of this data must comply with the original data providers' terms of service and any applicable licensing agreements.

References:
[1] Li, J., Shang, J., McAuley, J.: Uctopic: Unsupervised contrastive learning for phrase representations and topic mining. Annual Meeting of the Association for Computational Linguistics (ACL), 2022.
[2] Yan, A., He, Z., Li, J., Zhang, T., McAuley, J.: Personalized showcases: Generating multimodal explanations for recommendations. The 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 2023.
