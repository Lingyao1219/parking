# Crowdsourced parking perceptions from Google Maps reviews

## Overview

This project analyzes parking perceptions across the United States using crowdsourced online reviews from Google Maps. Overall, we employ multiple natural language processing techniques and regression analysis to investigate public perceptions of parking and its relationship with various socio-spatial factors.

## Abstract

Due to increased reliance on private vehicles and growing travel demand, parking remains a longstanding urban challenge globally. This study introduces a cost-effective and widely accessible data source, crowdsourced online reviews, to investigate public perceptions of parking across the U.S. We examine 4,987,483 parking-related reviews for 1,129,460 points of interest (POIs) across 911 core-based statistical areas (CBSAs) sourced from Google Maps.

We employ the Bidirectional Encoder Representations from Transformers (BERT) model to classify parking sentiment and conduct regression analyses to explore its relationships with socio-spatial factors. Our findings reveal significant variations in parking sentiment across POI types and CBSAs, with insights into the relationships between urban density, demographics, socioeconomic status, and parking experiences.

## Key Components

1. **Data Processing**: Scripts for filtering and preparing the dataset. 
2. **Modeling**: BERT classifier implementation, RoBERTa sentiment classification, model performance evaluation, and text processing utilities.
3. **Regression Analysis**: Feature building scripts, regression modeling, and results analysis.
4. **Results and Analysis**: Text cleaning, POI analysis, and textual analysis.

## Getting Started

1. Clone the repository
2. Run data processing scripts to prepare the dataset
3. Execute modeling notebooks to train and evaluate the classifiers
4. Perform regression analysis using the R script and Python analysis scripts
5. Analyze results using the provided notebooks in the results folder
