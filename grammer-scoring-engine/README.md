# Grammar Scoring Engine for Spoken Audio

## General Overview

This project presents a Grammar Scoring Engine for spoken audio samples, developed as part of the SHL Research Intern assessment.
Given an audio recording of a speaker (45–60 seconds), the task is to predict a continuous grammar score between 0 and 5, following a MOS Likert-style rubric.

The solution focuses on building a **clean, interpretable, and reproducible** machine learning pipeline, prioritizing methodological clarity over model complexity.

## Problem Framing
Grammar is inherently a **linguistic property of text**. When inferred from speech alone, a model cannot directly evaluate grammatical correctness. Instead, it learns from different signals such as fluency, articulation patterns, and pause structure, which may indirectly reflect grammatical competence.

This project explicitly acknowledges this limitation and frames the task as **learning indirect indicators of grammar proficiency from audio signals**.

## Dataset
#### [**NOTE** -> Due to Github Limitations for uploading large files, the specific audio data is not being uploaded]

* Training set: 409 audio samples with grammar scores
* Test set: 197 audio samples (unlabeled)
* Audio format: `.wav`
* Duration: ~45–60 seconds per sample

### Label Distribution
* Grammar scores range from 1 to 5
* Mean grammar score ~ 2.9
* Distribution is bimodal, with higher concentration around mid-range scores
* Extreme low and high scores are relatively sparse

This distribution has important implications for model behavior and error patterns.


## Methodology

### 1. Audio Preprocessing
- All audio files are resampled to 16 kHz for consistency
- Duration analysis confirms samples lie predominantly between 45–60 seconds
- No trimming or padding is applied, as durations are comparable
- Variability is handled at the feature aggregation stage

### 2. Feature Extraction
To convert variable-length audio signals into fixed-size representations:
- Mel-Frequency Cepstral Coefficients (MFCCs) are extracted
- 20 MFCC coefficients are computed per frame
- Temporal aggregation:
    - Mean
    - Standard deviation

This results in a 40-dimensional feature vector per audio sample, capturing global spectral characteristics related to speech articulation and fluency.

### 3. Modelling
The task is treated as a regression problem.
- Baseline Model: Ridge Regression
    - Provides a stable, regularized linear baseline
- Improved Model: Random Forest Regressor
    - Captures non-linear relationships between MFCC features and grammar scores

The Random Forest model is selected as the final model due to modest but consistent improvements in validation performance.

## Evaluation
### Metrics
- Root Mean Squared Error (RMSE)
- Pearson Correlation

[**NOTE** : As per evaluation requirement, the **Training RMSE is explicitly computed and reported** in the notebook]

### Observations
- Models tend to predict scores closer to the mean
- Prediction errors increase for extreme grammar scores
- This behavior aligns with the bimodal label distribution and limited data at the extremes

Interpretability is supported through:
- Actual vs Predicted scatter plots
- Error distribution and residual analysis

## Limitations
- Grammar is fundamentally a textual construct; audio-only models capture indirect correlates
- MFCC aggregation loses fine-grained temporal and syntactic cues
- Dataset size limits generalization, especially for extreme scores


## Future Work
Potential extensions include:

- Automatic Speech Recognition (ASR) followed by text-based grammar features
- Ordinal regression approaches aligned with Likert-style scoring
- Speaker normalization and prosodic feature enrichment
- Larger and more diverse training datasets

## Conclusion

This entire thing aimed to develop a grammar scoring engine that takes an audio file as input and outputs a continuous grammar score ranging from 0 to 5.

**Skills Demonstrated:**
- Data Analysis and Visualization
- Feature Engineering
- Machine Learning Model Development and Evaluation
- Basic Audio Handling Concepts

## Hope this adds value to your journey of learning more about how Scoring Engines work !
