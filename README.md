# Toxicity Detection: Logistic Regression Baseline

## Project Context

This project explores the efficiency of traditional machine learning algorithms in solving text classification problems.

While the industry trend leans heavily toward large Deep Learning models (like BERT or RoBERTa) for Natural Language Processing, this repository demonstrates that a simple **Logistic Regression** classifier can often achieve production-grade accuracy with a fraction of the compute time and resources.

## Problem Statement

The goal is to automatically flag comments from an online discussion forum as "Toxic" or "Not Toxic."

The dataset originates from the **Jigsaw Unintended Bias in Toxicity Classification** challenge. While the original challenge posed this as a regression problem (predicting a toxicity score from 0.0 to 1.0), this project simplifies the task into a binary classification problem to determine immediate content moderation needs.

## The Baseline Implementation

The current script (`toxicity_model.py`) establishes a baseline using a Bag-of-Words approach weighted by TF-IDF.

### Key Decisions

1.  **Thresholding:** The original data contains a float value for toxicity. We have arbitrarily set a threshold of **0.6**.
      * Score $< 0.6$: Label 0 (Safe)
      * Score $\ge 0.6$: Label 1 (Toxic)
      * *Reasoning:* This threshold allows for "free speech" and debate without flagging mildly heated arguments, focusing only on comments that are likely to be harmful.
2.  **Vectorization:** We utilize **TF-IDF (Term Frequency-Inverse Document Frequency)** rather than word embeddings. This creates a sparse matrix representing the importance of specific words relative to the corpus.
3.  **Algorithm:** We use **Logistic Regression**. Note that the code comments in previous versions may have referred to "Naive Bayes," but the actual implementation utilizes `sklearn.linear_model.LogisticRegression`.

## Prerequisites

### Libraries

Install the required dependencies:

```bash
pip install pandas scikit-learn numpy
```

### Data Configuration

**Important:** The script currently contains hardcoded file paths pointing to a local directory.

  * Line 14: `df = pd.read_csv("C:/Users/...")`
  * Line 49: `test = pd.read_csv("C:/Users/...")`

**Action Required:** You must update these paths to point to your local copy of the `train.csv` and `test.csv` files before running the script.

## The Challenge

Your goal is to iterate on this baseline. The current model trains in minutes and achieves acceptable accuracy. Can you improve performance without exponentially increasing training time?

### Recommended Experiments:

1.  **Regression vs. Classification:** The current script forces a binary target. Modify the code to perform a **Regression** analysis instead (predicting the actual float value). Does the Logistic Regressor handle this well, or do you need a different algorithm (e.g., Ridge Regression or Support Vector Regression)?
2.  **Compare Algorithms:** The code comments mention Naive Bayes. Implement a `MultinomialNB` classifier and compare its F1-score and training time against the Logistic Regression baseline.
3.  **Deep Learning:** If you have the compute resources, implement an LSTM (Long Short-Term Memory) network or fine-tune a HuggingFace Transformer. Compare the inference speed of these complex models against the Logistic Regression baseline. Is the accuracy gain worth the extra latency?

## Usage

Run the script to train the model and output the classification report:

```bash
python toxicity_model.py
```

## Performance Note

Initial tests with this simple architecture have shown high accuracy (approx. 96% in some iterations) with very low training overhead (approx. 20 minutes). This serves as the benchmark for any future complex implementations.
