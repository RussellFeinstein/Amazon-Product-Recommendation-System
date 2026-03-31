# Amazon Product Recommendation System

A PySpark-based NLP pipeline that generates product recommendations using TF-IDF cosine similarity and analyzes review sentiment using VADER.

## What It Does

Given an Amazon product review dataset (JSON):

1. **Recommendation engine** -- Builds TF-IDF vectors from review text across all products, then finds the top-K most similar products to a queried product ID using cosine similarity.

2. **Sentiment analysis** -- Extracts noun phrases from reviews, scores them with VADER, and outputs the top keywords for positive, negative, and neutral sentiment as CSVs.

## NLP Methodology

### Document Retrieval via TF-IDF

The recommendation pipeline constructs a **document-term matrix** from review text aggregated per product:

1. **Vocabulary construction** -- Selects the top 1,000 terms by corpus frequency from the full review corpus
2. **Term Frequency (TF)** -- Computes normalized term-frequency vectors per product (`bincount / sum`)
3. **Inverse Document Frequency (IDF)** -- Computes `log(N / df)` document-frequency weighting across all products. The IDF vector is **broadcast** to all Spark workers to avoid per-task serialization overhead.
4. **TF-IDF vectors** -- Element-wise multiplication of TF and IDF vectors, producing sparse representations in a high-dimensional vector space
5. **Cosine similarity** -- Computes `dot(a, b) / (||a|| * ||b||)` between the query product's TF-IDF vector and all other products, returning the top-K most similar

### Sentiment Analysis via Noun Phrase Extraction

The sentiment pipeline applies linguistic analysis to extract opinion-bearing phrases:

1. **Sentence tokenization and normalization** -- Splits reviews into sentences, tokenizes, removes stop words, strips punctuation, and applies WordNet lemmatization
2. **POS tagging** -- Tags tokens using NLTK's averaged perceptron tagger
3. **Noun phrase chunking** -- Applies a regex grammar (`{<NN.*|JJ>*<NN.*>}`) to extract noun phrases (nouns optionally preceded by adjectives)
4. **VADER sentiment scoring** -- Scores each extracted noun phrase using VADER's lexicon-based compound sentiment score, classifying as Positive (> 0), Negative (< 0), or Neutral (= 0)
5. **Frequency aggregation** -- Aggregates keyword frequencies per sentiment label and writes the top-N to CSV via Spark SQL

### Dataset

This pipeline was developed and tested on the [Amazon Product Data](https://jmcauley.ucsd.edu/data/amazon/) research datasets (McAuley et al., UCSD) -- line-delimited JSON files with product IDs (`asin`) and review text (`reviewText`).

## Tech Stack

- **Apache Spark / PySpark** -- distributed text processing and TF-IDF computation
- **NLTK** -- tokenization, POS tagging, noun phrase chunking, VADER sentiment, WordNet lemmatization
- **NumPy** -- TF-IDF vector math and cosine similarity
- **pytest** -- unit tests for all pure functions

## Setup

### Requirements

- Python 3.8+
- Java 8 or 11 (required by Spark)
- Apache Spark 3.x

### Install

```bash
pip install pyspark nltk numpy pytest
```

NLTK data packages are downloaded automatically on first run (`vader_lexicon`, `stopwords`, `punkt`, `averaged_perceptron_tagger`, `wordnet`).

## Usage

```bash
python main.py <reviews_file.json> <product_id> [product_id ...]
```

**Example:**

```bash
python main.py data/reviews_Electronics_5.json B00007GDFV B000068O48
```

This will:
- Print the top 20 recommended products for each queried product ID
- Write sentiment keyword CSVs to `TermProjectResults/`

### Input Format

The input file must be line-delimited JSON with at least these fields per record:
- `asin` -- Amazon product ID
- `reviewText` -- the review body

This format matches the [Amazon Product Data](https://jmcauley.ucsd.edu/data/amazon/) research datasets.

## Architecture

```
main.py                          CLI entry point, Spark session management
utils/
  config.py                      Constants (vocab size, output dir) and arg parser
  recommendation.py              TF-IDF pipeline: load -> build vectors -> cosine similarity -> top-K
  sentiment.py                   VADER pipeline: noun phrases -> score -> CSV output
  text_processing.py             Shared NLP: stop words, punctuation, lemmatization
tests/
  test_recommendation.py         TF array, one-hot encoding, cosine similarity
  test_sentiment.py              Noun phrase extraction, VADER scoring
  test_text_processing.py        Tokenization pipeline functions
```

### Recommendation Pipeline

1. Load reviews as RDD, extract `(product_id, review_text)` pairs
2. Tokenize and normalize text (lowercase, alphanumeric only)
3. Build vocabulary from top 1,000 terms by corpus frequency
4. Compute TF (term frequency) vectors per product
5. Compute IDF (inverse document frequency) from document-frequency counts
6. Multiply TF * IDF, broadcast IDF to avoid per-task serialization
7. For each queried product, compute cosine similarity against all other products
8. Return top-K most similar product IDs

### Sentiment Pipeline

1. Tokenize reviews into sentences, then words
2. Remove stop words and punctuation; lemmatize
3. Extract noun phrases using POS tagging and regex chunking
4. Score each phrase with VADER compound sentiment
5. Label as Positive / Negative / Neutral
6. Aggregate keyword frequencies per label, write top-N to CSV

## Running Tests

```bash
pytest
```

Tests cover all pure functions (no Spark dependency):
- `test_recommendation.py` -- TF array normalization, one-hot encoding, cosine similarity edge cases
- `test_sentiment.py` -- VADER scoring polarity, noun phrase extraction
- `test_text_processing.py` -- stop word removal, punctuation stripping, lemmatization
