import os
import sys
from pyspark.sql import SparkSession
from utils.config import parse_args, NUMBER_OF_WORDS, SENTIMENT_OUTPUT_DIR, SENTIMENT_TOP_N
from utils.recommendation import load_review_rdd, build_tfidf_vectors, get_top_k_recommendations
from utils.sentiment import run_sentiment_pipeline


def main():
    reviews_file, product_id = parse_args()

    if not os.path.exists(reviews_file):
        sys.exit(f"Error: reviews file not found: {reviews_file}")

    spark = SparkSession.builder.appName('amazon-review-analysis').getOrCreate()
    sc = spark.sparkContext

    # Recommendation pipeline
    id_and_terms, num_docs = load_review_rdd(spark, reviews_file)
    tfidf_vectors = build_tfidf_vectors(sc, id_and_terms, num_docs, NUMBER_OF_WORDS)
    top_k, queried_id = get_top_k_recommendations(product_id, 20, tfidf_vectors)
    print(f'Product recommendations for: {queried_id}')
    for pid in top_k:
        print(pid)

    # Sentiment pipeline
    run_sentiment_pipeline(spark, reviews_file, SENTIMENT_OUTPUT_DIR, SENTIMENT_TOP_N)

    spark.stop()


if __name__ == '__main__':
    main()
