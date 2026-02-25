import logging
import os
import sys
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
from utils.config import parse_args, NUMBER_OF_WORDS, SENTIMENT_OUTPUT_DIR, SENTIMENT_TOP_N
from utils.recommendation import load_review_rdd, build_tfidf_vectors, get_top_k_recommendations
from utils.sentiment import run_sentiment_pipeline


def main():
    reviews_file, product_ids = parse_args()

    if not os.path.exists(reviews_file):
        sys.exit(f"Error: reviews file not found: {reviews_file}")

    spark = SparkSession.builder.appName('amazon-review-analysis').getOrCreate()
    sc = spark.sparkContext
    df = spark.read.json(reviews_file)

    # Recommendation pipeline
    id_and_terms, num_docs = load_review_rdd(df)
    tfidf_vectors = build_tfidf_vectors(sc, id_and_terms, num_docs, NUMBER_OF_WORDS)
    for product_id in product_ids:
        top_k, queried_id = get_top_k_recommendations(product_id, 20, tfidf_vectors)
        logger.info(f'Product recommendations for: {queried_id}')
        for pid in top_k:
            logger.info(f'  {pid}')

    # Sentiment pipeline
    run_sentiment_pipeline(spark, df, SENTIMENT_OUTPUT_DIR, SENTIMENT_TOP_N)

    spark.stop()


if __name__ == '__main__':
    main()
