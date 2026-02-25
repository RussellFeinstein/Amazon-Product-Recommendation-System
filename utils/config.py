import sys

NUMBER_OF_WORDS = 1000
SENTIMENT_OUTPUT_DIR = "TermProjectResults"
SENTIMENT_TOP_N = 20


def parse_args():
    if len(sys.argv) != 3:
        sys.exit("Usage: python main.py <reviews_file> <product_id>")
    return sys.argv[1], sys.argv[2]
