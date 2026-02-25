import re
import numpy as np


def load_review_rdd(spark, reviews_file):
    all_reviews = spark.read.json(reviews_file).rdd
    num_docs = all_reviews.count()

    id_and_text = all_reviews.map(lambda x: (x['asin'], x['reviewText']))

    regex = re.compile('[^a-zA-Z]')
    id_and_terms = (
        id_and_text
        .map(lambda x: (x[0], regex.sub(' ', x[1]).lower().split()))
        .reduceByKey(lambda x, y: x + y)
    )
    return id_and_terms, num_docs


def build_tf_array(list_of_indices, num_words):
    counts = np.bincount(list_of_indices, minlength=num_words).astype(float)
    return counts / counts.sum()


def build_one_hot_array(list_of_indices, num_words):
    arr = np.bincount(list_of_indices, minlength=num_words)
    return np.clip(arr, 0, 1).astype(float)


def cosine_similarity(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


def build_tfidf_vectors(sc, id_and_terms, num_docs, num_words):
    id_and_terms.cache()

    # Build vocabulary: top num_words terms by corpus frequency
    term_one_pairs = id_and_terms.flatMap(lambda x: x[1]).map(lambda x: (x, 1))
    all_counts = term_one_pairs.reduceByKey(lambda x, y: x + y)
    top_terms = all_counts.takeOrdered(num_words, lambda x: -x[1])

    dictionary = sc.parallelize(range(num_words)).map(lambda x: (top_terms[x][0], x))

    # Build TF vectors
    term_id_pairs = id_and_terms.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    joined = dictionary.join(term_id_pairs)
    doc_pos_pairs = joined.map(lambda x: (x[1][1], [x[1][0]]))
    terms_in_each_doc = (
        doc_pos_pairs
        .reduceByKey(lambda x, y: x + y)
        .map(lambda x: (x[0], sorted(x[1])))
        .cache()
    )

    term_freq = terms_in_each_doc.map(lambda x: (x[0], build_tf_array(x[1], num_words)))
    one_hot = terms_in_each_doc.map(lambda x: (x[0], build_one_hot_array(x[1], num_words)))

    # Build IDF and broadcast to avoid per-task serialisation
    doc_freq = one_hot.reduce(lambda x, y: ('', np.add(x[1], y[1])))[1]
    terms_in_each_doc.unpersist()
    id_and_terms.unpersist()

    inverse_doc_freq = np.log(np.divide(np.full(num_words, num_docs), doc_freq))
    idf_bc = sc.broadcast(inverse_doc_freq)

    tfidf_vectors = term_freq.map(lambda x: (x[0], np.multiply(x[1], idf_bc.value)))
    return tfidf_vectors


def get_top_k_recommendations(product_id, k, tfidf_vectors):
    input_tfidf = tfidf_vectors.filter(lambda x: x[0] == product_id).collect()
    if not input_tfidf:
        raise ValueError(f"Product ID '{product_id}' not found in dataset")
    distances = tfidf_vectors.map(lambda x: (x[0], cosine_similarity(x[1], input_tfidf[0][1])))
    top_k = distances.top(k, lambda x: x[1])[1:]
    top_k_ids = list(zip(*top_k))[0]
    return top_k_ids, input_tfidf[0][0]
