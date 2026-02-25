import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from utils.text_processing import remove_stop_words, remove_punctuation, lemmatize_tokens, join_tokens

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

_STOP_WORDS = set(stopwords.words('english'))
_VADER_ANALYZER = SentimentIntensityAnalyzer()

_SENTENCE_RE = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
_GRAMMAR = r"""
NBAR:
    {<NN.*|JJ>*<NN.*>}

NP:
    {<NBAR>}
    {<NBAR><IN><NBAR>}
"""
_CHUNKER = nltk.RegexpParser(_GRAMMAR)


def extract_noun_phrases(text):
    def leaves(tree):
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            yield subtree.leaves()

    def acceptable_word(word):
        accepted = bool(
            3 < len(word) <= 20
            and 'https' not in word.lower()
            and 'http' not in word.lower()
            and '#' not in word.lower()
        )
        return accepted  # fixed: was `yield accepted`

    def get_terms(tree):
        for leaf in leaves(tree):
            term = [w for w, t in leaf if w not in _STOP_WORDS if acceptable_word(w)]
            yield term

    tokens = nltk.regexp_tokenize(text, _SENTENCE_RE)
    postoks = nltk.tag.pos_tag(tokens)
    tree = _CHUNKER.parse(postoks)
    terms = get_terms(tree)
    phrases = []
    for term in terms:
        if len(term):
            phrases.append(' '.join(term))
    return [w for w in phrases if w]


def score_sentiment(phrases):
    sentiment_list = []
    for phrase in phrases:
        text = ''.join(phrase)
        scores = _VADER_ANALYZER.polarity_scores(text)
        compound = scores['compound']
        if compound < 0.0:
            label = 'Negative'
        elif compound == 0.0:
            label = 'Neutral'
        else:
            label = 'Positive'
        sentiment_list.append((text, label))
    return sentiment_list


def write_top_keywords_csv(spark, sentiment_rdd, output_dir, label, top_n):
    freq_rdd = (
        sentiment_rdd
        .map(lambda x: (x[0], 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda x: x[1], ascending=False)
    )
    df = freq_rdd.toDF()
    df.createOrReplaceTempView('kw_table')
    result = spark.sql(
        f'SELECT _1 AS Keywords, _2 AS Frequency FROM kw_table LIMIT {top_n}'
    )
    result.coalesce(1).write.mode('overwrite').options(header=True).csv(
        f'{output_dir}/{label}'
    )


def run_sentiment_pipeline(spark, reviews_file, output_dir, top_n):
    df = spark.read.json(reviews_file)

    reviews_rdd = df.select('reviewText').rdd.flatMap(lambda x: x)
    tokens_rdd = (
        reviews_rdd
        .map(lambda x: x.lower())
        .map(lambda x: nltk.sent_tokenize(x))
        .map(lambda x: [word for line in x for word in line.split()])
        .map(remove_stop_words)
        .map(remove_punctuation)
        .map(lemmatize_tokens)
        .map(join_tokens)
    )

    sentiment_rdd = tokens_rdd.map(extract_noun_phrases).map(score_sentiment)

    pos_rdd = sentiment_rdd.flatMap(lambda lst: lst).filter(lambda y: y[1] == 'Positive')
    neg_rdd = sentiment_rdd.flatMap(lambda lst: lst).filter(lambda y: y[1] == 'Negative')
    neutral_rdd = sentiment_rdd.flatMap(lambda lst: lst).filter(lambda y: y[1] == 'Neutral')

    write_top_keywords_csv(spark, pos_rdd, output_dir, 'positive', top_n)
    write_top_keywords_csv(spark, neg_rdd, output_dir, 'negative', top_n)
    write_top_keywords_csv(spark, neutral_rdd, output_dir, 'neutral', top_n)
