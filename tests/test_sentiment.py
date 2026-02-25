from utils.sentiment import score_sentiment, extract_noun_phrases


def test_score_sentiment_positive():
    result = score_sentiment(['excellent quality'])
    assert any(label == 'Positive' for _, label in result)


def test_score_sentiment_negative():
    result = score_sentiment(['terrible awful horrible'])
    assert any(label == 'Negative' for _, label in result)


def test_score_sentiment_empty_input():
    assert score_sentiment([]) == []


def test_score_sentiment_returns_tuples():
    result = score_sentiment(['good sound'])
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)


def test_extract_noun_phrases_returns_list():
    result = extract_noun_phrases('the guitar has good sound quality')
    assert isinstance(result, list)


def test_acceptable_word_filters_short_words():
    # Words of 3 chars or fewer must not appear in extracted phrases.
    # This verifies the yield->return fix: if acceptable_word still used
    # yield, the filter would be a no-op and short words would leak through.
    result = extract_noun_phrases('the cat sat on the mat')
    flat = ' '.join(result)
    for word in flat.split():
        assert len(word) > 3, f"Short word '{word}' should have been filtered"
