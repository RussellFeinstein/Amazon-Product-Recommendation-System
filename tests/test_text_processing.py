from utils.text_processing import (
    remove_stop_words,
    remove_punctuation,
    lemmatize_tokens,
    join_tokens,
)


def test_remove_stop_words_removes_the():
    assert 'the' not in remove_stop_words(['the', 'guitar'])


def test_remove_stop_words_keeps_content_words():
    result = remove_stop_words(['guitar', 'sounds', 'great'])
    assert 'guitar' in result
    assert 'great' in result


def test_remove_punctuation_strips_punctuation():
    result = remove_punctuation(['hello,', 'world!'])
    assert result == ['hello', 'world']


def test_remove_punctuation_removes_empty_strings():
    result = remove_punctuation([',', '!', 'word'])
    assert '' not in result
    assert 'word' in result


def test_lemmatize_tokens_noun():
    result = lemmatize_tokens(['guitars', 'strings'])
    assert 'guitar' in result


def test_join_tokens_space_separated():
    assert join_tokens(['good', 'sound']) == 'good sound'


def test_join_tokens_empty():
    assert join_tokens([]) == ''
