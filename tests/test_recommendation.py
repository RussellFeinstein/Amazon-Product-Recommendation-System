import numpy as np
import pytest
from utils.recommendation import build_tf_array, build_one_hot_array, cosine_similarity


def test_build_tf_array_sums_to_one():
    result = build_tf_array([0, 1, 1, 2], num_words=5)
    assert pytest.approx(result.sum()) == 1.0


def test_build_tf_array_correct_counts():
    result = build_tf_array([0, 1, 1, 2], num_words=5)
    assert result[0] == pytest.approx(1 / 4)
    assert result[1] == pytest.approx(2 / 4)
    assert result[2] == pytest.approx(1 / 4)
    assert result[3] == pytest.approx(0.0)


def test_build_tf_array_length():
    result = build_tf_array([0, 2], num_words=10)
    assert len(result) == 10


def test_build_one_hot_array_binary():
    result = build_one_hot_array([0, 0, 1, 3], num_words=5)
    assert set(result).issubset({0.0, 1.0})


def test_build_one_hot_array_no_duplicates():
    result = build_one_hot_array([1, 1, 1], num_words=5)
    assert result[1] == 1.0


def test_build_one_hot_array_length():
    result = build_one_hot_array([0, 4], num_words=10)
    assert len(result) == 10


def test_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_known():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0])
    expected = 1.0 / np.sqrt(2)
    assert cosine_similarity(a, b) == pytest.approx(expected)
