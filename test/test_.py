import pytest
from pyclassify.utils import distance
from pyclassify.utils import majority_vote
from pyclassify.classifier import kNN

def test_distance():
    # Test valid distance calculation
    point1 = [1.0, 2.0, 3.0]
    point2 = [4.0, 5.0, 6.0]
    expected_distance = 27.0  # (4-1)^2 + (5-2)^2 + (6-3)^2
    assert distance(point1, point2) == expected_distance

    # Test invalid input types
    with pytest.raises(TypeError):
        distance("hey", point2)
    with pytest.raises(TypeError):
        distance(point1, 1.1)

def test_majority_vote():
    # Test majority vote algorithm
    labels = [1, 0, 0, 0]
    expected_majority = 0
    assert majority_vote(labels) == expected_majority

def test_knn_constructor():
    # Test valid kNN constructor
    knn = kNN(k=3)
    assert knn.k == 3

    # Test invalid kNN constructor
    with pytest.raises(TypeError):
        kNN(k='1')
    with pytest.raises(TypeError):
        kNN(k=[1])
    with pytest.raises(TypeError):
        kNN(k=1.0)
