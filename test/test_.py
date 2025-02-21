import pytest
import numpy as np
from pyclassify.utils import distance
from pyclassify.utils import distance_numpy
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
        
def test_distance_numpy():
    # Test valid distance calculation
    point1 = np.array([1.0, 2.0, 3.0])
    point2 = np.array([4.0, 5.0, 6.0])
    point3 = np.array([4.0, 5.0, 6.0, 7.0])
    expected_distance = 27.0  # (4-1)^2 + (5-2)^2 + (6-3)^2
    assert distance_numpy(point1, point2) == expected_distance

    # Test invalid input types
    with pytest.raises(TypeError):
        distance_numpy([1,2,3], point2)
    with pytest.raises(TypeError):
        distance_numpy(point1, [1,2,3])
    with pytest.raises(TypeError):
        distance_numpy(point1, point3)

def test_majority_vote():
    # Test majority vote algorithm
    labels = [1, 0, 0, 0]
    expected_majority = 0
    assert majority_vote(labels) == expected_majority

def test_knn_constructor():
    # Test valid kNN constructor
    knn = kNN(k=3)
    assert knn.k == 3
    assert knn.backhand == 'plain'
    
    knn = kNN(k=3,backhand='numpy')
    assert knn.backhand == 'numpy'
    

    # Test invalid kNN constructor
    with pytest.raises(TypeError):
        kNN(k='1')
    with pytest.raises(TypeError):
        kNN(k=[1])
    with pytest.raises(TypeError):
        kNN(k=1.0)
        
    # Test invalid input backend
    with pytest.raises(ValueError):
    	kNN(k=3,backhand='np')
