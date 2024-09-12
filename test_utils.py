import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = cosine_similarity(vector1, vector2)
    
    # Compute the expected result manually
    expected_dot_product = dot_product(vector1, vector2)
    norm_v1 = np.linalg.norm(vector1)
    norm_v2 = np.linalg.norm(vector2)
    expected_result = expected_dot_product / (norm_v1 * norm_v2)
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    target_vector = np.array([5, 0, 5])
    vectors = np.array([
        [5, 4, 3],
        [5, 1, 5],
        [10, -5, 10],
        [12, 12, 12]
    ])
    
    result = nearest_neighbor(target_vector, vectors)
    
    expected_index = 0 

    assert result == expected_index, f"Expected index {expected_index}, but got {result}"

