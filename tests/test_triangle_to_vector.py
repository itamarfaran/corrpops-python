import numpy as np
from triangle_vector import triangle_to_vector, vector_to_triangle
import unittest


class TestTriangleVector(unittest.TestCase):
    def test_triangle_to_vector(self):
        arr = np.arange(16).reshape((4, 4))
        arr = np.stack([arr, arr])
        arr = triangle_to_vector(arr, True)
        print(vector_to_triangle(arr, True))
        print(arr)
        print(triangle_to_vector(arr, True))
        print(triangle_to_vector(arr, False))

        print(triangle_to_vector(arr, True))
        print(triangle_to_vector(arr, False))



