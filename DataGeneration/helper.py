"""
Helper rotation matrix functions for data generation
"""

import numpy as np

def custom_rotation_matrix(angle, d):
    # Construct rotation matrix in the orthonormal basis
    angle = np.deg2rad(angle)
    R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    # Construct identity matrix of dimension (n-2)
    I = np.eye(d- 2)
    # Assemble the rotation matrix in the canonical basis
    rotation_matrix = np.block([[R, np.zeros((2, d - 2))], [np.zeros((d - 2, 2)), I]])
    return rotation_matrix

def random_rotation_matrix(d):
    # Generate a random rotation matrix in d dimensions
    random_matrix = np.random.randn(d, d)
    orthogonal_matrix, _ = np.linalg.qr(random_matrix)
    return orthogonal_matrix