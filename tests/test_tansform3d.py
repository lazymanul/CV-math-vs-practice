import os 
import sys

# add parent dir to import path 
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

import pytest

import numpy as np
import utils.transform3d as t3d


def test_compose():
    t1 = t3d.Transform3d()
    t2 = t3d.Transform3d()
    composed = t1.compose(t2)
    
    assert isinstance(composed, t3d.Transform3d) == True
    assert len(composed._transforms) == 1


def test_get_matrix_identity():
    identity_transform = t3d.Transform3d()

    assert np.allclose(identity_transform.get_matrix(), 
                       np.eye(4, dtype=np.float32))


def test_get_matrix_indentity_composition():
    t1 = t3d.Transform3d()
    t2 = t3d.Transform3d()
    composed = t1.compose(t2)

    assert np.allclose(composed.get_matrix(), 
                       np.eye(4, dtype=np.float32))


def test_get_matrix_composition_order():
    np.random.seed(42)
    m1 = np.random.random_sample((4,4))
    m2 = np.random.random_sample((4,4))
    t1 = t3d.Transform3d(m1)
    t2 = t3d.Transform3d(m2)
    composed = t1.compose(t2)

    assert np.allclose(composed.get_matrix(), m2 @ m1)


def test_inverse_self():
    m = np.array([[1,1,1,1],
                  [0,2,2,2], 
                  [0,0,3,3],
                  [0,0,0,4]])
    t = t3d.Transform3d(matrix=m)

    assert np.allclose(t.inverse().get_matrix(), np.linalg.inv(m))

    
def test_inverse_composition():
    m1 = np.array([[1,1,1,1],
                   [0,2,2,2], 
                   [0,0,3,3],
                   [0,0,0,4]])
    m2 = np.array([[2,2,2,2],
                   [0,3,3,3], 
                   [0,0,4,4],
                   [5,0,0,5]])
    t1 = t3d.Transform3d(m1)
    t2 = t3d.Transform3d(m2)
    composed = t1.compose(t2)    
    inv_composed = composed.inverse()

    assert np.allclose(inv_composed.get_matrix(), 
                       np.linalg.inv(m2 @ m1))
    assert len(inv_composed._transforms) == 2
    assert np.allclose(inv_composed._transforms[0].get_matrix(),
                       np.linalg.inv(m2))
    assert np.allclose(inv_composed._transforms[1].get_matrix(),
                       np.linalg.inv(m1))


def test_transform_points():
    m1 = np.array([[1,1,1,1],
                   [0,2,2,2], 
                   [0,0,3,3],
                   [0,0,0,4]])
    m2 = np.array([[2,2,2,2],
                   [0,3,3,3], 
                   [0,0,4,4],
                   [5,0,0,5]])
    t1 = t3d.Transform3d(m1)
    t2 = t3d.Transform3d(m2)
    composed = t1.compose(t2)    
    inv_composed = composed.inverse()

    points = np.array([[0,0,0],
                       [1,1,1],
                       [1,2,3],
                       [2,3,4]])

    points_transformed = composed.transform_points(points)
    points_restored = inv_composed.transform_points(points_transformed)

    assert np.allclose(points, points_restored)


def test_Translate_init():
    t_matrix = np.array([[1, 0, 0, 1],
                         [0, 1, 0, 2],
                         [0, 0, 1, 3],
                         [0, 0, 0, 1]])
    tvec = np.array((1, 2, 3))
    t = t3d.Translate(tvec)
    assert np.allclose(t.get_matrix(), t_matrix)
    
    t_int = t3d.Translate(tvec, dtype=np.int64)
    assert t_int.get_matrix().dtype == np.int64


def test_Scale_init():
    s_matrix = np.array([[2, 0, 0, 0],
                         [0, 3, 0, 0],
                         [0, 0, 4, 0],
                         [0, 0, 0, 1]])
    svec = np.array((2, 3, 4))
    s = t3d.Scale(svec)
    assert np.allclose(s.get_matrix(), s_matrix)


def test_Rotate_init():
    r_matrix = np.array([[1, 0, 0, 0],
                         [0, 0,-1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
    R = np.array([[1, 0, 0],
                  [0, 0,-1],
                  [0, 1, 0]])
    r = t3d.Rotate(R=R)
    assert np.allclose(r.get_matrix(), r_matrix)