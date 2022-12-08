# raw code for camera parameters estimation
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from scipy.linalg import norm

def gen_x_row(x, P):
    return np.hstack((P, [0,0,0,0], -x * P))

def gen_y_row(y, P):
    return np.hstack(([0,0,0,0], P, -y * P))

def estimate_projection_matrix(points, proj_points): # points coords (x,y,z,1), proj_points (x, y, 1)
    rows = []
    for i, p in enumerate(points):
        rows.append(gen_x_row(proj_points[i][0], p))
        rows.append(gen_y_row(proj_points[i][1], p))

    M = np.array(rows)
    if np.linalg.matrix_rank(M) < 11:
        raise ValueError("Degenerate points configuration, can't estimate projection")

    x0 = np.ones(M.shape[1])
    nlc = NonlinearConstraint(lambda x: norm(x), 0.999, 1.001)
    res = minimize(lambda x: norm(M @ x), x0, method='SLSQP', constraints=(nlc))

    return res.x.reshape((3, 4))

def estimate_parameters():
    pass