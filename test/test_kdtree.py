import numpy as np

from networker.geomath import nn_dists, ang_to_vec_coords, euclidean_distance
from networker.classes.kdtree import KDTree


def simple_coord_set():

    coords = np.array([
       [0.35893606, 0.67506964],
       [0.33262164, 0.66603501],
       [0.34752438, 0.68935952]])

    return coords


def test_kdtree():

    simple_coords = simple_coord_set()
    proj_coords = ang_to_vec_coords(simple_coords)
    coord_nn_dists = nn_dists(simple_coords)
    proj_nn_dists = nn_dists(proj_coords, spherical=False)

    coord_idx = np.arange(proj_coords.shape[0])
    kdt = KDTree(proj_coords)
    kd_nn_idx = [kdt.query_subset(proj_coords[i],
                coord_idx[coord_idx != i])[0] for i in range(len(proj_coords))]

    assert all(coord_nn_dists[1] == proj_nn_dists[1]),\
        "spherical and cartesian adjacency do not match"
    assert all(kd_nn_idx == proj_nn_dists[1]),\
        "kdtree and cartesian adjacency do not match"

def test_query_radius():
    
    uniform_coords = np.random.rand(200).reshape(100, 2)
    kdt = KDTree(uniform_coords)
    radius = 0.1
    point = np.array([0.5, 0.5])
    expected = [i for i in range(len(uniform_coords)) if 
                euclidean_distance([uniform_coords[i], point]) <= radius]
    result = [node[0] for node in kdt.query_radius(point, radius)]
    assert sorted(expected) == sorted(result),\
        "KDTree.query_radius returned unexpected results"

def test_breadth_first():
    """
    ensure that nodes are arranged in proper order
    """
    num_coords = 100
    uniform_coords = np.random.rand(num_coords * 2).reshape(num_coords, 2)
    kdt = KDTree(uniform_coords)
    count = 0
    for tree in kdt.breadth_first_trees():
        count += 1
        if tree.left.node is not None and \
           tree.left.node[tree.axis] > tree.node[tree.axis]:
           break
        if tree.right.node is not None and \
           tree.right.node[tree.axis] < tree.node[tree.axis]:
           break
    
    assert count == num_coords,\
        "KDTree nodes are not split correctly"
