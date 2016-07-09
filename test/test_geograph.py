# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np

from networker.classes.geograph import GeoGraph
import networker.geomath as gm
import networker.utils as utils


# fixtures
def network_nodes_projections():
    """
    Create a network and node graph to be merged, and
    the expected result of project_onto for testing

    rough picture of this test

              4
             /
            /
           /
          /
         /
        /      2
       /       |
      /     6  | 8
     /   5 0---1
    3            7

    nodes 5,6,7,8 should be projected onto graph path (0,1,2)
    graph path (3,4) is meant to test whether a long segment whose
    bbox overlaps all nodes interferes with the projection
    (it had in the past)
    """

    net_coords = [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0],
                  [-6.0, -1.0], [6.0, 11.0]]
    net_edges = [(0, 1), (1, 2), (3, 4)]
    node_coords = {5: [-1.0, 0.0], 6: [1.0, 1.0],
                   7: [4.0, -1.0], 8: [4.0, 1.0]}

    projected_coords = {5: [0.0, 0.0], 6: [1.0, 0.0],
                        7: [3.0, 0.0], 8: [3.0, 1.0]}

    g_net = GeoGraph(gm.PROJ4_FLAT_EARTH, dict(enumerate(net_coords)), data=net_edges)
    g_nodes = GeoGraph(gm.PROJ4_FLAT_EARTH, node_coords)

    return g_net, g_nodes, projected_coords


def test_project_xyz_vs_geo():
    """
    ensure that project_onto works the same with xyz vs lat_lon points
    """
    net_coords = {'grid-1': [0.0, 0.0], 
                  'grid-2': [10.0, 10.0]}

    net_edges = [('grid-1', 'grid-2')]
    
    node_coords = {0: [5.0, 5.0]}

    g_net = GeoGraph(gm.PROJ4_LATLONG, net_coords, data=net_edges)
    g_nodes = GeoGraph(gm.PROJ4_LATLONG, node_coords)
    
    g_project = g_net.project_onto(g_nodes, spherical_accuracy=True)

    g_net_xyz = GeoGraph(gm.PROJ4_LATLONG, 
                         g_net.lon_lat_to_cartesian_coords(),
                         data=net_edges)

    g_nodes_xyz = GeoGraph(gm.PROJ4_LATLONG, 
                           g_nodes.lon_lat_to_cartesian_coords())

    g_project_xyz = g_net_xyz.project_onto(g_nodes_xyz, spherical_accuracy=True)

    g_project_coords_ll = g_project_xyz.cartesian_to_lon_lat()

    def round_coords(coords, round_precision=8):
        return {i: tuple(map(lambda c: round(c, round_precision), coord))
               for i, coord in coords.items()}

    assert(round_coords(g_project_coords_ll) == round_coords(g_project.coords))


def test_project_onto():
    """
    test the project_onto function of geograph
    """

    net, nodes, projections = network_nodes_projections()

    new_net = net.project_onto(nodes)

    # check that the projected coords are in the new net
    neighbor_coords = {node: [list(new_net.coords[neigh])
        for neigh in new_net.neighbors(node)]
        for node in projections.keys()}

    # need to 'cast' to list in case coords are numpy arrays
    tests = [projections[p] in neighbor_coords[p] for p in projections.keys()]
    assert all(tests), \
        "expected projection does not match actual"

    # ensure that the rtree based projection results in the same network
    rtree = net.get_rtree_index()
    new_net_rt = net.project_onto(nodes, rtree_index=rtree)

    assert nx.is_isomorphic(new_net, new_net_rt) and \
        [list(c) for c in new_net.coords.values()] == \
        [list(c) for c in new_net_rt.coords.values()], \
        "project_onto with and without rtree inputs don't match"


def test_connected_graph():
    """
    test get_connected_weighted_graph function of geograph
    """
    g, _, _ = network_nodes_projections()

    g_conn = g.get_connected_weighted_graph()

    nodes = g.nodes()
    edges = set(zip(np.tile(nodes, len(nodes)), np.repeat(nodes, len(nodes))))
    diagonal = set(zip(nodes, nodes))
    edges = edges - diagonal
    # order does NOT matter
    edges = set([frozenset(edge) for edge in edges])
    def dist_for_edge(edge):
        return gm.euclidean_distance([g.coords[edge[0]], g.coords[edge[1]]])

    weights = {edge: dist_for_edge(edge) for edge in \
                [tuple(set_edge) for set_edge in edges]}

    g_conn_weights = {(e[0],e[1]): e[2]['weight']
                        for e in g_conn.edges(data=True)}

    assert weights == g_conn_weights,\
        "fully connected edges/weights are not correct"

def test_merge_nearby_nodes():
    """                       
         --2                    2
        /                      /
      --                      /
     /                       /
    1                       /
     \                     /
      0-----------5   =>  0------------5
     / 
    3 
    |
    4

    """

    coords = [[ 0,  0],
              [-1,  1],
              [ 2,  6],
              [-1, -1],
              [-1, -2],
              [ 5,  0]]

    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5)]

    geo = GeoGraph(coords=dict(enumerate(coords)), data=edges)
    geo.merge_nearby_nodes(radius=2.0)
    assert geo.edges() == [(0, 2), (0, 5)],\
        "nodes were not merged correctly"

