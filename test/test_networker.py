# -*- coding: utf-8 -*-

import os
import itertools
import json
import numpy as np
import networkx as nx
import networker.io as nio
from networker.exception import SpatialReferenceMismatchException
from networker import networker_runner
from networker import geomath as gm
from networker.algorithms import mod_boruvka
from networker.algorithms import mod_kruskal
from networker.classes.geograph import GeoGraph
from networker.utils import get_rounded_edge_sets

from nose.tools import eq_

def get_config(config_file):
    """ return cfg as dict from json cfg_file """
    cfg_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), config_file)
    return json.load(open(cfg_path))


def networker_run_compare(cfg, known_results_file, output_dir):
    nwk = networker_runner.NetworkerRunner(cfg, output_dir)
    nwk.validate()
    nwk.run()

    # compare this run against existing results
    test_geo = nio.read_shp_geograph(os.path.join(output_dir,
                            "edges.shp"), simplify=False)
    known_geo = nio.read_shp_geograph(known_results_file, simplify=False)

    # compare sets of edges
    test_edges = get_rounded_edge_sets(test_geo, round_precision=8)
    known_edges = get_rounded_edge_sets(known_geo, round_precision=8)

    assert test_edges == known_edges, \
        "edges in test do not match known results"

    # This is redundant (but leave it here as an example as it's more lenient
    # than test above)
    # assert nx.is_isomorphic(test_geo, known_geo), \
    #    "test result graph is not isomorphic to known result graph"


def test_networker_run():
    """ test on randomly generated set of nodes (demand only) """

    run_config = "networker_config_med100.json"
    results_file = os.path.join("data", "med_100", "networks-proposed.shp")
    output_dir = os.path.join("data", "tmp")

    cfg = get_config(run_config)
    networker_run_compare(cfg, results_file, output_dir)


def test_networker_leona_run():
    """ test previously generated results match """

    run_config = "networker_config_leona_net.json"
    # TODO:  Determine why this fails with networkplanner_results_file
    # results_file = "data/leona/expected/networks-proposed.shp"
    results_file = os.path.join("data", "leona", "expected", "edges.shp")
    output_dir = os.path.join("data", "tmp")

    cfg = get_config(run_config)
    networker_run_compare(cfg, results_file, output_dir)

def test_networker_leona_spherical_run():
    """ test previously generated results match for spherical treatment """

    run_config = "networker_config_leona_net_spherical.json"
    # TODO:  Determine why this fails with networkplanner_results_file
    # results_file = "data/leona/expected/networks-proposed.shp"
    results_file = "data/leona/expected_spherical/edges.shp"
    output_dir = "data/tmp"

    cfg = get_config(run_config)
    networker_run_compare(cfg, results_file, output_dir)



def test_srs_mismatch():
    """ ensure that mismatched demand/network srs fails"""
    run_config = "networker_config_leona_mixed_srs.json"
    cfg = get_config(run_config)
    output_dir = "data/tmp"
    nwk = networker_runner.NetworkerRunner(cfg, output_dir)
    try:
        nwk.run()
    except Exception as e:
        assert isinstance(e, SpatialReferenceMismatchException),\
               "Exception was {}, should be SpatialReferenceMismatchException".format(e)
    else:
        assert False, "SpatialReferenceMismatchException expected"


def random_settlements(n):

    coords = np.random.uniform(size=(n, 2))

    # get all perm's of points (repetitions are ok here)
    points_left = np.tile(coords, (len(coords), 1))
    points_right = np.repeat(coords, len(coords), axis=0)
    point_pairs = np.concatenate((points_left[:,np.newaxis],
                                  points_right[:,np.newaxis]), axis=1)
    all_dists = gm.spherical_distance_haversine(point_pairs)

    full_dist_matrix = all_dists.reshape(len(coords), len(coords))
    zero_indices = (np.array(range(len(coords))) * (len(coords) + 1))
    non_zero_dists = np.delete(all_dists, zero_indices).\
        reshape((len(coords), len(coords) - 1))

    # find all minimum distances
    # apply min over ranges of the dist array
    min_dists = np.min(non_zero_dists, axis=1)

    # assign same median budget to all nodes
    # outside a really degenerate case (all edges in line in shortest
    # distance order...)
    # this should ensure some "dead" nodes
    budget_vals = np.repeat(np.median(min_dists), len(coords))

    # build graph
    graph = GeoGraph(gm.PROJ4_FLAT_EARTH, dict(enumerate(coords)))
    nx.set_node_attributes(graph, 'budget', dict(enumerate(budget_vals)))

    return graph, full_dist_matrix


def test_msf_components():

    grid, dist_matrix = random_settlements(500)

    msf = mod_boruvka(grid)

    msf_subgraph = lambda components: nx.subgraph(msf, components)
    component_graphs = map(msf_subgraph, nx.connected_components(msf))

    def full_graph(g):
        new_graph = nx.Graph()
        new_graph.add_nodes_from(g.nodes(data=True))
        if len(g.nodes()) < 2:
            return new_graph

        new_graph.add_weighted_edges_from([(u, v, dist_matrix[u][v])
            for u, v in itertools.product(g.nodes(), g.nodes())
            if u != v])
        return new_graph

    full_graphs = map(full_graph, component_graphs)
    mst_graphs = map(nx.mst.minimum_spanning_tree, full_graphs)

    diff_component_mst = []
    for i in range(len(component_graphs)):
        c_sets = set([frozenset(e) for e in component_graphs[i].edges()])
        mst_sets = set([frozenset(e) for e in mst_graphs[i].edges()])
        if not c_sets == mst_sets:
            diff_component_mst.append(i)

    assert len(diff_component_mst) == 0, str(len(diff_component_mst)) + \
        " components are not MSTs"


def nodes_plus_existing_grid():
    """
    return net plus existing grid with certain properties for testing
    nodes by id (budget in parens)

           1 (3)
            \
             \
               0 (2)
               |       2 (5)
               |       |
     +-+-+-+-+-+-+-+-+-+-+  <-- existing grid

    """

    # setup grid
    grid_coords = np.array([[-5.0, 0.0], [5.0, 0.0]])
    grid = GeoGraph(gm.PROJ4_FLAT_EARTH, {'grid-' + str(n): c for n, c in
                    enumerate(grid_coords)})
    nx.set_node_attributes(grid, 'budget', {n: 0 for n in grid.nodes()})
    grid.add_edges_from([('grid-0', 'grid-1')])

    # setup input nodes
    node_coords = np.array([[0.0, 2.0], [-1.0, 4.0], [4.0, 1.0]])
    nodes = GeoGraph(gm.PROJ4_FLAT_EARTH, dict(enumerate(node_coords)))
    budget_values = [2, 3, 5]
    nx.set_node_attributes(nodes, 'budget', dict(enumerate(budget_values)))

    # setup resulting edges when creating msf through the sequence of nodes
    # Note: Fake nodes integer label begins at the total number of nodes + 1
    # Hence why the fake node in the test is incremented by one on each
    # iteration
    edges_at_iteration = [[(0, 1)],  # 0 connects to fake_node
                          [(0, 2)],  # 0, 1 can't connect
                          [(0, 3), (2, 5), (1, 0)]] # 2 connects grid

    return grid, nodes, edges_at_iteration


def run_network_algo_iteratively(algorithm):
    grid, nodes, edges_at_iteration = nodes_plus_existing_grid()

    for n, _ in enumerate(nodes.node):
        sub = nodes.subgraph(range(n+1))
        sub.coords = {i: nodes.coords[i] for i in range(n+1)}
        G, DS, R = networker_runner.merge_network_and_nodes(grid, sub)
        msf = algorithm(G, DS, R)
        msf_sets = set([frozenset(e) for e in msf.edges()])
        iter_edge_set = set([frozenset(e) for e in edges_at_iteration[n]])
        eq_(msf_sets, iter_edge_set)


def test_algos_iteratively():

    run_network_algo_iteratively(mod_boruvka)
    run_network_algo_iteratively(mod_kruskal)


def simple_nodes_disjoint_grid():
    """
    return disjoint net plus nodes with fakes
    fakes are associated with disjoint subnets

    nodes by id (budget in parens)


           (5) 0-------1 (5)
               |       |
             +-+-+   +-+-+  <-- disjoint existing grid

    Useful for testing treating existing grid as single grid
    vs disjoint

    """
    # setup grid
    grid_coords = np.array([[-1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    grid = GeoGraph(gm.PROJ4_FLAT_EARTH, {'grid-' + str(n): c for n, c in
                    enumerate(grid_coords)})
    nx.set_node_attributes(grid, 'budget', {n: 0 for n in grid.nodes()})
    grid.add_edges_from([('grid-0', 'grid-1'), ('grid-2', 'grid-3')])

    # setup input nodes
    node_coords = np.array([[0.0, 1.0], [4.0, 1.0]])
    nodes = GeoGraph(gm.PROJ4_FLAT_EARTH, dict(enumerate(node_coords)))
    budget_values = [5, 5]
    nx.set_node_attributes(nodes, 'budget', dict(enumerate(budget_values)))

    fakes = [2, 3]
    return grid, nodes, fakes


def test_merge_network_and_nodes():

    grid, nodes, fakes = simple_nodes_disjoint_grid()
    # test disjoint merge
    G, DS, R = networker_runner.merge_network_and_nodes(grid, nodes,
                                                        single_network=False)
    fake_parents = [DS[fake] for fake in fakes]
    assert len(np.unique(fake_parents)) == len(fake_parents), \
        "fake nodes should be associated with distinct parents"

    msf = mod_boruvka(G, DS, R)
    assert msf.has_edge(0, 1), "edge between nodes 0, 1 should exist"

    G, DS, R = networker_runner.merge_network_and_nodes(grid, nodes,
                                                        single_network=True)
    fake_parents = [DS[fake] for fake in fakes]
    assert len(np.unique(fake_parents)) == 1, \
        "fake nodes should be associated with single parent"

    msf = mod_boruvka(G, DS, R)
    assert not msf.has_edge(0, 1), "edge between nodes 0, 1 should not exist"

def grid_and_non_grid():
    """
    return networkplan GeoGraph with grid and non-grid components

               0       2
               |       |
             +-6-+     1   3-4-5
              (inf)

    where node 3 is a fake node connecting node 0 to the grid

    """
    node_coords = np.array([[0.0, 1.0],
                            [4.0, 0.0],
                            [4.0, 1.0],
                            [5.0, 0.0],
                            [6.0, 0.0],
                            [7.0, 0.0],
                            [0.0, 0.0]])

    grid = GeoGraph(gm.PROJ4_FLAT_EARTH, dict(enumerate(node_coords)))
    budget_values = [5, 5, 5, 5, 5, 5, np.inf]
    nx.set_node_attributes(grid, 'budget', dict(enumerate(budget_values)))
    grid.add_edges_from([(0, 6), (1, 2), (3, 4), (4, 5)])

    return grid


def test_min_node_filter():
    """
    Test whether min node filter is applied properly
    """

    grid = grid_and_non_grid()
    min_node_count = 3

    grid_connected = filter(lambda sub: networker_runner.has_grid_conn(sub),
                            nx.connected_component_subgraphs(grid))

    non_grid_connected = filter(lambda sub:
                                not networker_runner.has_grid_conn(sub),
                                nx.connected_component_subgraphs(grid))

    filtered = networker_runner.filter_min_node_subnetworks(grid, min_node_count)

    grid_filtered = filter(lambda sub: networker_runner.has_grid_conn(sub),
                         nx.connected_component_subgraphs(filtered))

    non_grid_filtered = filter(lambda sub:
                               not networker_runner.has_grid_conn(sub),
                               nx.connected_component_subgraphs(filtered))

    assert len(grid_connected) == len(grid_filtered),\
           "number grid connected subnets should not change"

    assert min([len(g) for g in non_grid_filtered]) >= min_node_count,\
           "non-grid networks should have >= {} nodes".format(min_node_count)

    # make sure it works without existing grid
    filtered_non_grid = networker_runner.filter_min_node_subnetworks(
        nx.union_all(non_grid_connected), min_node_count)

    subnet_lens = [len(g) for g in
                   nx.connected_component_subgraphs(filtered_non_grid)]
    assert min(subnet_lens) >= min_node_count,\
           "non-grid networks should have >= {} nodes".format(min_node_count)
