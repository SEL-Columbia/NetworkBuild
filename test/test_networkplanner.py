# -*- coding: utf-8 -*-

import os
import json
import networker.io as nio
from np.lib import dataset_store
from networker import networkplanner_runner
from networker.utils import get_rounded_edge_sets

def get_config(config_file):
    """ return cfg as dict from json cfg_file """
    cfg_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), config_file)
    return json.load(open(cfg_path))


def networkplanner_run_compare(cfg, known_results_file, output_dir):
    nwk_p = networkplanner_runner.NetworkPlannerRunner(cfg, output_dir)
    nwk_p.validate()
    nwk_p.run()

    # compare this run against existing results
    test_geo = nio.read_shp_geograph(os.path.join(output_dir,
                            "networks-proposed.shp"),
                            simplify=False)
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


def test_networkplanner_run():
    """ test on randomly generated set of nodes (demand only) """

    run_config = "networkplanner_config_pop100.json"
    results_file = os.path.join("data", "pop_100", "networks-proposed.shp")
    output_dir = os.path.join("data", "tmp")

    cfg = get_config(run_config)

    networkplanner_run_compare(cfg, results_file, output_dir)


def test_networkplanner_leona_run():
    """ test on randomly generated set of nodes (demand only) """

    run_config = "networkplanner_config_leona_net.json"
    results_file = os.path.join("data", "leona", "expected", "networks-proposed.shp")
    output_dir = os.path.join("data", "tmp")

    cfg = get_config(run_config)

    # get config in case we want to modify
    networkplanner_run_compare(cfg, results_file, output_dir)


def test_dataset_store_to_geograph():
    """
    make sure we can go from shp to geograph to dataset_store
    """
    data_dir = os.path.join("data", "pop_100")
    test_geo = nio.read_shp_geograph(os.path.join(data_dir,
                            "networks-proposed.shp"),
                            simplify=False)
    dataset = dataset_store.load(os.path.join(data_dir, "dataset.db"))
    dataset_geo = networkplanner_runner.dataset_store_to_geograph(dataset)

    test_edges = test_geo.get_coord_edge_set()
    dataset_edges = dataset_geo.get_coord_edge_set()

    assert test_edges == dataset_edges, \
        "edges in test do not match dataset results"
