# -*- coding: utf-8 -*-

import os
import logging
import json
import jsonschema

import networkx as nx
import numpy as np

from networker.classes.unionfind import UnionFind
from networker.classes.geograph import GeoGraph
from networker.exception import SpatialReferenceMismatchException
import networker.io as nio
import networker.geomath as gm
import networker.algorithms as algo

log = logging.getLogger('networker')

class NetworkerRunner(object):

    """
    class for running minimum spanning forest algorithms on a set of
    spatially referenced nodes

    Attributes:
        config:  dict (potentially nested) of configuration params
            params are documented in networker_config_schema.json
    """

    ALGOS = {'mod_boruvka': algo.mod_boruvka,
             'mod_kruskal': algo.mod_kruskal}

    SCHEMA_FILE = "networker_config_schema.json"

    def __init__(self, config, output_directory="."):
        self.config = config
        self.output_directory = output_directory

    def run(self):
        """
        run a minimum spanning forest algorithm on inputs and write output
        based on configuration
        """

        demand_nodes = load_node_metrics(**self.config['demand_nodes'])
        existing_networks = None
        if 'existing_networks' in self.config:
            existing_networks = load_existing_networks(
                prefix="grid-",
                **self.config['existing_networks'])
            if len(existing_networks.edges()) < 0:
                log.warn("existing network has no edges")
                existing_networks = None
            if not demand_nodes.is_same_srs(existing_networks):
                message = "SpatialReference mismatch:  "\
                          "demand node srs ({}) and existing network srs ({}) "\
                          "do not match".format(demand_nodes.srs, 
                                                existing_networks.srs)
                log.error(message)
                raise SpatialReferenceMismatchException(message)

        network_algorithm = self.config['network_algorithm']

        min_node_count = 0
        single_network = True
        if 'network_parameters' in self.config:
            network_params = self.config['network_parameters']
            min_node_count = network_params.get('minimum_node_count', 0)
            single_network = network_params.get('single_network', True)
            spherical_accuracy = network_params.get('spherical_accuracy', False)

        log.info("building network")
        msf = build_network(demand_nodes,
                            existing=existing_networks,
                            min_node_count=min_node_count,
                            single_network=single_network,
                            network_algorithm=network_algorithm,
                            spherical_accuracy=spherical_accuracy)

        log.info("writing output")
        # now save it
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        nio.write_shp(msf, self.output_directory)
        # write geojson too
        nio.write_geojson(msf, os.path.join(self.output_directory, "proposed.geojson"))

    def validate(self):
        """
        validate configuration
        throws jsonschema Validate exception if invalid
        """

        # load schema and validate it via jsonschema
        schema_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), NetworkerRunner.SCHEMA_FILE)
        schema = json.load(open(schema_path))
        jsonschema.validate(self.config, schema)


def build_network(demand_nodes,
                  existing=None,
                  min_node_count=2,
                  single_network=True,
                  network_algorithm='mod_boruvka',
                  spherical_accuracy=False,
                  one_based=False):
    """
    project demand nodes onto optional existing supply network and
    return the 'optimized' network

    Args:
        demand_nodes:  GeoGraph of demand nodes
        existing:  GeoGraph of existing grid (assumes node ids
            don't conflict with demand_nodes
        min_node_count:  minimum number of nodes allowed in a subgraph
            of the result
        network_algorithm:  Algorithm from ALGOS to run
        spherical_accuracy:  Whether to connect nodes to network on a sphere
        one_based:  Whether result GeoGraph's nodes should be one_based
            (if not, they are 0 based)

    Returns:
        msf: GeoGraph of minimum spanning forest proposed by the chosen
            network algorithm
        existing: The existing grid GeoGraph (None if it doesn't exist)

    """
    geo_graph = subgraphs = rtree = None

    if existing:
        log.info("merging network and nodes")
        geo_graph, subgraphs, rtree = \
            merge_network_and_nodes(existing, demand_nodes,
                                    single_network=single_network,
                                    spherical_accuracy=spherical_accuracy)
    else:
        geo_graph = demand_nodes

    log.info("running {} on {} demand nodes and {} total nodes".format(
              network_algorithm, len(demand_nodes), len(geo_graph)))

    # now run the selected algorithm
    network_algo = NetworkerRunner.ALGOS[network_algorithm]
    result_geo_graph = network_algo(geo_graph, subgraphs=subgraphs,
                                    rtree=rtree)

    filtered_graph = filter_min_node_subnetworks(result_geo_graph,
                                                 min_node_count)

    # map coords back to geograph
    # NOTE:  explicit relabel to int as somewhere in filtering above, some
    # node ids are set to numpy types which screws up comparisons to tuples
    # in write op
    # NOTE:  relabeling nodes in-place here drops node attributes for some
    #   reason so create a copy for now
    def id_label(i):
        id = int(i+1) if one_based else int(i)
        return id

    msf = None
    if filtered_graph:
        coords = {id_label(i): result_geo_graph.coords[i]
                  for i in filtered_graph}
        relabeled = nx.relabel_nodes(filtered_graph, {i: id_label(i)
                                                      for i in filtered_graph},
                                     copy=True)
        msf = GeoGraph(result_geo_graph.srs, coords=coords, data=relabeled)

    log.info("filtered result has {} nodes and {} edges".format(
              len(msf.nodes()), len(msf.edges())))

    return msf


def has_grid_conn(g):
    """
    Whether graph has grid connection
    (assumes "fake" nodes connecting graph to grid have budget == inf)

    Args:
        g (GeoGraph):  a networkplan GeoGraph
    """
    for node in g.nodes(data=True):
        if node[1]['budget'] == np.inf:
            return True

    return False


def filter_min_node_subnetworks(g, min_node_count):
    """
    remove "non-grid connected" connected components from the networkplan
    GeoGraph

    Args:
        g (GeoGraph):  networkplan as GeoGraph
        min_node_count:  min number of nodes in non-grid connected component

    Returns:
        networkx graph with appropriate subnetworks removed

    Note:  If you need a GeoGraph from result, you'll need to convert it
    """

    grid_connected = filter(lambda sub: has_grid_conn(sub),
                            nx.connected_component_subgraphs(g))
    non_grid_connected = filter(lambda sub: not has_grid_conn(sub),
                                nx.connected_component_subgraphs(g))

    def union_all_wrap(graph_list):
        """ handle empty list so that union_all returns empty graph """
        if len(graph_list) == 0:
            return nx.Graph()
        else:
            return nx.union_all(graph_list)

    # now filter out non-grid subnetworks via minimum node count
    filtered_non_grid = union_all_wrap(filter(
        lambda sub: len(sub.node) >= min_node_count, non_grid_connected))

    grid = union_all_wrap(grid_connected)

    # finally, merge back grid_connected
    filtered_graph = nx.union(grid, filtered_non_grid)
    return filtered_graph


def merge_network_and_nodes(network, demand_nodes,
                            single_network=True, spherical_accuracy=False):
    """
    merge the network and nodes GeoGraphs to set up the Graph, UnionFind
    (DisjoinSet), and RTree datastructures for use in network algorithms

    Args:
        network:  graph representing existing network
            (assumes node ids don't conflict with net (demand) nodes)
        demand_nodes:  graph of nodes representing demand
        single_network:  whether subgraphs of network are unioned into
            a single network
        spherical_accuracy:  Whether to connect nodes to network on a sphere

    Returns:
        graph:  graph with demand nodes and their nearest nodes to the
            existing network (i.e. 'fake' nodes)
        subgraphs:  UnionFind datastructure populated with fake nodes and
            associated with the appropriate connected component or the entire
            subgraph (depending on ``single_subgraph`` param)
        rtree:  spatial index populated with the edges from the
            existing network
    """

    # project demand nodes onto network
    rtree = network.get_rtree_index()
    grid_with_fakes = network.project_onto(demand_nodes, rtree_index=rtree,
                                           spherical_accuracy=spherical_accuracy)

    # get only the fake nodes and the associated network edges
    demand_node_set = set(demand_nodes.nodes())
    net_plus_demand = set(network.nodes()).union(demand_node_set)
    fakes = set(grid_with_fakes.nodes()) - net_plus_demand

    def get_fake_edge(node):
        """
        fake node should only have 2 neighbors from the existing network
        that is the nearest edge
        """
        return tuple(set(grid_with_fakes.neighbors(node)) - demand_node_set)

    edge_fakes = [(get_fake_edge(fake), fake) for fake in fakes]

    # Init the DisjointSet
    subgraphs = UnionFind()

    assert len(network.nodes()) > 1, \
        "network must have more than 1 node"

    if single_network:
        # just union all nodes to a single parent
        nodes = network.nodes()
        # add parent
        parent = nodes[0]
        subgraphs.add_component(parent, budget=network.node[parent]['budget'])
        for node in nodes[1:]:
            subgraphs.add_component(node, budget=network.node[node]['budget'])
            # The existing grid nodes are on the grid (so distance is 0)
            subgraphs.union(parent, node, 0)
    else:
        # Build the subnet components
        # Get the network components to init budget centers
        subnets = nx.connected_components(network)

        for sub in subnets:
            # union all nodes to parent of subnet
            sub_list = list(sub)
            parent = sub_list[0]
            subgraphs.add_component(parent,
                                    budget=network.node[parent]['budget'])
            # Merge remaining nodes with component
            for node in sub_list[1:]:
                subgraphs.add_component(node,
                                        budget=network.node[node]['budget'])
                # The existing grid nodes are on the grid (so distance is 0)
                subgraphs.union(parent, node, 0)

    # setup merged graph to be populated with fake nodes
    merged = GeoGraph(demand_nodes.srs, demand_nodes.coords, data=demand_nodes)
    # merge fakes in
    for ((u, v), fake) in edge_fakes:

        # Make sure something wonky isn't going on
        assert(subgraphs[u] == subgraphs[v])

        # Add the fake node to the big net
        # NOTE:  fake nodes always have np.inf budget
        merged.add_node(fake, budget=np.inf)
        merged.coords[fake] = grid_with_fakes.coords[fake]

        # Merge the fake node with the grid subgraph
        subgraphs.add_component(fake, budget=np.inf)
        subgraphs.union(fake, u, 0)

    return merged, subgraphs, rtree


def _clean_geograph(network):
    """
    Checks geograph network for duplicate nodes and removes them

    INTERNAL USE ONLY (modifies network in place)
    """
    num_found = 0
    for zero_len_edge in network.find_zero_len_edges():
        node0 = zero_len_edge[0]
        node1 = zero_len_edge[1]
        num_found += 1
        log.warn("removing zero length edge ({},{}), with coords ({},{})".
                 format(node0, node1, network.coords[node0],
                        network.coords[node1]))

        network.remove_edge(node0, node1)


def load_existing_networks(filename, budget_value=0, prefix=None):
    """
    load existing_networks shp into GeoGraph nodes, edges and assign budget

    Args:
        filename:  existing_networks (geojson or shapefile)
        budget_value:  default budget value for nodes in existing network
        prefix: if not None, relabel node ids with the prefix

    Returns:
        GeoGraph of existing networks with budget attribute
    """

    # simplify is passed in case it's a shapefile
    geo_net = nio.read_geograph(filename, simplify=False)

    nx.set_node_attributes(geo_net, 'budget', {n: budget_value
                                               for n in geo_net.nodes()})

    # determine edge weight/distance function by whether they're geocentric
    # (assuming coordinates are not in 3-space here)

    distance = gm.spherical_distance if geo_net.is_geographic else \
        gm.euclidean_distance

    nx.set_edge_attributes(geo_net, 'weight', {(u, v):
                           distance(map(geo_net.coords.get, [u, v]))
                           for u, v in geo_net.edges()})

    if prefix:
        nx.relabel_nodes(geo_net,
                         {n: prefix + str(n) for n in geo_net.nodes()},
                         copy=False)
        geo_net.coords = {prefix + str(n): c for n, c in
                          geo_net.coords.items()}

    # check and clean
    _clean_geograph(geo_net)

    return geo_net


def load_node_metrics(filename, x_column="X", y_column="Y",
                      budget_column="metric", budget_value=1000):
    """
    load node_metrics csv into GeoGraph (nodes and x,y,budget attributes only)

    Args:
        filename:  nodal metrics file (csv, geojson)
        x_column, y_column:  col names to take x, y from (if csv)
        budget_column:  col to take budget from
        budget_value: default value for nodal budget

    Returns:
        GeoGraph of nodes with only the budget attribute
    """

    # nodes loaded with all attributes
    geo_nodes = nio.read_geograph(filename, x_column, y_column)

    # ensure nodes only have budget attribute
    for node in geo_nodes.nodes_iter():
        if budget_column in geo_nodes.node[node]:
            geo_nodes.node[node] = {'budget':
                                    geo_nodes.node[node][budget_column]}
        else:
            geo_nodes.node[node] = {'budget': budget_value}

    return geo_nodes
