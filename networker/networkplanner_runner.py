# -*- coding: utf-8 -*-

import json
import jsonschema
import os
import logging
import networkx as nx
import numpy as np

from np.lib import dataset_store, metric, variable_store as VS

import networker.geomath as gm
from networker.classes.geograph import GeoGraph
from networker import networker_runner
import networker.io as nio

log = logging.getLogger('networker')


class NetworkPlannerRunner(object):

    """
    class for running combined metric computation and minimum spanning forest
    on spatially referenced nodes

    i.e. This is a wrapper for running the NetworkPlanner process with it's
    network algorithm replaced by networker's algorithm

    Attributes:
        config:  dict (potentially nested) of configuration params
            params are documented in networkplanner_config_schema.json

    """

    SCHEMA_FILE = "networkplanner_config_schema.json"

    def __init__(self, config, output_directory="."):
        self.config = config
        self.output_directory = output_directory

    def run(self):
        """
        run metrics calculations and then  minimum spanning forest algorithm
        on inputs and write output based on configuration
        """

        # make output dir if not exists
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        metric_config = json.load(open(
            self.config['metric_model_parameters_file']))
        # read in metrics and setup dataset_store
        demand_proj = nio.read_csv_projection(self.config['demand_nodes_file'])
        target_path = os.path.join(self.output_directory, "dataset.db")
        self.store = dataset_store.create(target_path,
                                          self.config['demand_nodes_file'])

        log.info("running metric model {}".format(self.config['metric_model']))
        metric_model = metric.getModel(self.config['metric_model'])
        metric_vbobs = self._run_metric_model(metric_model, metric_config)
        demand_nodes = self._get_demand_nodes(input_proj=demand_proj)

        existing_networks = None

        if 'existing_networks' in self.config:
            existing_networks = networker_runner.load_existing_networks(
                prefix="grid-",
                **self.config['existing_networks'])

        network_algorithm = self.config['network_algorithm']

        min_node_count = 0
        single_network = True
        if 'network_parameters' in self.config:
            network_params = self.config['network_parameters']
            min_node_count = network_params.get('minimum_node_count', 0)
            single_network = network_params.get('single_network', True)

        header_type = VS.HEADER_TYPE_SECTION_OPTION
        if 'output_parameters' in self.config:
            output_params = self.config['output_parameters']
            header_type = output_params.get('header_type',
                                            VS.HEADER_TYPE_SECTION_OPTION)

        log.info("building network")
        msf = networker_runner.build_network(
            demand_nodes,
            existing=existing_networks,
            min_node_count=min_node_count,
            single_network=single_network,
            network_algorithm=network_algorithm,
            one_based=True)

        log.info("writing output")
        self._store_networks(msf, existing_networks)
        metric_vbobs = self._update_metrics(metric_model, metric_vbobs)
        self._save_output(metric_vbobs, metric_config, metric_model,
                          header_type=header_type)
        # at this point, dataset_store has all node/network values
        # so get full geograph and write as geojson
        full_geograph = dataset_store_to_geograph(self.store)
        nio.write_geojson(full_geograph, os.path.join(self.output_directory, "networks-proposed.geojson"))

    def _run_metric_model(self, metric_model, metric_config):

        """
        Run the 'metrics' or 'demand' process of networkplanner
        """

        metric_value_by_option_by_section = self.store.applyMetric(
            metric_model, metric_config)

        return metric_value_by_option_by_section

    def _get_default_proj4(self, coords):
        """
        in case there's no proj, guess
        """
        input_proj = gm.PROJ4_FLAT_EARTH
        if gm.is_in_lon_lat(coords):
            input_proj = gm.PROJ4_LATLONG

        return input_proj

    def _get_demand_nodes(self, input_proj=None):
        """
        Converts the dataset_store metrics records to a GeoGraph of nodes
        (prereq:  _run_metric_model to populate store)

        Args:
            input_proj:  projection of demand node coordinates

        Returns:
            GeoGraph:  demand nodes as GeoGraph
        """

        coords = [node.getCommonCoordinates() for node in
                  self.store.cycleNodes()]

        # set default projection
        if not input_proj:
            input_proj = self._get_default_proj4(coords)

        # NOTE:  Although dataset_store nodes id sequence starts at 1
        # leave the GeoGraph ids 0 based because there are places in the
        # network algorithm that assume 0 based coords
        # This will be realigned later
        coords_dict = {i: coord for i, coord in enumerate(coords)}
        budget_dict = {i: node.metric for i, node in
                       enumerate(self.store.cycleNodes())}

        geo_nodes = GeoGraph(input_proj, coords_dict)
        nx.set_node_attributes(geo_nodes, 'budget', budget_dict)
        return geo_nodes

    def _store_networks(self, msf, existing=None):

        # Add the existing grid to the dataset_store
        if existing:
            dataset_subnet = dataset_store.Subnet()
            for u, v in existing.edges():
                segment = dataset_store.Segment(u, v)
                segment.subnet_id = dataset_subnet.id
                segment.is_existing = True
                segment.weight = existing[u][v]['weight']
                self.store.session.add(segment)

        # Translate the NetworkX Graph to dataset_store objects
        if msf:
            # add the fake nodes in the msf to the store
            for fake in [n for n in msf if msf.node[n]['budget'] == np.inf]:
                dataset_node = self.store.addNode(msf.coords[fake],
                                                  is_fake=True)
                dataset_node.id = fake
                self.store.session.add(dataset_node)
                self.store.session.commit()

            for subgraph in nx.connected_component_subgraphs(msf):
                # Initialize the subgraph in the store
                dataset_subnet = dataset_store.Subnet()
                self.store.session.add(dataset_subnet)
                self.store.session.commit()

                # Extend the dstore subnet with its segments
                for u, v, data in subgraph.edges(data=True):
                    edge = u, v

                    # Add the edge to the subnet
                    segment = dataset_store.Segment(*edge)
                    segment.subnet_id = dataset_subnet.id
                    segment.is_existing = False
                    segment.weight = data['weight']
                    self.store.session.add(segment)

        # Commit changes
        self.store.session.commit()

    def _update_metrics(self, metric_model, metric_value_by_option_by_section):
        """
        calculate and return summary metrics after network has been
        determined and stored
        """
        return self.store.updateMetric(metric_model,
                                       metric_value_by_option_by_section)

    def _save_output(self, metric_value_by_option_by_section, metric_config,
                     metric_model, header_type=VS.HEADER_TYPE_SECTION_OPTION):

        output_directory = self.output_directory
        metric.saveMetricsConfigurationCSV(os.path.join(output_directory,
                                                        'metrics-job-input'),
                                           metric_config)
        metric.saveMetricsCSV(os.path.join(output_directory,
                                           'metrics-global'),
                              metric_model,
                              metric_value_by_option_by_section)
        self.store.saveMetricsCSV(os.path.join(output_directory,
                                               'metrics-local'),
                                  metric_model,
                                  header_type)
        # underlying library can't handle unicode strings so cast via str
        self.store.saveSegmentsSHP(os.path.join(str(output_directory),
                                                'networks-proposed'),
                                   is_existing=False)

    def validate(self):
        """
        validate configuration
        throws jsonschema Validate exception if invalid
        """

        # load schema and validate it via jsonschema
        schema_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), NetworkPlannerRunner.SCHEMA_FILE)
        schema = json.load(open(schema_path))
        jsonschema.validate(self.config, schema)


def dataset_store_to_geograph(dataset_store):
    """
    convenience function for converting a network stored in a dataset_store
    into a GeoGraph

    Args:
        dataset_store containing a network

    Returns:
        GeoGraph representation of dataset_store network

    TODO: determine projection from dataset_store?
    """


    all_nodes = list(dataset_store.cycleNodes()) + \
        list(dataset_store.cycleNodes(isFake=True))
    
    # nodes in output GeoGraph are id'd from 0 to n (via enumerate)
    np_to_nx_id = {node.id: i for i, node in enumerate(all_nodes)}

    coords = [node.getCommonCoordinates() for node in all_nodes]
    coords_dict = dict(enumerate(coords))

    G = GeoGraph(coords=coords_dict)

    # only set population, system and budget for now
    # TODO:  Do we need all from the output?
    for i, node in enumerate(all_nodes):
        if not node.is_fake:
            properties = {
                'budget': node.metric,
                'population': node.output['demographics']['population count'],
                'system': node.output['metric']['system']
            }
            G.node[i] = properties
     

    def seg_to_nx_ids(seg):
        """
        Return the networkx segment ids
        """
        return (np_to_nx_id[seg.node1_id],
                np_to_nx_id[seg.node2_id])

    edges = [seg_to_nx_ids(s) for s in
             dataset_store.cycleSegments(is_existing=False)]
    edge_weights = {seg_to_nx_ids(s): s.weight for s in
                    dataset_store.cycleSegments(is_existing=False)}
    edge_is_existing = {seg_to_nx_ids(s): s.is_existing for s in
                        dataset_store.cycleSegments(is_existing=False)}
    edge_subnet_id = {seg_to_nx_ids(s): s.subnet_id for s in
                      dataset_store.cycleSegments(is_existing=False)}
    G.add_edges_from(edges)
    nx.set_edge_attributes(G, 'weight', edge_weights)
    nx.set_edge_attributes(G, 'is_existing', edge_is_existing)
    nx.set_edge_attributes(G, 'subnet_id', edge_subnet_id)

    return G
