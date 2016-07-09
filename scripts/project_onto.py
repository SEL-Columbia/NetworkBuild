#!/usr/bin/env python

import argparse
import sys
import os
import logging
import networkx as nx
import networker
import networker.io as nio
from networker.classes.geograph import GeoGraph

# setup log
logger = logging.getLogger('networker')

logger.info("networker %s (Python %s)" % (
                networker.__version__,
                '.'.join(map(str, sys.version_info[:3]))))

parser = argparse.ArgumentParser(description="Project nodes to nearest point on network")
parser.add_argument("node_filename",
                    help="csv of nodes to be joined to network")
parser.add_argument("network_filename",
                    help="network nodes will be joined to (.shp OR .json)")
parser.add_argument("--x_column", "-x", \
        default="x", \
        help="column name for x value in node csv")
parser.add_argument("--y_column", "-y", \
        default="y", \
        help="column name for x value in node csv")
parser.add_argument("--json", "-j", \
        dest="write_json", action="store_true", default=False,
        help="write json to projected.json")
parser.add_argument("--rtree", "-r", \
        dest="rtree", action="store_true", default=False,
        help="use rtree for segment lookup")
parser.add_argument("--spherical_accuracy", "-s", \
        dest="spherical_accuracy", action="store_true", default=False,
        help="connect nodes to edges as though on a sphere (ignored unprojected inputs)")
parser.add_argument("--output_directory", "-o", \
        default=".", \
        help="directory where all output files will be written")
args = parser.parse_args()

nodes = nio.read_csv_geograph(args.node_filename,
                              args.x_column,
                              args.y_column)

net = None
if len(args.network_filename) > 5 and args.network_filename[-5:] == '.json':
    net = nio.read_json_geograph(args.network_filename)
else:
    net = nio.read_shp_geograph(args.network_filename, simplify=False)

# relabel nodes/coords so that merge can work
prefix = "net-"
nx.relabel_nodes(net,
    {n: prefix + str(n) for n in net.nodes()}, copy=False)
net.coords = {prefix + str(n): c for n, c in
    net.coords.items()}

def project_helper(use_rtree, spherical_accuracy):
    if use_rtree:
        logger.info("building rtree...")
        rtree = net.get_rtree_index()
        logger.info("projecting nodes...")
        return net.project_onto(nodes, rtree_index=rtree, 
                                spherical_accuracy=spherical_accuracy)
    else:
        logger.info("projecting nodes...")
        return net.project_onto(nodes, spherical_accuracy=spherical_accuracy)

# project_onto returns geograph with projected nodes PLUS
# the network edges they were projected onto
projected = project_helper(args.rtree, args.spherical_accuracy)

# get only the projected edges
# construct geograph of only projected edges
projected_edges = GeoGraph(srs=projected.srs)
for node in nodes:
    edge = (node, projected.neighbors(node)[0])
    projected_edges.add_edge(*edge)
    projected_edges.coords[edge[0]] = projected.coords[edge[0]]
    projected_edges.coords[edge[1]] = projected.coords[edge[1]]

if(args.write_json):
    nio.write_json(projected_edges, 
                   open(os.path.join(args.output_directory, 'projected.json'), 'w'))
else:
    nio.write_shp(projected_edges, args.output_directory)
