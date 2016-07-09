#!/usr/bin/env python

import sys
import argparse
import logging
import networker
import networker.io as nio

# setup log
logger = logging.getLogger('networker')

logger.info("networker %s (Python %s)" % (
                networker.__version__,
                '.'.join(map(str, sys.version_info[:3]))))

parser = argparse.ArgumentParser(description="Clean network and report it")
parser.add_argument("network_filename",
                    help="network to be cleaned")
parser.add_argument("--output_directory", "-o", \
        default=".", \
        help="directory where network output files will be written")

args = parser.parse_args()

net = nio.read_shp_geograph(args.network_filename, simplify=False)

num_found = 0
for zero_len_edge in net.find_zero_len_edges():
    node0 = zero_len_edge[0]
    node1 = zero_len_edge[1]
    num_found += 1
    logger.warn("found zero length edge ({},{}), with coords ({},{})".\
                format(node0, node1, net.coords[node0], net.coords[node1]))

    net.remove_edge(node0, node1)

logger.info("found {} total zero length edges".format(num_found))

#TODO:  Additional checks?

nio.write_shp(net, args.output_directory)
