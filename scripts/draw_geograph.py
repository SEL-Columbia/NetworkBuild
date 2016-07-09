#!/usr/bin/env python

import argparse
import networker.utils as utils
import networker.io as nio

parser = argparse.ArgumentParser(description="Render png of graph")
parser.add_argument("network_files", metavar="NETWORK_FILE", type=str, 
                    nargs="+", help="network to be rendered (.shp OR .json)")
parser.add_argument("--node_size", "-n", type=int, default=100,
                    help="size of nodes in pixels")
parser.add_argument("--output_file", "-o", \
        default="network.png", \
        help="file to write image to")

args = parser.parse_args()

nets = []
for net_file in args.network_files:
    if len(net_file) > 5 and net_file[-5:] == '.json':
        nets.append(nio.read_json_geograph(net_file))
    else:
        nets.append(nio.read_shp_geograph(net_file, simplify=False))

# do the rendering and save to file
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
for net in nets:
    utils.draw_geograph(net, node_size=args.node_size)

# don't render axes
cur_axes = plt.gca()
cur_axes.set_frame_on(False)
cur_axes.axes.get_xaxis().set_visible(False)
cur_axes.axes.get_yaxis().set_visible(False)

plt.savefig(args.output_file)
