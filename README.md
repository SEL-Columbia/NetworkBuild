# Networker

Networker contains tools for planning distribution networks from supply and demand data.  

[![Build Status](https://travis-ci.org/SEL-Columbia/networker.svg?branch=master)](https://travis-ci.org/SEL-Columbia/networker)

[![Coverage Status](https://coveralls.io/repos/SEL-Columbia/networker/badge.svg?branch=master)](https://coveralls.io/r/SEL-Columbia/networker?branch=master)

It relies on numpy, networkx and also utilizes several spatial indexes.  

## Overview 

The main interfaces for generating networks are the following classes:
- NetworkerRunner:  Assumes a pre-computed budget for each node.  It follows 
networkx output conventions for spatial data. 
- NetworkPlannerRunner:  Uses existing networkplanner econometric models
to compute nodal budget (aka mvMax).  It's output follows networkplanner 
conventions.  

Included are wrapper scripts for leveraging these classes from the command line.  

## Examples

Once installed with source, navigate to the top level source directory and activate
your environment via `source activate networker`.

### Networker

```
> python scripts/run_networker.py test/networker_config_max100.json -o output
```

Once run, you should see an output directory with edges and nodes shapefiles 
(with supporting files).  This particular example had sufficient nodal "budget"
for the resulting network to be a minimum spanning tree.  

From within python, you can run this via the NetworkerRunner class and work
directly with the resulting GeoGraph (subtype of networkx Graph):

```
from networker import networker_runner
import json

# load demand nodes and generate geograph
cfg = json.load(open("test/networker_config_max100.json"))
nr = networker_runner.NetworkerRunner(cfg)
geograph = nr.build_network()

# plot it
import matplotlib.pyplot as plt
from networker import utils

utils.draw_geograph(geograph)
plt.show()
```

![geograph image](http://i.imgur.com/r7ei1VR.png)

### NetworkPlanner

If you want to leverage the econometric models from networkplanner to compute 
the budget (aka mvMax) values for nodes, use the NetworkPlannerRunner class.  
This will also generate outputs that are consistent with networkplanner.  
A script to run this is included.  Here's a sample run:

```
> python scripts/run_networkplanner.py test/networkplanner_config_leona_net.json -o output
```

Once complete, the output directory should contain a dataset.db and networks-proposed 
shapefiles (along with addtional networkplanner outputs).  

### Other tools

The scripts directory contains various other useful tools for working with geographic nodes and networks.  Below is a sample script demonstrating these tools.  Change the number of nodes to test performance at scale.  

Pass --help to any tool to get detailed help on its parameters.  

```
# generate random demand nodes within bounds [-1, -1], [1, 1] into a csv
cat <(echo "x,y") <(python scripts/generate_random_nodes.py -x -1 1 -y -1 1 5) > demand_nodes.csv

# generate a random minimum spanning tree within the same bounds into a networkx json file
python scripts/generate_random_nodes.py -x -1 1 -y -1 1 30 | python scripts/generate_mst.py > mst.json

# project the nodes onto the nearest segment in the mst 
# (network goes into output/projected.json)
python scripts/project_onto.py -x x -y y -r -j -o output demand_nodes.csv mst.json

# render the network and projections to a png
python scripts/draw_geograph.py -n 50 -o mst.png mst.json output/projected.json
```

Minimum Spanning Tree from above
![geograph image](http://i.imgur.com/eM0HY6w.png)

5 random nodes and shortest edges to the MST as plotted via `draw_geograph.py` above
![geograph image](http://i.imgur.com/LvnWcsx.png)

## Configuration

Both NetworkerRunner and NetworkPlannerRunner take json configuration files
which define their inputs and how they should run.  Annotated schemas for these
configuration files are:

[NetworkRunner Config](https://github.com/SEL-Columbia/networker/blob/master/networker/networker_config_schema.json)

[NetworkPlannerRunner Config](https://github.com/SEL-Columbia/networker/blob/master/networker/networkplanner_config_schema.json)

## Installation

These instructions should work for linux-64 and osx-64 platforms.  

The simplest method of installation is via [conda](http://www.continuum.io/blog/conda).  

Once conda is installed ([guide here](http://docs.continuum.io/anaconda/install.html)), 
setup a new python 2.7 environment via:

```
conda create -n networker python=2.7
```

and source it to subsequently install libraries in that environment:
```
source activate networker
```

To use the NetworkPlannerRunner, you will need the networkplanner library 
installed in this environment:

```
# the geojson library will be installed from the ioos channel so add it
conda install -c ioos -c sel networkplanner-metrics
```

Now install the networker library

```
# on OSX do this (asmeurer is owner for osx version of pyproj)
conda install -c ioos -c sel networker
# on Linux do this
conda install -c ioos -c sel networker
```

On OSX, there appears to be an issue related to [this](https://github.com/conda/conda/issues/308) and [this](https://github.com/ioos/conda-recipes/issues/141) where the libspatialindex library cannot be found.  As a workaround, you can do the following from within your networker environment before running any scripts (this is not ideal...alternative approaches appreciated):  

```
# OSX Only
export LD_LIBRARY_PATH=~/anaconda/envs/networker/lib
# to unset when done do
# unset LD_LIBRARY_PATH
``` 

Similar issues may occur on Ubuntu/Debian.  You can check whether libspatialindex can be found via the following:

```
ldconfig -p 2>/dev/null | grep spatial
```

The issue can be resolved by adding your conda lib directory to the ldconfig cache via:

```
ldconfig ~/anaconda/envs/networker/lib
```

## Development Setup

To setup a development environment, in addition to the above installation steps...

clone this repository or fork to your own repo and then clone:
```
cd <src_dir>
git clone <repo_name>/networker.git 
```

In order to develop/test against the source directory, you'll need to install it's dependencies (and remove networker from the conda environment if it alread exists) and use setup tools to link you source to the conda environment. 
``` 
source activate networker
cd <src_dir>/networker
conda remove networker # if already installed
conda install -c conda-forge --file requirements.txt
# links local source directory to python path
python setup.py develop 
```

At this point, you should be ready to develop against local source.  

See [our style guide](https://github.com/SEL-Columbia/StyleGuides) for how we develop.

## Testing

Once setup for development, run nosetests from the top level repo directory. 
