# -*- coding: utf-8 -*-

from decorator import decorator
import ogr
import osr
import json
import numpy as np
import pandas as pd
import networkx as nx
import networker.geomath as gm
from networker.exception import NetworkerException, SpatialReferenceInvalidException
from networkx.readwrite import json_graph
from networkx.utils.decorators import open_file
from networker.utils import nested_dict_getter
from networker.classes.geograph import GeoGraph
import warnings
import os
import re

"""
Package for reading/writing networkx based GeoGraphs
Note:  these wrap existing networkx functions for custom behavior
"""


def open_shp_read(path_arg):
    """
    read decorator specific to shapefiles

    for some consistency with networkx.utils.decorators.open_file
    (see that implementation for well-documented explanations)

    Example:
    @open_read_shp(0)
    read_proj(shp_file):
        lyr = shp_file.GetLayer(0)
        return lyr.GetSpatialRef().ExportToProj4()
        
    """
    @decorator
    def _open_shp_read(func, *args, **kwargs):
        try:
            from osgeo import ogr
        except ImportError:
            raise ImportError("reading shapefiles requires ogr: http://www.gdal.org/")
     
        # path_arg must be a positional argument
        path = args[path_arg]
        if isinstance(path, basestring):
            # NOTE:  if path is unicode ogr io doesn't work for some reason
            path = path.encode('ascii', 'ignore')
            shp = ogr.Open(path)
            # replace path_arg with shp
            args = args[:path_arg] + (shp,) + args[path_arg + 1:]
        elif not isinstance(path, ogr.DataSource):
            msg = "path argument {} must be string or ogr.DataSource"
            raise NetworkerException(msg.format(path_arg))            

        return func(*args, **kwargs)
        
    return _open_shp_read


def edges_from_line(geom, attrs, simplify=True, geom_attrs=True):
    """
    Generate edges for each line in geom
    Written as a helper for read_shp

    Parameters
    ----------

    geom:  ogr line geometry
        To be converted into an edge or edges

    attrs:  dict
        Attributes to be associated with all geoms

    simplify:  bool
        If ``True``, simplify the line as in read_shp

    geom_attrs:  bool
        If ``True``, add geom attributes to edge as in read_shp


    Yields:
        edges:  generator of edges
            each edge is a tuple of form
            (node1_coord, node2_coord, attribute_dict)
            suitable for expanding into a networkx Graph add_edge call
    """
    if geom.GetGeometryType() == ogr.wkbLineString:
        if simplify:
            edge_attrs = attrs.copy()
            last = geom.GetPointCount() - 1
            if geom_attrs:
                edge_attrs["Wkb"] = geom.ExportToWkb()
                edge_attrs["Wkt"] = geom.ExportToWkt()
                edge_attrs["Json"] = geom.ExportToJson()
            yield (geom.GetPoint_2D(0), geom.GetPoint_2D(last), edge_attrs)
        else:
            for i in range(0, geom.GetPointCount() - 1):
                pt1 = geom.GetPoint_2D(i)
                pt2 = geom.GetPoint_2D(i + 1)
                edge_attrs = attrs.copy()
                if geom_attrs:
                    segment = ogr.Geometry(ogr.wkbLineString)
                    segment.AddPoint_2D(pt1[0], pt1[1])
                    segment.AddPoint_2D(pt2[0], pt2[1])
                    edge_attrs["Wkb"] = segment.ExportToWkb()
                    edge_attrs["Wkt"] = segment.ExportToWkt()
                    edge_attrs["Json"] = segment.ExportToJson()

                yield (pt1, pt2, edge_attrs)

    elif geom.GetGeometryType() == ogr.wkbMultiLineString:
        for i in range(geom.GetGeometryCount()):
            geom_i = geom.GetGeometryRef(i)
            for edge in edges_from_line(geom_i, attrs, simplify, geom_attrs):
                yield edge


@open_shp_read(0)
def read_shp_networkx_graph(shp, simplify=True, geom_attrs=True):
    """Generates a networkx.DiGraph from shapefiles. Point geometries are
    translated into nodes, lines into edges. Coordinate tuples are used as
    keys. Attributes are preserved, line geometries are simplified into start
    and end coordinates. Accepts a single shapefile or directory of many
    shapefiles.

    "The Esri Shapefile or simply a shapefile is a popular geospatial vector
    data format for geographic information systems software [1]_."

    Parameters
    ----------
    shp: string or ogr.DataSource
        string path name to shapefile or pre-opened ogr.DataSource 

    simplify:  bool
        If ``True``, simplify line geometries to start and end coordinates.
        If ``False``, and line feature geometry has multiple segments, the
        non-geometric attributes for that feature will be repeated for each
        edge comprising that feature.

    geom_attrs: bool
        If ``True``, include the Wkb, Wkt and Json geometry attributes with
        each edge.

        NOTE:  if these attributes are available, write_shp will use them
        to write the geometry.  If nodes store the underlying coordinates for
        the edge geometry as well (as they do when they are read via
        this method) and they change, your geomety will be out of sync.


    Returns
    -------
    G : NetworkX graph

    Examples
    --------
    >>> G=nx.read_shp_networkx_graph('test.shp') # doctest: +SKIP

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Shapefile
    """

    net = nx.DiGraph()
    for lyr in shp:
        fields = [x.GetName() for x in lyr.schema]
        for f in lyr:
            flddata = [f.GetField(f.GetFieldIndex(x)) for x in fields]
            g = f.geometry()
            attributes = dict(zip(fields, flddata))
            attributes["ShpName"] = lyr.GetName()
            # Note:  Using layer level geometry type
            if g.GetGeometryType() == ogr.wkbPoint:
                net.add_node((g.GetPoint_2D(0)), attributes)
            elif g.GetGeometryType() in (ogr.wkbLineString,
                                         ogr.wkbMultiLineString):
                for edge in edges_from_line(g, attributes, simplify,
                                            geom_attrs):
                    net.add_edge(*edge)
            else:
                raise TypeError("GeometryType {} not supported".
                                  format(g.GetGeometryType()))

    return net

@open_shp_read(0)
def read_shp_geograph(shp, simplify=True):
    """ 
    loads a shapefile into a networkx based GeoGraph object

    Args:
        shp: string or ogr.DataSource
            string path name to shapefile or pre-opened ogr.DataSource 


        simplify:  Only retain start/end nodes of multi-segment lines

    Returns:
        geograph:  GeoGraph

    """

    # Note:  There's already a nx.read_shp which we've contributed to and
    #        may want to just adopt over our own read_shp_networkx_graph
    g = read_shp_networkx_graph(shp, simplify=simplify, geom_attrs=False)
    coords = dict(enumerate(g.nodes()))

    # needed for SRS
    layer = shp.GetLayer()
    spatial_ref = layer.GetSpatialRef()
    proj4 = None
    if not spatial_ref:
        if gm.is_in_lon_lat(coords):
            proj4 = gm.PROJ4_LATLONG
        else:
            warnings.warn("Spatial Reference could not be set for {}".
                          format(shp.GetName()))

    else:
        proj4 = spatial_ref.ExportToProj4()

    g = nx.convert_node_labels_to_integers(g)

    return GeoGraph(srs=proj4, coords=coords, data=g)

@open_file(0, 'r')
def read_csv_projection(path):
    """
    Get projection from csv file of geo data

    Args:
        path: path to the csv as string or file

    Returns:
        Proj4 projection string if included in header else None
    """

    header = path.readline()
    if 'PROJ.4' in header:
        return header


@open_file(0, 'r')
def read_csv_geograph(csv_file, x_column="X", y_column="Y"):
    """
    load nodes csv into GeoGraph (nodes only for now)

    Args:
        csv_file:  nodal metrics csv file as string or file
        x_column, y_column:  col names to take x, y from

    Returns:
        GeoGraph of nodes including all attributes from input csv
    """

    input_proj = read_csv_projection(csv_file)
    header_row = 1 if input_proj else 0

    # read in the csv
    # NOTE:  Convert x,y via float cast to preserve precision of input
    # go back to beginning 1st
    csv_file.seek(0)
    metrics = pd.read_csv(csv_file, header=header_row,
                          converters={x_column: float, y_column: float})

    coord_cols = [x_column, y_column]
    assert all([hasattr(metrics, col) for col in coord_cols]), \
        "metrics file does not contain coordinate columns {}, {}".\
        format(x_column, y_column)

    # Stack the coords
    coords = np.column_stack(map(metrics.get, coord_cols))

    # set default projection
    if not input_proj:
        if gm.is_in_lon_lat(coords):
            input_proj = gm.PROJ4_LATLONG
        else:
            input_proj = gm.PROJ4_FLAT_EARTH

    coords_dict = dict(enumerate(coords))

    geo_nodes = GeoGraph(input_proj, coords_dict)

    # populate the rest of the attributes
    metrics_no_coords = metrics[metrics.columns.difference(coord_cols)]
    for row in metrics_no_coords.iterrows():
        index = row[0]
        attrs = row[1].to_dict()
        geo_nodes.node[index] = attrs

    return geo_nodes


def write_shp(geograph, shp_dir):
    """ 
    writes a shapefile from a networkx based GeoGraph object

    Args:
        geograph: GeoGraph object
        shp_dir:  string path to dir to write shape files

    """

    assert geograph.is_aligned()

    # looks like networkx wants us to relabel nodes by their coords
    tup_map = {i: tuple(coords) for i, coords in geograph.coords.items()}

    # copy geograph to plain networkx graph
    # (relabeling a GeoGraph doesn't seem to work)
    nx_coord_graph = nx.Graph(data=geograph)
    nx.relabel_nodes(nx_coord_graph, tup_map, copy=False)

    nx.write_shp(nx_coord_graph, shp_dir)

    if geograph.srs:
        # write srs info to prj file (nx seems to miss this)
        sr = osr.SpatialReference()
        sr.ImportFromProj4(geograph.srs)
        main_prj_filename = shp_dir + '.prj'
        edge_prj_filename = os.path.join(shp_dir, 'edges.prj')
        node_prj_filename = os.path.join(shp_dir, 'nodes.prj')

        def write_prj(prj_filename):
            out = open(prj_filename, 'w')
            out.write(sr.ExportToWkt())
            out.close()

        write_prj(main_prj_filename)
        write_prj(edge_prj_filename)
        write_prj(node_prj_filename)


@open_file(0, 'r')
def read_json_geograph(json_file):
    """
    Args:
        json_file: path to json file as string or file 

    Assumes the json is in networkx link-node format
    """
    
    js = json.load(json_file)
    g = json_graph.node_link_graph(js)

    assert all([nd.has_key('coords') for nd in g.node.values()]),\
           "json node-link graph must have nodes with coords for GeoGraph"

    # get coords
    coords = [v['coords'] for v in g.node.values()]

    # set default projection
    input_proj = ""
    if gm.is_in_lon_lat(coords):
        input_proj = gm.PROJ4_LATLONG
    else:
        input_proj = gm.PROJ4_FLAT_EARTH

    coords_dict = {k: v['coords'] for k, v in g.node.items()}
    # now get rid of 'coords' key,val for each node
    for node in g.node.values():
        node.pop('coords', None)

    geo_nodes = GeoGraph(srs=input_proj, coords=coords_dict, data=g)
    return geo_nodes

@open_file(1, 'r')
def write_json(geograph, json_file):
    """
    Args:
        geograph:  A GeoGraph object
        json_file:  as string path or file to output networkx link-node format json rep
    """
    g2 = geograph.copy()
    for nd in geograph.nodes():
        if isinstance(geograph.coords[nd], np.ndarray):
            g2.node[nd]['coords'] = geograph.coords[nd].tolist()
        else:
            g2.node[nd]['coords'] = geograph.coords[nd]

    js_g = json_graph.node_link_data(g2)
    json_file.write(json.dumps(js_g, cls=FloatEncoder))


def geojson_get_nodes(features):
    """
    Get all features that comply with GeoGraph node format within GeoJSON
    features collection

    Returns:
        nodes:  list of (node_id, node_attr_dict) tuples
        coords:  coordinates dict indexed by node_id
    """
    dict_getter = nested_dict_getter()
    nodes = []
    coords = {}
    for feature in features:
        # Only allow Point type as nodes for now
        if dict_getter(feature, ['geometry', 'type']) != 'Point':
            continue

        node_id = dict_getter(feature, ['properties', 'node_id'])
        # Nodes are only valid with a node_id property
        if node_id is None:
            continue

        node_attrs = feature['properties']
        node_attrs.pop('node_id')
        nodes.append((node_id, node_attrs))
        # At this point, we assume we have coordinates
        coords[node_id] = feature['geometry']['coordinates']

    return nodes, coords


def geojson_get_edges(features):
    """
    Get all features that comply with GeoGraph edge format within GeoJSON
    features collection

    Returns:
        edges:  list of (from_node_id, to_node_id, edge_attr_dict) tuples
    """
    dict_getter = nested_dict_getter()
    edges = []
    for feature in features:
        # Only allow LineString type as edges for now
        if dict_getter(feature, ['geometry', 'type']) != 'LineString':
            continue

        node_from_id = dict_getter(feature, ['properties', 'node_from'])
        node_to_id = dict_getter(feature, ['properties', 'node_to'])
        # Edges are only valid with node_from_id and node_to_id properties
        if node_from_id is None or node_to_id is None:
            continue

        edge_attrs = feature['properties']
        edge_attrs.pop('node_from')
        edge_attrs.pop('node_to')
        edges.append((node_from_id, node_to_id, edge_attrs))

    return edges

@open_file(0, 'r')
def read_geojson_geograph(geojson_path):
    """
    Load GeoGraph from geojson (as undirected)

    See:  docs/geograph_geojson.md for format details

    Args:
        geojson_path:  Path to geojson

    Returns:
        geograph:  GeoGraph object
    """

    geojson = json.load(geojson_path)
    dict_getter = nested_dict_getter()

    # build GeoGraph from a plain-old networkx graph
    g = nx.Graph()

    # Only add crs if it's explicitly set in crs.properties.name
    crs = dict_getter(geojson, ['crs', 'properties', 'name'])
    if crs:
        try:
            prj.Proj(crs)
        except Exception as e:
            raise SpatialReferenceInvalidException(
                "Spatial reference must comply with proj4, got {}" % crs)

    # TODO:  Apply geojson schema validation
    features = geojson['features']

    # Currently this separates nodes and coordinates due to structure of
    # GeoGraph.  This may change to allow Nodes and Edges to have distinct 
    # Geometry.
    nodes, coords = geojson_get_nodes(features)
    g.add_nodes_from(nodes)
    
    edges = geojson_get_edges(features)
    g.add_edges_from(edges)

    if crs is None and gm.is_in_lon_lat(coords.values()):
        crs = gm.PROJ4_LATLONG
    
    if crs is None:
        warnings.warn("Spatial Reference could not be set for {}".
                          format(geojson_path))

    # TODO:  More srs/projection validation?
    return GeoGraph(srs=crs, coords=coords, data=g)


def node_geojson():
    return  {
                "type": "Feature",
                "geometry": 
                {
                    "type": "Point",
                    "coordinates": []
                },
                "properties":
                {
                    "node_id": None
                }
            }

def edge_geojson():
    return  {
                "type": "Feature",
                "geometry": 
                {
                    "type": "LineString",
                    "coordinates": []
                },
                "properties":
                {
                    "node_from": None,
                    "node_to": None
                }
            }

def geojson_instance():
    return {
                "type": "FeatureCollection",
                "features": []
            }

def to_geojson(geograph):
    """
    Args:
        geograph:  A GeoGraph object

    Returns:
        geojson:  dict representing GeoGraph as dictionary
    """
    def get_coords(node_id):
        if isinstance(geograph.coords[node_id], np.ndarray):
            return geograph.coords[node_id].tolist()
        else:
            return geograph.coords[node_id]

    def node_to_geojson(node_id):
        node_geo_dict = node_geojson()
        node_geo_dict["properties"]["node_id"] = node_id
        node_geo_dict["properties"].update(geograph.node[node_id])
        node_geo_dict["geometry"]["coordinates"] = get_coords(node_id)
        return node_geo_dict

    def edge_to_geojson(node_from, node_to):
        edge_geo_dict = edge_geojson()
        edge_geo_dict["properties"]["node_from"] = node_from
        edge_geo_dict["properties"]["node_to"] = node_to
        edge_dict = geograph.edge[node_from][node_to]
        coordinates = [get_coords(node_from), get_coords(node_to)]
        # TODO:  is there a better way to handle custom coordinates than 
        # looking for "coordinates" attribute?
        if edge_dict.has_key("coordinates"):    
            coordinates = edge_dict.pop("coordinates")
        edge_geo_dict["properties"].update(edge_dict)
        edge_geo_dict["geometry"]["coordinates"] = coordinates
        return edge_geo_dict


    node_list = [node_to_geojson(node) for node in geograph.nodes()]
    edge_list = [edge_to_geojson(node_from, node_to) for 
                 node_from, node_to in geograph.edges()]
    geojson = geojson_instance()
    geojson["features"] = node_list + edge_list
    return geojson


@open_file(1, 'w')
def write_geojson(geograph, geojson_path):
    """
    Args:
        geograph:  A GeoGraph object
        geojson_path:  file to output GeoGraph GeoJSON format rep
    """
 
    geojson = to_geojson(geograph) 
    geojson_path.write(json.dumps(geojson, cls=FloatEncoder))


def read_geograph(filename, *args, **kwargs):
    """
    Read a geograph from a file whose format is defined by its extension
    args, kwargs are used to pass params along to the format 
    specific read function
    """

    # throw out arguments if we don't need 'em
    read_map = {
        '.json': lambda filename: read_json_geograph(filename),
        '.geojson': lambda filename: read_geojson_geograph(filename),
        '.shp': lambda filename: read_shp_geograph(filename, *args, **kwargs),
        '.csv': lambda filename: read_csv_geograph(filename, *args, **kwargs)
    }

    match = re.search(r'\.[^\.]*$', filename)
    if match is None or match.group() not in read_map:
        msg = "input filename {} does not have extension of .shp, .csv, "\
              ".json or .geojson".format(filename)
        raise NetworkerException(msg)
    else:
        return read_map[match.group()](filename)

def write_geograph(geograph, output_name):
    """
    Write a geograph to a file whose format is defined by its extension
    """
    match = re.search(r'\.[^\.]*$', output_name)
    if match is not None and match.group() == '.geojson':
        write_geojson(geograph, output_name)
    elif os.path.isdir(output_name):
        write_shp(geograph, output_name)
    else:
        msg = "output filename {} does not have extension of .geojson and is "\
              "not a dir (for shp)".format(output_name)
        raise NetworkerException(msg)


class FloatEncoder(json.JSONEncoder):
    """
    Hack to override how null, Infinity and -Infinity are
    json serialized.  Without this class, they are not serialized as
    strings which is not valid JSON.  

    Discussion here: http://stackoverflow.com/q/17503981

    No great solutions, but this seemed to be the safest and most efficient
    
    Mostly based on this:  https://gist.github.com/pauloalem/6244976
    """

    def __init__(self, nan_str='"null"', 
                       inf_str='"Infinity"', 
                       neg_inf_str='"-Infinity"', 
                       **kwargs):
        super(FloatEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str
        self.inf_str = inf_str
        self.neg_inf_str = neg_inf_str

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string
        representation as available.

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)
        """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring
        if self.encoding != 'utf-8':
            def _encoder(o, _orig_encoder=_encoder, _encoding=self.encoding):
                if isinstance(o, str):
                    o = o.decode(_encoding)
                return _orig_encoder(o)

        def floatstr(o, allow_nan=self.allow_nan, _repr=json.encoder.FLOAT_REPR,
                _inf=json.encoder.INFINITY, _neginf=-json.encoder.INFINITY):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = self.nan_str
            elif o == _inf:
                text = self.inf_str
            elif o == _neginf:
                text = self.neg_inf_str
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))

            return text

        _iterencode = json.encoder._make_iterencode(
                markers, self.default, _encoder, self.indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)

