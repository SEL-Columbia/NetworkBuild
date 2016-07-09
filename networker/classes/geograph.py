# -*- coding: utf-8 -*-

import osr
import networkx as nx
import copy
import pyproj as prj
import numpy as np
from rtree import Rtree
from itertools import chain
from collections import deque
from networker.classes.kdtree import KDTree
import networker.geomath as gm
import networker.utils as utils

"""
Module for GeoGraph extension to networkx Graph class
"""


class GeoObject(object):

    """
    Base class for Geo objects

    Attributes:
        srs:  spatial reference system for the coords
            In proj4 string format (http://trac.osgeo.org/proj/)
        coords:  The coordinates of this object
            Abstractly, this may be a vector or a collection of vectors 
            depending on the object type

            NOTE:  coords are deep copied in attempt to avoid shared
            mutability issues

    NOTE:  Because coordinates can be set independent of the srs, the client
    must ensure these stay sane in case of any changes to one or the other. 
    """

    def __init__(self, srs, coords):
        self.srs = srs
        self.coords = copy.deepcopy(coords)

    def is_geographic(self):
        """ Returns whether the coords are geographic based on proj4 """

        sr = osr.SpatialReference()
        sr.ImportFromProj4(self.srs)
        return bool(sr.IsGeographic())

    def is_same_srs(self, other):
        """ Returns whether the SRS are the same """

        sr = osr.SpatialReference()
        sr.ImportFromProj4(self.srs)
        sr_other = osr.SpatialReference()
        sr_other.ImportFromProj4(other.srs)

        return bool(sr.IsSame(sr_other))

    def transform_coords(self, to_srs):
        """ use pyproj to transform coordinates to the srs projection """

        from_proj = prj.Proj(self.srs)
        to_proj = prj.Proj(to_srs)
        coords = {nd: prj.transform(from_proj, to_proj,
                                    self.coords[nd][0], self.coords[nd][1])
                  for nd in self.coords}
        return coords


    def lon_lat_to_cartesian_coords(self):
        """
        convert lon/lat coordinates into x,y,z
        """
        assert self.is_geographic(),\
            "GeoObject must be in Geographic coordinates to convert to x,y,z"

        assert len(self.coords.values()[0]) == 2,\
            "GeoObject must have long, lat coordinates to convert to x,y,z"


        coords_array2d, index_map = utils.coords_dict_to_array2d(self.coords)
        coords_array_xyz = gm.ang_to_vec_coords(coords_array2d)
        coords_xyz_dict = utils.array2d_to_coords_dict(coords_array_xyz, index_map)
        
        return coords_xyz_dict


    def cartesian_to_lon_lat(self):
        """
        convert x, y, z coordinates into lon/lat
        """

        assert self.is_geographic(),\
            "GeoObject must be in Geographic coordinates to convert to long, lat"

        assert len(self.coords.values()[0]) == 3,\
            "GeoObject must have x, y, z coordinates to convert to long, lat"

        coords_array2d, index_map = utils.coords_dict_to_array2d(self.coords)
        coords_array_ll = gm.vec_to_ang_coords(coords_array2d)
        coords_ll_dict = utils.array2d_to_coords_dict(coords_array_ll, index_map)
        
        return coords_ll_dict


class GeoGraph(GeoObject, nx.Graph):

    """
    class representing networkx Graph with Geo components

    Attributes:
        inherited from GeoObject and nx.Graph
        coords:  coords collection associated with node ids by ix

    See Also
    --------

    GeoObject
    networkx.Graph

    """

    def __init__(self, srs=gm.PROJ4_FLAT_EARTH, coords=None, data=None, **attr):
        """ 
        Initialize via both parent classes 
        """

        GeoObject.__init__(self, srs, coords)
        nx.Graph.__init__(self, data, **attr)

        # handle case where coords have keys not referenced by edges
        if self.coords is None:
            self.coords = dict()

        coord_keys = range(len(self.coords))
        if isinstance(self.coords, dict):
            coord_keys = self.coords.keys()

        new_nodes = set(coord_keys) - set(self.node.keys())
        self.add_nodes_from(new_nodes)

    def is_aligned(self):
        """
        Test whether this GeoGraph node dict and coords dict are aligned

        Useful in case you have a GeoGraph that has its coords/nodes modified
        """

        coord_keys = range(len(self.coords))
        if isinstance(self.coords, dict):
            coord_keys = self.coords.keys()

        assert sorted(self.nodes()) == sorted(coord_keys), \
            "GeoGraph nodes and coords not aligned"

        return True

    @staticmethod
    def compose(left, right, force_distinct=False):
        """
        'override' of networkx compose to handle GeoGraph (really just coords)
        """    
        left_geo = left
        right_geo = right

        def coord_iter(geo_coords):
            if isinstance(geo_coords, dict):
                return geo_coords.iteritems()
            else:
                return enumerate(geo_coords)

        def coord_map(old_coords, new_nodes):
            assert len(old_coords) == len(new_nodes)
            nodes = iter(new_nodes)
            d = dict()
            for k, coord in coord_iter(old_coords):
                d[nodes.next()] = copy.copy(coord)
            return d

        if force_distinct:
            left_geo = nx.convert_node_labels_to_integers(left_geo)
            left_geo.coords = coord_map(left.coords, left_geo.nodes())
            right_geo = nx.convert_node_labels_to_integers(right_geo, first_label=max(left_geo.nodes()))

            right_geo.coords = coord_map(right.coords, right_geo.nodes())

        geo = nx.compose(left_geo, right_geo)
        coord_chain = chain(coord_iter(left_geo.coords), 
                            coord_iter(right_geo.coords))

        coords = {k: copy.copy(v) for k, v in coord_chain}
        geo.coords = coords
        return geo

    def project_onto(self, other, rtree_index=None, spherical_accuracy=False):
        """
        project other GeoGraph nodes onto their nearest point (edge) in this 
        GeoGraph

        Assumes there is no node label overlap between self and other

        Args:
            other:  GeoGraph with nodes to be projected onto this GeoGraph
            rtree_index:  rtree of edges within self to be used for speeding
                up matching of self edges to other nodes
            spherical_accuracy:  if True, will try to use spherical
                calculations for more accurate results (only if 
                is_geographic() is True.  Falls back to euclidean)

        Returns:
            GeoGraph:  With nearest edges from self to nodes in other
                and additional nodes (and coords) for projected points on those
                edges (aka 'fake' nodes).  fake nodes will split these edges?
        """
        assert len(set(self.nodes()).intersection(set(other.nodes()))) == 0, \
            "the intersection of self and other graphs should be empty"

        assert self.is_same_srs(other), \
            "Spatial Reference Systems need to match in order to project onto"

        projections = {}
        for node in other.nodes():
            edge, coords = self.find_nearest_edge(other.coords[node],
                                rtree_index=rtree_index, 
                                spherical_accuracy=spherical_accuracy)
            projections[node] = (edge, coords)

        # create new GeoGraph with others coords
        geo = GeoGraph(other.srs, other.coords)

        # add nodes and edges
        # assumes nodes are numeric
        # assign new node ids by sequence starting at max node val of other
        max_node = max(other.nodes())
        for i, node in enumerate(other.nodes(), 1):
            edge, coords = projections[node]
            new_node = max_node + i
            # add 'real' node to 'fake' node edge
            geo.add_edge(node, new_node)
            # split edge with new node
            # this is where overlap between self and other would be bad
            # because we're adding nodes directly from self to new geograph
            geo.add_edge(edge[0], new_node)
            geo.add_edge(new_node, edge[1])
            # add coords
            geo.coords[new_node] = coords
            geo.coords[edge[0]] = self.coords[edge[0]]
            geo.coords[edge[1]] = self.coords[edge[1]]

        return geo

    def _has_duplicate_coords(self, tolerance=gm.POINT_MATCH_TOLERANCE, 
                              spatial_index=None):

        if spatial_index is None:
            spatial_index = KDTree(np.array(self.coords.values()))
   
        keys = self.coords.keys()

        for i in range(len(keys)):
            # Note:  i is the index of the node in the spatial_index
            coord = self.coords[keys[i]]
            gen = spatial_index.query_radius(np.array(coord), tolerance)
            try:
                for i in range(2):
                    gen.next()

            except StopIteration:
                pass

            # if there is more than 1, we have a duplicate
            if i > 1:
                return True
            else:
                return False
    
    def merge_nodes(self, u, v, node_data_map=None, self_loops=False):
        """
        merge nodes in-place
        
        ``u`` replaces ``v`` with attribute data defined by node_data_map
        (default v attributes are merged into u)

        would liked to have used [contracted_nodes](http://bit.ly/21n88s8)
        but, it doesn't do in-place merge.  So this is mainly copied
        """
        if self.is_directed():
            in_edges = ((w, u, d) for w, x, d in self.in_edges(v, data=True)
                        if self_loops or w != u)
            out_edges = ((u, w, d) for x, w, d in self.out_edges(v, data=True)
                         if self_loops or w != u)
            new_edges = chain(in_edges, out_edges)
        else:
            new_edges = ((u, w, d) for x, w, d in self.edges(v, data=True)
                         if self_loops or w != u)

        self.add_edges_from(new_edges)

        # Need to do this after edges added (generators above are lazy)
        v_data = self.node[v]
        self.remove_node(v)
        del self.coords[v]

        if node_data_map is None:
            def node_data_map(u_data, v_data):
                result = {}
                result.update(u_data)
                result.update(v_data)
                return result

        self.node[u] = node_data_map(self.node[u], v_data)


    def merge_nearby_nodes(self, radius=gm.POINT_MATCH_TOLERANCE, 
                           node_data_map=None, edge_data_map=None):
        """
        merge 'nearby' nodes within radius of eachother

        Note:  'nearby' is not necessarily an equivalence relation in
        that transitivity does not hold...if a 'near' b and b 'near' c
        a is NOT necessarily 'near' c, BUT they will be grouped.
        So, a long chain of 'near' nodes could result in the furthest nodes
        being really far apart.

        See:  https://en.wikipedia.org/wiki/Equivalence_relation

        TODO:  
        handle node_data_map, edge_data_map functions to map attributes
        from original nodes/edges to new nodes/edges

        """

        spatial_index = KDTree(np.array(self.coords.values()))
        node_ids = self.coords.keys()

        visited = set()
        for node in self.nodes():
            if node in visited:
                continue

            queue = deque()
            def add_near_nodes_to_queue(node_id):
                result = spatial_index.query_radius(np.array(self.coords[node_id]),
                                                    radius)
                near_nodes = [node_ids[idx_node[0]] for idx_node in result]
                # only add non-merged nodes to queue
                queue.extend(filter(lambda cur_node: cur_node not in visited, 
                                    near_nodes))
                # make sure we don't add them again
                visited.update(near_nodes)

            visited.add(node)
            add_near_nodes_to_queue(node)

            # keep 'matching' node ids in queue, appending as we find more
            while len(queue) > 0:
                other = queue.popleft()
                add_near_nodes_to_queue(other)
                self.merge_nodes(node, other)
       

    def get_connected_weighted_graph(self):
        """
        Get fully connected graph with edge weights defined by appropriate
        distance function
        """

        node_pairs = set(nx.product.product(self.nodes(), self.nodes()))
        node_pairs = node_pairs - set(zip(self.nodes(), self.nodes()))
        # order does NOT matter here
        node_pairs = set([frozenset(pair) for pair in node_pairs])

        dist_fun = (gm.spherical_distance if self.is_geographic()
                                          else gm.euclidean_distance)

        pairs_weights = [(pair[0], pair[1],
                         dist_fun((self.coords[pair[0]],
                                   self.coords[pair[1]])))
                         for pair in [tuple(pair_set)
                                      for pair_set in node_pairs]]

        geo = GeoGraph(self.srs, self.coords)
        geo.add_weighted_edges_from(pairs_weights)

        return geo

    def find_nearest_edge(self, coord, rtree_index=None, spherical_accuracy=False):
        """
        Find the nearest edge to the coordinate in space

        Args:
            coord:  coordinate to lookup nearest edge to
            rtree_index:  rtree of edges within self
            spherical_accuracy:  if True, will try to use spherical
                calculations for more accurate projections (only if 
                is_geographic() is True.  Falls back to euclidean)

        Returns:
            edge:  a tuple of nodes
            coords:  coordinates of nearest point on edge

        """
        assert len(self.edges()) > 0, "GeoGraph must have edges"

        project_on_edge_fun = (self._project_onto_edge_spherical 
                               if spherical_accuracy and self.is_geographic()
                               else self._project_onto_edge)
              
        if rtree_index:
            nearest_segment = rtree_index.nearest(np.ravel((coord, coord)),
                                                  objects=True).next()
            near_edge, coords = nearest_segment.object

            near_coords = project_on_edge_fun(near_edge, coord)
            
            # Note:  We always use euclidean_distance to compare segment
            # distances (even though it's not entirely accurate when dealing
            # with angular coordinates).  
            # 2 Reasons:  
            # 1. Rtree uses euclidean space (via bbox comparisons)
            # 2. It ensures that the bbox for the refinement step below is
            #    in the same units as the coords, keeping things simple
            cur_dist = gm.euclidean_distance((coord, near_coords))
            
            """
            Because the rtree's nearest function searches based on bounding
            boxes we need to continue looking through tree for closer segments.

            Here's the problem

            0
             \      2
              \  p. |
               \    3
                \
                 \
                  \
                   1

            We're looking for the nearest edge to point p.
            In this case, the bounding box for edge (0,1) encompasses point p
            whereas the bbox for edge (2,3) does not.  Therefore, edge (0,1)
            will be returned by nearest.

            To find the *actual* nearest segment, we find the distance from p
            to the *bbox based* nearest, we create a new bbox centered at p
            with length and width of this distance and use rtree's intersection
            function to find all edges within this new bbox.  If there's a
            closer edge than the *bbox based* nearest, we're guaranteed to
            find it within this new bbox.
            """

            # use nearest as a bound on intersection search to find
            # actual nearest segment (since nearest works on bbox)
            new_bbox = (coord[0] - cur_dist, coord[1] - cur_dist,
                        coord[0] + cur_dist, coord[1] + cur_dist)

            # iterate over candidates intersecting bbox finding nearest
            # to coord as actual nearest must be within bbox of current
            # nearest
            candidates = rtree_index.intersection(new_bbox, objects=True)
            for candidate in candidates:
                c_edge, c_coords = candidate.object
                p_coords = project_on_edge_fun(c_edge, coord)

                candidate_dist = gm.euclidean_distance((coord, p_coords))
                if candidate_dist < cur_dist:
                    cur_dist = candidate_dist
                    near_edge = c_edge
                    near_coords = p_coords

            return near_edge, near_coords

        else:
            # exhaustively compare segments
            min_dist = np.inf
            near_edge = None
            near_coords = None

            for edge in self.edges():
                p_coords = project_on_edge_fun(edge, coord)
                dist = gm.euclidean_distance((coord, p_coords))
                if dist < min_dist:
                    near_edge = edge
                    min_dist = dist
                    near_coords = p_coords

            return near_edge, near_coords

    def get_coord_edge_set(self):
        """
        get edges as a set of frozensets of coordinate pairs
        """

        tup_map = {i: tuple(coords) for i, coords in self.coords.items()}
        edge_sets = map(lambda e: frozenset([tup_map[e[0]], tup_map[e[1]]]),
                        self.edges())
        return set(edge_sets)

    def _project_onto_edge(self, edge, coord):
    
        """
        helper to determine the orthogonal projection of a point onto an 
        edge (segment). The projection is the nearest point to the 
        given point on the segment.

        NOTE:  This method simplifies the computation by assuming 
        coordinates on a 2d plane 

        Args:
            edge:  tuple of nodes representing edge
            coord:  coordinates of point to find projection of

        Returns:
            proj_coord:  coordinates of nearest point on edge
        """

        c0 = self.coords[edge[0]]
        c1 = self.coords[edge[1]]

        proj_coord = gm.project_point_on_segment(coord, c0, c1)
        return proj_coord

    def _project_onto_edge_spherical(self, edge, coord):
        """
        *Experimental*:  This may be slow...it computes the distance and 
        point on the sphere if the coordinates are spherical

        More useful as a test for now (to see how far off spherical vs 
        cartesian treatment is)

        Helper to determine the orthogonal projection of a point onto an 
        edge (segment). The projection is the nearest point to the 
        given point on the segment.

        Args:
            edge:  tuple of nodes representing edge
            coord:  coordinates of point to find projection of

        Returns:
            proj_coord:  coordinates of nearest point on edge
        """

        c0 = self.coords[edge[0]]
        c1 = self.coords[edge[1]]

        dimensions = np.shape(coord)[0]
        assert np.shape(c0)[0] == np.shape(c1)[0] == dimensions, \
               "coordinate space of nodes and coord must match"

        assert self.is_geographic(), ("only geographic coordinates are "
               "supported for spherical treatment")

        def get_proj_fun():
            if dimensions == 2:
                return gm.project_geopoint_on_arc
            else:
                assert dimensions == 3,\
                "coords with {} dimensions are not supported".format(dimensions)
                return gm.project_point_on_arc

        proj_fun = get_proj_fun()
        proj_coord = proj_fun(coord, c0, c1)

        return proj_coord

    def get_rtree_index(self):
        """
        Get an rtree index of the edges within this GeoGraph

        Returns:
            rtree:  rtree of edges indexed by bounding_box (4 coordinates)
                allowing lookup of the edge (node1, node2) and it's segment
        """
        def edge_generator():
            edges_bounds = map(lambda tup: (tup, gm.make_bounding_box(
                    *map(lambda n: self.coords[n], tup))), self.edges())

            for edge, box in edges_bounds:
                # Object is in form of (u.label, v.label), (u.coord, v.coord)
                yield (hash(edge), box, (edge,
                                         map(lambda ep:
                                             np.array(self.coords[ep]), edge)))

        # something's not working right with latest version of rtree/spatialib
        # index where we need to insert the objects in a loop rather than
        # via a generator to their constructor
        # Init rtree and store grid edges
        rtree = Rtree()
        for e in edge_generator():
            # takes id, bbox, and segment/edge
            rtree.insert(e[0], e[1], e[2])

        return rtree

    def find_zero_len_edges(self):
        """
        Find edges whose coordinates match (i.e. have length of 0)

        Returns:
            edges:  generator of zero length edges
        """

        for edge in self.edges():
            if self.coords[edge[0]] == self.coords[edge[1]]:
                yield edge
