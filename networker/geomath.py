# -*- coding: utf-8 -*-

import numpy as np
import osr

from collections import defaultdict
from numba import jit
import math

"""
Module for geometric/geographic utility functions
"""

MEAN_EARTH_RADIUS_M = 6371010

# Note:  this says nothing about the order in the coordinates themselves
#        coordinates in this library are assumed to be in the order x, y, [z]
#        or longitude, latitude, [altitude]
PROJ4_LATLONG = "+proj=latlong +datum=WGS84"

# aka EPSG:3587 or 'tiling' projection
PROJ4_FLAT_EARTH = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 \
                    +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext \
                    +no_defs"

# meant to represent xyz based coordinates from center of earth
# (in this implementation we simplify earth into a sphere,
#  not an ellipsoid as in WGS84)
PROJ4_GEOCENTRIC = "+proj=geocent +datum=WGS84 +units=m +no_defs"

# Tolerance to determine coordinate matches...this naively applies
# to all points regardless of coordinate system, but for lat/lon
# coordinates, the affect can be gauged via the table here:
# https://en.wikipedia.org/wiki/Decimal_degrees
POINT_MATCH_TOLERANCE = 1e-8


def ang_to_vec_coords(coords, radius=MEAN_EARTH_RADIUS_M):
    """
    Transform angular coordinates on a sphere to x, y, z coordinates
    with origin at the center of the sphere

    Args:
        coords:  nx2 array of angular (degrees) longitude, latitudes
        radius:  the radius of the sphere on which to project

    Returns:
        array:   nx3 array of cartesian x, y, z coordinates with origin
            at the center of the earth (depending on the radius)

      (z) |  /
          | /.(λ,φ)
          |/ |
     -----+-------
         /|\ |  (y)
        / | \|
    (x)/  |  (h)

    """

    assert np.shape(coords)[-1] == 2, "coords last dim must be 2 (lon, lat)"
    # transpose to operate on easily and xform to radians
    lon_rad, lat_rad = np.transpose(coords) * (np.pi / 180.0)

    # the adjacent side of the Δ between x,y plane and x,y,z point
    # (this is the hypotenuse of the Δ on x,y plane defined by λ)
    h = np.cos(lat_rad)

    # use h to calc x, y points based on λ
    x = h * np.cos(lon_rad)
    y = h * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # transpose to nx3 and project from unit circle to sphere via radius
    return np.array([x, y, z]).T * radius

def _math_vec_to_ang_coords(x, y, z):
    """ 
    math functions are faster than numpy for scalars 
    http://stackoverflow.com/questions/35183787/in-python-is-math-acos-faster-than-numpy-arccos-for-scalars

    Plus, this might be clearer than thinking in vectors
    """

    h = math.sqrt(x**2 + y**2)
    
    lon_rad = math.acos(x / h)

    if y < 0:
        lon_rad = -1*lon_rad

    lat_rad = math.atan(z / h)

    # transpose to nx2 and convert back to degrees
    return (lon_rad * (180.0 / math.pi), lat_rad * (180.0 / math.pi))

def vec_to_ang_coords(coords):
    """
    Transform vector (x, y, z) coordinates on a sphere to angular coordinates
    with equatorial and polar axis

    Args:
        coords:  nx3 array of x, y, z

    Returns:
        array:   nx2 array of angular coordinates

      (z) |  /
          | /.(λ,φ)
          |/ |
     -----+-------
         /|\ |  (y)
        / | \|
       /  |  (h)
      (x)
    
    
    Looked at from the top (only x, y plane)

                    (-x <-> ±180°)
                         |
                         |
                         |
    (-y <-> -90°)--------+--------(y <-> 90°)
                         |
                         |
                         |
              (x <-> 0° <-> greenwich meridian)
    """

    assert np.shape(coords)[-1] == 3, "coords last dim must be 3 (x, y, z)"

    dimensions = np.ndim(coords)

    # transpose to operate on easily and xform to degrees
    x, y, z = np.transpose(coords)

    # we get a pretty good speedup from this
    if dimensions < 2:
        return np.array(_math_vec_to_ang_coords(x, y, z))

    h = np.sqrt(x**2 + y**2)
    
    lon_rad = np.arccos(x / h)

    # since arccos's range is (0, pi), we need to 
    # convert the angle to negative if y < 0
    if dimensions > 1:
        lon_rad[y < 0] = -1*lon_rad[y < 0]
    else:
        if y < 0:
            lon_rad = -1*lon_rad

    lat_rad = np.arctan(z / h)

    # transpose to nx2 and convert back to degrees
    return np.array([lon_rad * (180.0 / np.pi), lat_rad * (180.0 / np.pi)]).T


def spherical_distance_haversine(coord_pairs, radius=MEAN_EARTH_RADIUS_M):
    """
    Calculate distance on sphere between pairs of coordinates via
    haversine formula

    Args:
        coord_pairs:  nx2x2 array of angular (degrees) longitude, latitude
        pairs (i.e. arow has a pair of coordinates to calculate dist between)
        radius:  the radius of the sphere on which to project

    Returns:
        array:   array of distances in same units as radius

    """

    assert np.shape(coord_pairs)[1:] == (2, 2), "coord_pairs must be nx2x2"

    coord_pairs_rad = coord_pairs * np.pi / 180.0

    lat0 = coord_pairs_rad[:,0,1]
    lat1 = coord_pairs_rad[:,1,1]
    delta_lon = coord_pairs_rad[:,0,0] - coord_pairs_rad[:,1,0]
    delta_lat = lat0 - lat1

    # use haversine formula (more numerically stable than cos based)
    def haversine(theta):
        return np.sin(theta/2.0)**2

    def inv_haversine(dist):
        return np.arcsin(np.sqrt(dist))

    hav_of_dist = haversine(delta_lat) + np.cos(lat0) * np.cos(lat1) * \
        haversine(delta_lon)

    # we have 1/2 angular dist, now take inverse and * 2 to get
    # central angle between points
    central_angle = 2 * inv_haversine(hav_of_dist)
    return radius * central_angle


def spherical_distance_any(coord_pair, radius=MEAN_EARTH_RADIUS_M):
    """
    Calculate distance on sphere between pairs of coordinates
    
    Determines method to use based on number of coordinates
    """
    
    assert np.shape(coord_pair[0])[0] == np.shape(coord_pair[1])[0], \
           "Coordinate shape must match"
    dimensions = np.shape(coord_pair[0])[0]

    if dimensions == 2:
        return spherical_distance(coord_pair, radius)
    else:
        assert dimensions == 3,\
               "coords with {} dimensions are not supported".format(dimensions)
        return spherical_distance_xyz(coord_pair, radius)


def spherical_distance_xyz(coord_pair, radius=MEAN_EARTH_RADIUS_M):
    """
    Calculate distance on sphere between pairs of coordinates in r^3
    via linear algebra

    Based on:
    v1, v2 are vectors in r^3
    theta:  angle between v1, v2
    | v1 x v2 | = |v1||v2|cos(theta)
    v1 dot v2 = |v1||v2|sin(theta)

    theta = atan(|v1 x v2|/(v1 dot v2))
    
    (we use atan2 to handle special cases)

    Args:
        coord_pair:  2x3 array of cartesian x, y, z's
        radius:  the radius of the sphere on which to project

    Returns:
        distance betwteen vectors on surface of sphere in same units as radius

    """

    v1 = coord_pair[0]
    v2 = coord_pair[1]
    angle_in_rad = np.math.atan2(np.linalg.norm(np.cross(v1, v2)),
                                 np.dot(v1, v2))

    return radius * angle_in_rad


def spherical_distance(coord_pair, radius=MEAN_EARTH_RADIUS_M):
    """
    Wrapper for spherical_distance which takes a single set of pairs

    Args:
        coord_pair:  2x2 array of lon, lat pairs (e.g. [[30, 30], [31, 31]])
        radius:  radius of sphere on which to project

    Returns:
        distance:  single distance between the pair
    """

    return spherical_distance_haversine(np.array([coord_pair]), radius)[0]


def spherical_distance_dot(coord_pairs, radius=MEAN_EARTH_RADIUS_M):
    """
    Calculate distance on sphere between pairs of coordinates via
    transforming into vectors and taking dot products

    Args:
        coord_pairs:  nx2x2 array of angular (degrees) longitude, latitude
        pairs (i.e. a row has a pair of coordinates to calculate dist between)
        radius:  the radius of the sphere on which to project

    Returns:
        array:   array of distances in same units as radius

    """

    assert np.shape(coord_pairs)[1:] == (2, 2), "coord_pairs must be nx2x2"
    # convert to 3-space
    # make radius=1 because we want unit vectors to make below math easier
    coords_3 = ang_to_vec_coords(coord_pairs, radius=1)

    # law of cosines and vector algebra:
    # cosθ = u·v/(||u||*||v||)
    # in this case, ||u|| and ||v|| are 1 (on the unit sphere)
    # since sin^2(θ/2) = cosθ, use it instead since sin is
    # more stable for small θ

    dot_prod = np.sum(np.product(coords_3, axis=1), axis=1)
    theta = 2 * np.arcsin(np.sqrt((dot_prod / -2.0) + 0.5))
    return radius * theta


def euclidean_distance(coord_pair):
    """
    Euclidean distance

    Args:
        coord_pair:  lists representing points in space

    Returns:
        distance:  Euclidean distance between points
    """

    return np.sqrt(np.sum((np.asarray(coord_pair[0]) -
                   np.asarray(coord_pair[1]))**2))


def square_distance(a, b):
    """
    Calculates square distance to reduce performance overhead of square root

    Args:
        a, b:  vectors representing points in space

    Returns:
        distance:  Euclidean distance between points, squared
    """
    return np.sum((a-b)**2)


def make_bounding_box_vectors(coord_pairs):
    """
    Return a bbox for each pair in coord_pairs

    Args:
        coord_pairs:  nx2x2 array of coordinate pairs

    Returns:
        array:   array of n bboxes for each coordinate pair where
            1st pair in bbox represents min coordinates, 2nd
            pair represents max coordinates

    """

    # indexing multi-d arrays by multi-d arrays is a bit tricky
    # need to be explicit
    # this was helpful:  http://bit.ly/1BcSm5y
    coord_pair_grid = np.mgrid[[slice(x) for x in coord_pairs.shape]]
    i0 = coord_pair_grid[0]
    # order by the 2nd axis of the input coord pairs
    # (i.e. compare lon to lon, lat to lat)
    i1 = np.argsort(coord_pairs, axis=1)
    i2 = coord_pair_grid[2]

    return coord_pairs[i0, i1, i2]


def make_bounding_box_array(coords):
    """
    Return a bbox for entire coord_array

    Args:
        coords:  castable to nx2 array of coordinates

    Returns:
        array: 4 element tuple (min_x, min_y, max_x, max_y)

    """

    coord_array = np.array(coords)
    x_sort = np.argsort(coord_array[:, 0])
    y_sort = np.argsort(coord_array[:, 1])
    return coord_array[x_sort[0]][0], coord_array[y_sort[0]][1],\
        coord_array[x_sort[-1]][0], coord_array[y_sort[-1]][1]


@jit
def make_bounding_box(coord1, coord2):
    """
    Return a bbox for the pair of coordinates

    Args:
        coord1, coord2:  pair of coordinates (i.e. a segment)

    Returns:
        tuple:  4 element tuple (min_x, min_y, max_x, max_y)
    """

    bbox = make_bounding_box_vectors(np.array([[coord1, coord2]]))[0]
    return bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]


def direction(p, p1, p2):
    """
    Determine the orientation of a point relative to a segment

    Args:
        p:  point to compare to segment
        p1, p2:  points of segment to compare

    Returns:
        orientation:  + if point is right of segment
                      - if point is left of segment

    Uses cross product...recall that if:
    A=vector(p1, p); B=vector(p1, p2)

    then

    AXB > 0 if A is clockwise from B and
    AXB < 0 if A is counterclockwise from B

     p1               p1
     |\               |\
     | \              | \
    A|+ \B           B|- \A
     |   \            |   \
     p    p2          p2   p

    From this, we can say that point p is to the right or left of (p1,p2)

    """
    return np.cross((p - p1), (p2 - p1))


def on_segment_collinear(p, p1, p2):
    """
    Test whether a point collinear with a segment is ON it

    Args:
        p:  point
        p1, p2: segment points 1 and 2

    Returns:
        on_segment:  True if point is ON segment else False
    """

    # if the sign of the product of differences in one coord to
    # 2 others is <= 0, then that coordinate is between those 2.
    #
    # if p is "between" the 2 points in both dimensions and is
    # collinear with the line formed by those points, then it's ON
    # the segment formed by those points
    p_x_in_s = (p[0]-p1[0])*(p[0]-p2[0])
    p_y_in_s = (p[1]-p1[1])*(p[1]-p2[1])
    return p_x_in_s <= 0 and p_y_in_s <= 0


def segments_intersect_simple(p1, p2, p3, p4):
    """
    Do 2 2D segments intersect in Euclidean space?

    Taken from Intro to Algorithms 2nd Ed, p 937

    Args:
        p1, p2:  points comprising segment 1
        p3, p4:  points comprising segment 2

    Returns:
        True if segments intersect
    """

    # find direction of point relative to opposite segment
    d1 = direction(p1, p3, p4)
    d2 = direction(p2, p3, p4)
    d3 = direction(p3, p1, p2)
    d4 = direction(p4, p1, p2)

    # determine whether segments straddle eachother
    # use multiplication to simplify test...this ensures that
    # p1 & p2 are on opposite sides of p3, p4 AND
    # p3 & p4 are on opposite sides of p1, p2
    if d1*d2 < 0 and d3*d4 < 0:
        return True

    # test collinear cases
    if d1 == 0 and on_segment_collinear(p1, p3, p4):
        return True

    if d2 == 0 and on_segment_collinear(p2, p3, p4):
        return True

    if d3 == 0 and on_segment_collinear(p3, p1, p2):
        return True

    if d4 == 0 and on_segment_collinear(p4, p1, p2):
        return True

    return False


def segments_share_one_endpoint(p1, p2, p3, p4):
    """
    Determine whether segments share an endpoint (but are not equal)

    Args:
        p1, p2:  points comprising segment 1
        p3, p4:  points comprising segment 2

    Returns:
        True if there's a single pair of shared points between segments
    """
    # NOTE:  This will return False if segments share ALL points
    #        (i.e. are approximately equal)
    return (np.allclose(p1, p3, atol=POINT_MATCH_TOLERANCE, rtol=0) and not
            np.allclose(p2, p4, atol=POINT_MATCH_TOLERANCE, rtol=0)) or (
            np.allclose(p1, p4, atol=POINT_MATCH_TOLERANCE, rtol=0) and not
            np.allclose(p2, p3, atol=POINT_MATCH_TOLERANCE, rtol=0)) or (
            np.allclose(p2, p3, atol=POINT_MATCH_TOLERANCE, rtol=0) and not
            np.allclose(p1, p4, atol=POINT_MATCH_TOLERANCE, rtol=0)) or (
            np.allclose(p2, p4, atol=POINT_MATCH_TOLERANCE, rtol=0) and not
            np.allclose(p1, p3, atol=POINT_MATCH_TOLERANCE, rtol=0))


def segments_share_endpoint(p1, p2, p3, p4):
    """
    Determine whether segments share an endpoint

    Args:
        p1, p2:  points comprising segment 1
        p3, p4:  points comprising segment 2

    Returns:
        True if there are any shared points between segments
    """
    return (np.allclose(p1, p3, atol=POINT_MATCH_TOLERANCE, rtol=0)) or (
            np.allclose(p1, p4, atol=POINT_MATCH_TOLERANCE, rtol=0)) or (
            np.allclose(p2, p3, atol=POINT_MATCH_TOLERANCE, rtol=0)) or (
            np.allclose(p2, p4, atol=POINT_MATCH_TOLERANCE, rtol=0))


def segments_intersect(p1, p2, p3, p4):
    """
    Do 2 2D segments intersect in Euclidean space?
    *slightly* (~20%) faster than segments_intersect_simple

    http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    Args:
        p1, p2:  points comprising segment 1
        p3, p4:  points comprising segment 2

    Returns:
        True if segments intersect
    """

    # make vectors from segments
    v1 = p2 - p1
    v2 = p4 - p3
    numerator = np.cross((p3 - p1), v1)
    denominator = np.cross(v1, v2)

    if numerator == 0 and denominator == 0:
        # lines are collinear, test if overlapping

        # at least one of the points must be on the other segment
        overlapping = (on_segment_collinear(p3, p1, p2) or
                       on_segment_collinear(p4, p1, p2) or
                       on_segment_collinear(p1, p3, p4) or
                       on_segment_collinear(p2, p3, p4))

        return overlapping

    if denominator == 0:
        # lines are parallel
        return False

    u = numerator / denominator
    t = np.cross((p3 - p1), v2) / denominator

    intersecting = (0 <= t <= 1) and (0 <= u <= 1)
    return intersecting


@jit
def line_subgraph_intersection(subgraphs, rtree, p1, p2):
    """
    test for line segment intersection
    http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    TODO: Should compare this methodology to the dot product
    method to see which is more performant

    Args:
        subgraphs:  UnionFind structure representing subgraphs of a forest
        rtree:  libspatialindex rtree structure containing segments
            of subgraphs
        p1, p2:  points representing segment to test for intersection with
            subgraphs

    Returns:
        invalid_edge:  whether the edge is invalid due to > 1 intersection
            point with any subgraph
        intersecting_subnets: dict of subnet ids to number of intersections

    """

    box = make_bounding_box(p1, p2)

    # query for overlapping rectangles
    intersecting_bounds = rtree.intersection(box, objects=True)
    intersecting_subnets = defaultdict(int)

    # go through the possible intersections to validate
    for possible in intersecting_bounds:

        # Query object is in form of (u.label, v.label), (u.coord, v.coord)
        (up, vp), (p3, p4) = possible.object

        if segments_intersect(p1, p2, p3, p4):
            if segments_share_endpoint(p1, p2, p3, p4):
                continue
            else:
                # Make sure something didn't go awry such that this edge
                # doesn't represent a single subnet
                assert(subgraphs[up] == subgraphs[vp])

                # Get the subgraph the segment intersects
                subgraph_parent = subgraphs[up]
                intersecting_subnets[subgraph_parent] += 1

                # If the subgraph is intersected in more than a single location
                # this results in a 'cycle' and the segment is rejected
                if intersecting_subnets[subgraph_parent] > 1:
                    return True, intersecting_subnets

    # TODO: If this edge is valid, we need to update
    # the mv for all intersecting subnets
    return False, intersecting_subnets


def project_point_on_segment(p, v1, v2):
    """
    Find point on segment (v1, v2) nearest to point p

    v1, v2 ϵ R^2 (and must be floating point types!)

    Args:
        p:  point to project
        v1, v2:  points representing a segment

    Returns:
        proj:  projected point

    The following main concept is used to compute

        /|
       / |
    v /  |
     / θ |
    /____|______
    v*cos(θ)  u

    where v*cos(θ) is u∙v/||u|| (i.e. the scalar projection)

    The vector projection is (u∙v/||u||^2)*u

    This was helpful:
    http://math.stackexchange.com/questions/108980/projecting-a-point-onto-a-vector-2d

    """

    space = np.shape(p)[0]
    assert np.shape(v1)[0] == np.shape(v2)[0] == space, \
        "coordinate space of point and vectors must match"

    # make 2 vectors with origin at point v1
    u = np.array([v2[0] - v1[0], v2[1] - v1[1]])
    v = np.array([p[0] - v1[0], p[1] - v1[1]])

    # Dot product and magnitude^2
    dp = np.dot(u, v)
    u_mag_2 = np.sum(u ** 2)

    if dp <= 0:
        return v1

    if dp > u_mag_2:
        return v2

    # project onto u and convert back to original reference system
    point_on_u = (dp / u_mag_2) * u
    return point_on_u + v1


def arc_intersection(a1, a2, radius=MEAN_EARTH_RADIUS_M, on_arc_test=True):
    """
    EXPERIMENTAL (needs more testing)

    Find point intersecting arcs a1, a2

    a1, a2 ϵ R^3

    Args:
        a1, a2:  arcs, each represented by 2 points in R^3
        radius:  the radius to project the intersecting vector by
        on_arc_test:  if True, will return None if intersecting point not ON
                      both arcs

    Returns:
        point:  projected point in R^3 (or None if no intersection and \
            on_arc_test is True)

    The main idea here is:
    1. get orthogonal vectors to a1 and a2 (o1 and o2)
    2. find unit orthogonal vector to o1 and o2 (u)
    3. scale u by the radius to get intersecting point p
    4. determine if p is on both arcs

    """

    o1 = np.cross(a1[0], a1[1])
    o2 = np.cross(a2[0], a2[1])
    u = np.cross(o1, o2)
    # are we introducing some imprecision/sluggishness with sqrt here?
    u_hat = u / np.sqrt(np.sum(u ** 2))
    p = u_hat * radius

    if on_arc_test:
        # if the dist from p to any points of original arcs is
        # larger than the dist between those arcs, then it's
        # not ON that arc
        # (work with square dists for speed)
        d_a1 = np.sum(a1[0] - a1[1] ** 2)
        d_a2 = np.sum(a2[0] - a2[1] ** 2)
        # calculate dists to arc points
        d_p_a1_0 = np.sum(a1[0] - p ** 2)
        d_p_a1_1 = np.sum(a1[1] - p ** 2)
        d_p_a2_0 = np.sum(a2[0] - p ** 2)
        d_p_a2_1 = np.sum(a2[1] - p ** 2)

        if any([(d_p_a1_0 > d_a1), (d_p_a1_1 > d_a1),
                (d_p_a2_0 > d_a2), (d_p_a2_1 > d_a2)]):
            return None

    return p


def project_geopoint_on_arc(p, v1, v2, radius=MEAN_EARTH_RADIUS_M):
    """
    EXPERIMENTAL

    Convert latlong to 3d point, project and convert back to latlong
    """
    p_3 = ang_to_vec_coords(p, radius=radius)
    v1_3 = ang_to_vec_coords(v1, radius=radius)
    v2_3 = ang_to_vec_coords(v2, radius=radius)
    projected = project_point_on_arc(p_3, v1_3, v2_3, radius=radius)
    lon, lat = vec_to_ang_coords(projected)
    return lon, lat


def project_point_on_arc(p, v1, v2, radius=MEAN_EARTH_RADIUS_M):
    """
    EXPERIMENTAL (needs more testing)

    Find point on arc (v1, v2) nearest to point p

    v1, v2 ϵ R^3

    Args:
        p:  point to project
        v1, v2:  points representing an arc on sphere with radius
        radius:  radius of sphere

    Returns:
        point:  projected point

    Uses arc_intersection

    """

    # ensure numpy arrays
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1)

    if not isinstance(v2, np.ndarray):
        v2 = np.array(v2)

    # check if arc is 0 length (i.e. it's a point)
    # if so, then we're just projecting a point onto another point
    # which IS that other point
    if all(v1 == v2):
        return v1

    # create arc on the plane made by the orthogonal to v1, v2 and p
    o = np.cross(v1, v2)
    p_i = arc_intersection(np.array([o, p]), np.array([v1, v2]),
                           on_arc_test=False, radius=radius)

    # if p_i is further from any arc endpoint than the
    # length of arc, then return the closest endpoint to
    # p_i
    arc_length = np.sum((v1 - v2) ** 2)

    v_arr = np.array([v1, v2])
    distance_p_v = np.sum((v_arr - p_i) ** 2, axis=1)

    if any(distance_p_v > arc_length):
        return v_arr[np.argmin(distance_p_v)]

    return p_i


def is_in_lon_lat(coords):
    """
    guess whether coordinates are in lon_lat srs based on their bounds
    """

    bounds = make_bounding_box_array(coords)
    xbounds = np.array([bounds[0], bounds[2]])
    ybounds = np.array([bounds[1], bounds[3]])
    # NOTE:  some systems handle "wrapping" longitudes by going beyond 360
    return np.all(xbounds) < 360.0 and np.all(xbounds > -180.0) and \
        np.all(ybounds) < 90.0 and np.all(ybounds > -90)


def all_dists(coords, spherical=True):
    """
    distances between all coordinates (i.e. nxn)

    Args:
        coords:  nxk array of coordinates (k is the number of dimensions)
        spherical:  whether to interpret coordinates as angular and to
            calculate spherical distances

    Returns:
        distances:  n array of distances
    """

    # get all perm's of coords
    a = np.tile(coords, (len(coords), 1))
    b = np.repeat(coords, len(coords), axis=0)

    def sq_dist(a, b): return np.sum((a - b)**2, axis=1)

    dists = np.sqrt(sq_dist(a, b))
    if spherical:
        coord_pairs = np.concatenate((a[:,np.newaxis], b[:,np.newaxis]),
                                     axis=1)
        dists = spherical_distance_haversine(coord_pairs)

    zero_indices = np.array(range(len(coords))) * (len(coords) + 1)

    # so that mins are not the zero [i, i] vals
    dists[zero_indices] = np.inf
    full_dist_matrix = dists.reshape(len(coords), len(coords))
    return full_dist_matrix


def all_pair_dists(a, b, spherical=True):

    # get all perm's of coords
    a_ = np.repeat(a, len(b), axis=0)
    b_ = np.tile(b, (len(a), 1))

    def sq_dist(c, d): return np.sum((c - d)**2, axis=1)

    dists = np.sqrt(sq_dist(a_, b_))
    if spherical:
        coord_pairs = np.concatenate((a_[:,np.newaxis], b_[:,np.newaxis]),
                                     axis=1)
        dists = spherical_distance_haversine(coord_pairs)

    return np.reshape(dists, (len(a), len(b)))


def nn_dists(coords, spherical=True):
    """
    associate nearest neighbor coords by index

    Args:
        coords:  nxk array of coordinates (k is the number of dimensions)
        spherical:  whether to interpret coordinates as angular and to
            calculate spherical distances

    Returns:
        min_dists:  distance from coord to nearest neighbor
        min_ix:  index of coords nearest neighbor
    """

    full_dist_matrix = all_dists(coords, spherical)

    # find all minimum distances
    # apply min over ranges of the dist array
    min_dists = np.min(full_dist_matrix, axis=1)
    min_ix = np.argmin(full_dist_matrix, axis=1)

    return min_dists, min_ix


def coordinate_transform(srs1, srs2, coords):
    """
    Converts an array of coords from one spatial reference to another

    Args:
        srs1, srs2:  from/to spatial reference systems
        coords (np.array): [[x, y]] coords in spatial reference sr1 to be
            converted

    Returns:
        coords (np.array): [[x, y]] in spatial reference system srs2
    """

    # create transformation helper
    converter = osr.CoordinateTransformation(srs1, srs2)

    return np.asarray(converter.TransformPoints(coords))[:, :-1]


def coordinate_transform_proj4(proj1, proj2, coords):
    """
    Converts an array of coords from one spatial reference to another
    named via proj4 strings

    Args:
        proj1, proj2:  from/to spatial reference systems
        coords (np.array): [[x, y]] coords to be converted

    Returns:
        coords (np.array): [[x, y]] in spatial reference system proj2
    """

    srs1 = osr.SpatialReference()
    srs2 = osr.SpatialReference()
    srs1.ImportFromProj4(proj1)
    srs2.ImportFromProj4(proj2)

    return coordinate_transform(srs1, srs2, coords)


def get_arc_3D(v1, v2, points_per_radian=100, radius=1):
    """
    Get 3D points along sphere from angular coordinates

    Args:
        v1, v2:  angular points defining arc
        radius:  of sphere to compute arc on

    Returns:
        array:  num_points x 3 array of 3d points representing arc on sphere

    From LCKurtz and Simon Bridge responses here:
    https://www.physicsforums.com/threads/how-to-find-all-points-along-a\
    -great-circle-given-two-points-on-circle.571535/

    """

    # v1 and w become the x, y axes of the great circle
    v1_3D = ang_to_vec_coords(v1, radius=radius)
    v2_3D = ang_to_vec_coords(v2, radius=radius)
    w_axis_3D = np.cross(np.cross(v1_3D, v2_3D), v1_3D)
    # make w a vector of proper radius
    w_len = np.sqrt(square_distance([0,0,0], w_axis_3D))
    w_3D = w_axis_3D * (radius / w_len)
    arc_len = np.arccos(np.dot(v1_3D, v2_3D))
    num_points = arc_len * points_per_radian
    t = np.linspace(0, arc_len, num_points)
    u, cos_t = np.meshgrid(v1_3D, np.cos(t))
    w, sin_t = np.meshgrid(w_3D, np.sin(t))
    arc_points = u*cos_t + w*sin_t
    return arc_points
