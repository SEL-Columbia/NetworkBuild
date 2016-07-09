# GeoGraph as GeoJSON

This document describes an informal extension to [GeoJSON](http://geojson.org/geojson-spec.html) 
in order to support reading/writing networkx-based GeoGraph objects.

## Rationale

A single-file, intermediate format for representing a GeoGraph was needed to 
facilitate transfer between tools.  In choosing a format, the following were priorities:

1.  Single file

2.  Explicit linkages between nodes and edges to eliminate "fuzziness" when joining them

3.  Widespread support (i.e. format converters and renderers)

With a minor extension to support priority 2, GeoJSON fits the bill.  Regarding point 2, [MapBox](https://github.com/mapbox) has many tools to support GeoJSON (see [geojson.io](http://geojson.io)).  GeoJSON can also be rendered natively by github.  

## Approach

We extend GeoJSON to support the notion of a graph by adding attributes to
a feature that allows them to be interpreted as a `node` or an `edge`.  

At this point, only `Point` features can be considered nodes and only 
`LineString` features can be considered edges.  

A Node is indicated by a `node_id` property.
An Edge is indicated by a `node_from` and `node_to` properties 
(each reference a `node_id`).  

This can be best seen with an example:

```
{ 
    "type": "FeatureCollection",
    "features": 
    [
        { 
            "type": "Feature",
            "geometry": 
            {
                "type": "Point", 
                "coordinates": [0.0, 0.0]
            },
            "properties": 
            {
                "node_id": 0,
                "name": "node-0"
            }
        },
        { 
            "type": "Feature",
            "geometry": 
            {
                "type": "Point", 
                "coordinates": [0.5, 0.5]
            },
            "properties": 
            {
                "node_id": 1,
                "name": "node-1"
            }
        },
        { 
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [0.0, 0.0], [0.5, 0.0], [0.5, 0.5]
                ]
            },
            "properties": {
                "name": "edge-0"
                "node_from": 0,
                "node_to": 1
            }
        }
    ]
}

```

The above represents a GeoGraph with a single edge (named 'edge-0') whose
'from' node has 'node_id' == '0' (named 'node-0') and 'to' node has 
'node_id' == '1' (named 'node-1'). 

Note that the geometry of an edge is distinct from the nodes it connects.  

## Issues

The following are some potential downsides to the chosen representation:

1.  GeoJSON is verbose, making it expensive to transfer/store (Compression or [geobuf](https://github.com/mapbox/geobuf) may help)

2.  Streaming large GeoGraphs may be difficult since the edges depend on the existence of the nodes (If required, bundling nodes with their edges may help)

3.  "Directedness" of graph is left up to the client

4.  The `node_id`, `node_from` and `node_to` properties may clash with other data provider properties (Should a namespace qualifier be prepended?)


## Future

It may become useful to consider a `Polygon` as a `node` (e.g. cities with 
area and shape) connected by some `Relation` (non-geometric) type as an `edge`
(e.g. a category describing the city).  The approach taken here allows for 
this flexibility (though the GeoGraph implementation would need to be updated
to support this). 
