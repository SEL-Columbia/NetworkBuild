# -*- coding: utf-8 -*-
"""
Exceptions

Base exceptions for Networker library
"""

class NetworkerException(Exception):
    """Root of all Networker exceptions"""

class SpatialReferenceException(NetworkerException):
    """Exceptions related to spatial reference and projections"""

class SpatialReferenceMismatchException(SpatialReferenceException):
    """2 or more spatial reference systems did not match"""

class SpatialReferenceInvalidException(SpatialReferenceException):
    """Spatial reference system is invalid"""

