# -*- coding: utf-8 -*-
import numpy as np
# from numba import jit


# @jit
def distance(u, v):
    return np.sum((u - v)**2)


class KDTree(object):

    def __init__(self, data, index=None, depth=0):
        """
        Creates a recursive space partitioning DataStructure where each node
        splits the dimension at the median of that axis. Similar to a BST,
        provides O(n log n) creation and O(log n) queries.

        Args:
            data  (np.array): dataset with shape n, k (n obs, k dim).

            Optional
            index (np.array): Index corresponding to each node in data
                              if left empty, the data is zero indexed.
            depth (int)     : This determines the axis in which to first
                              partition on e.g  0 -> x, 1 -> y, 2 -> z


        Notes:
            http://en.wikipedia.org/wiki/K-d_tree
            http://en.wikipedia.org/wiki/K-d_tree#mediaviewer/File:KDTree-animation.gif
        """
        # Build index at top level
        if isinstance(index, type(None)):
            index = np.arange(data.shape[0])

        self.n = None
        self.k = None
        self.idx = None
        self.node = None
        self.axis = None
        self.left = None
        self.right = None
        self.children = None

        self._build(data, index, depth)

    def _build(self, data, index, depth):
        """Recursively builds the child nodes of the KDTree"""
        # If there is data to partition create nodes
        if data[index].size:

            # Store the dimensions of the data and the axis to partition on
            self.n, self.k = data[index].shape
            self.axis = (self.k + depth) % self.k

            # list of nodes beneath this node
            self.children = index

            # Find the index of the data sorted on the current axis
            # and the midpoint in which to partition
            idx_data = np.column_stack((data[index], index))
            sort_ax = idx_data[np.argsort(idx_data[:, self.axis]), -1].\
                astype(int)
            partition = sort_ax.size / 2

            # Node index and data
            self.idx = sort_ax[partition]
            self.node = data[self.idx]

            # Build the branches, partitioning on the next axis
            self.left = KDTree(data, sort_ax[:partition], depth+1)
            self.right = KDTree(data, sort_ax[partition+1:], depth+1)

    def near_branch(self, point):
        """Returns the branch nearest the input point"""
        if point[self.axis] < self.node[self.axis]:
            return self.left
        return self.right

    def far_branch(self, point):
        """Returns the branch furthest the input point"""
        if self.near_branch(point) == self.left:
            return self.right
        return self.left

    def orthogonal_dist(self, point):
        """computes the distance from a point to the partition"""
        orth_point = np.copy(point)
        alt_axes = np.arange(self.k) != self.axis
        orth_point[alt_axes] = self.node[alt_axes]
        return distance(orth_point, self.node)

    def query(self, point, best=None):
        """Find the nearest neighbor of point in KDTree"""

        # Dead end backtrack up the tree
        if self.node is None:
            return best

        # Initialize best
        if best is None:
            best = (self.idx, self.node)

        # check if current node is closer than best
        if distance(self.node, point) < distance(best[1], point):
            best = (self.idx, self.node)

        # continue traversing the tree
        best = self.near_branch(point).query(point, best)

        # traverse the away branch if the orthogonal distance is less than best
        if self.orthogonal_dist(point) < distance(best[1], point):
            best = self.far_branch(point).query(point, best)
        return best

    def query_subset(self, point, subset):
        """Find the nearest neighbor of point in subset"""
        subset_vec = np.zeros(self.n)
        subset_vec[subset] = 1

        return self._query_subset(point, subset_vec, None)

    def _query_subset(self, point, subset, best=None):
        """Recursively implements constrained nearest neighbor search"""

        # Dead end backtrack up the tree
        if self.node is None:
            return best

        # Initialize node vectors
        idx_vec = np.empty_like(subset)
        child_vec = np.empty_like(subset)
        idx_vec[:] = child_vec[:] = 0
        idx_vec[self.idx] = child_vec[self.children] = 1

        # if point in subset, try to update best
        if np.dot(idx_vec, subset) != 0:
            # if closer than current best, or best is none update
            # is_closer is a thunk to prevent '__getitem__' error
            is_closer = lambda: distance(self.node, point) < \
                                distance(best[1], point)
            if best is None or is_closer():
                best = (self.idx, self.node)

        near = self.near_branch(point)
        far = self.far_branch(point)

        # check the near branch, if its nodes intersect with the queried subset
        # otherwise move to the away branch

        # Below logic doesn't make sense.  Why traverse the "far"
        # branch if we know the "best" cannot be in any of the children
        # if np.dot(child_vec, subset) > 0:
        #    best = near._query_subset(point, subset, best)
        # else:
        #    best = far._query_subset(point, subset, best)

        # This logic makes more sense
        children_in_subset = (np.dot(child_vec, subset) > 0)
        if children_in_subset:
            best = near._query_subset(point, subset, best)
        else:
            # otherwise, the best is not anywhwere below this
            # so the current best is best
            return best

        # validate best, by ensuring closer point doesn't exist just beyond
        # partition if best still has yet to be found also look
        # into this further branch
        if (best is not None and
            self.orthogonal_dist(point) < distance(best[1], point)) or \
           best is None:
            best = far._query_subset(point, subset, best)

        return best
