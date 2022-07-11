#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:17:38 2021

@author: lucasmurtinho
"""

import numpy as np
import ctypes as ct
from ExKMC.Tree import Node

LIB2 = ct.CDLL('./lib_best_cut.so')
C_FLOAT_P = ct.POINTER(ct.c_float)
C_INT_P = ct.POINTER(ct.c_int)


LIB2.best_cut_single_dim.restype = ct.c_void_p
LIB2.best_cut_single_dim.argtypes = [C_FLOAT_P, C_INT_P, C_FLOAT_P, 
                                     C_FLOAT_P, C_INT_P, ct.c_int, 
                                     ct.c_int, C_FLOAT_P, ct.c_double, 
                                     ct.c_bool, ct.c_bool]

# KEEP
def get_best_cut_dim(data, data_count, valid_data, centers, valid_centers,
                     distances_pointer, dist_order_pointer, n, k, dim,
                     func, depth_factor, cuts_row):
    """
    Calls the C function that finds the cut in data (across dimension dim)
    with the smallest cost.
    """

    data_f = np.asarray(data[valid_data, dim], dtype=np.float64)
    data_p = data_f.ctypes.data_as(C_FLOAT_P)

    data_count_f = np.asarray(data_count[valid_data], dtype=np.int32)
    data_count_p = data_count_f.ctypes.data_as(C_INT_P)

    centers_f = np.asarray(centers[valid_centers,dim], dtype=np.float64)
    centers_p = centers_f.ctypes.data_as(C_FLOAT_P)

    bool_cut_left = bool(cuts_row[0])
    bool_cut_right = bool(cuts_row[1])

    ans = np.zeros(4, dtype=np.float64)
    ans_p = ans.ctypes.data_as(C_FLOAT_P)
    func(data_p, data_count_p, centers_p, distances_pointer,
         dist_order_pointer, n, k, ans_p, depth_factor, 
         bool_cut_left, bool_cut_right)
    return ans

# KEEP
def best_cut(data, data_count, valid_data, centers, valid_centers, 
                distances, depth_factor, cuts_matrix):
    """
    Finds the best cut across any dimension of data.
    """
    dim = centers.shape[1]
    best_cut = -np.inf
    best_dim = -1
    best_cost = np.inf

    n = valid_data.sum()
    k = valid_centers.sum()

    full_dist_mask = np.outer(valid_data, valid_centers)
    distances_f = np.asarray(distances[full_dist_mask], dtype=np.float64)
    distances_p = distances_f.ctypes.data_as(C_FLOAT_P)

    dist_shape = distances_f.reshape(n, k)
    dist_order = np.argsort(dist_shape, axis=1)
    dist_order_f = np.asarray(dist_order, dtype=np.int32).reshape(n*k)
    dist_order_p = dist_order_f.ctypes.data_as(C_INT_P)

    c_centers_below = np.zeros(dim)
    c_data_below = np.zeros(dim)

    terminal = False

    for i in range(dim):
        if len(np.unique(data[valid_data,i])) == 1:
            continue
        ans = get_best_cut_dim(data, data_count, valid_data, centers, 
                                valid_centers, distances_p, dist_order_p, 
                                n, k, i, LIB2.best_cut_single_dim,
                                depth_factor, cuts_matrix[i])
        cut, cost, c_centers_below[i], c_data_below[i] = ans
        if cost < best_cost:
            best_cut = cut
            best_dim = i
            best_cost = cost
    if best_cut == -np.inf:
        terminal = True
    return best_dim, best_cut, best_cost, terminal

# KEEP
def build_tree(data, data_count, centers,
                distances, valid_centers, valid_data,
                depth_factor, cuts_matrix):
    """
    Builds a tree that induces an explainable partition (from axis-aligned
    cuts) of the data, based on the centers provided by an unrestricted
    partition.
    """
    node = Node()
    k = valid_centers.sum()
    n = valid_data.sum()
    if k == 1:
        node.value = np.argmax(valid_centers)
        return node

    dim, cut, _, terminal = best_cut(data, data_count, valid_data, centers,
                                     valid_centers, distances, depth_factor, 
                                     cuts_matrix)
    if terminal:
        node.value = np.argmax(valid_centers)
        return node

    node.feature = dim
    node.value = cut

    n = data.shape[0]
    data_below = 0
    left_valid_data = np.zeros(n, dtype=bool)
    right_valid_data = np.zeros(n, dtype=bool)
    for i in range(n):
        if valid_data[i]:
            if data[i,dim] <= cut:
                left_valid_data[i] = True
                data_below += 1
            else:
                right_valid_data[i] = True

    k = centers.shape[0]
    centers_below = 0
    left_valid_centers = np.zeros(k, dtype=bool)
    right_valid_centers = np.zeros(k, dtype=bool)
    for i in range(k):
        if valid_centers[i]:
            if centers[i, dim] <= cut:
                left_valid_centers[i] = True
                centers_below += 1
            else:
                right_valid_centers[i] = True

    cuts_matrix[node.feature,0] += 1
    node.left = build_tree(data, data_count, centers,
                            distances, left_valid_centers, 
                            left_valid_data, depth_factor, cuts_matrix)
    cuts_matrix[node.feature,0] -= 1
    cuts_matrix[node.feature,1] += 1
    node.right = build_tree(data, data_count, centers,
                            distances, right_valid_centers, 
                            right_valid_data, depth_factor, cuts_matrix)
    cuts_matrix[node.feature,1] -= 1
    return node