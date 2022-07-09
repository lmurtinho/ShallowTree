#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:17:38 2021

@author: lucasmurtinho
"""

import numpy as np
import ctypes as ct
from ExKMC.Tree import Node
import time

LIB2 = ct.CDLL('./lib_best_cut.so')
C_FLOAT_P = ct.POINTER(ct.c_float)
C_INT_P = ct.POINTER(ct.c_int)


LIB2.best_cut_single_dim.restype = ct.c_void_p
LIB2.best_cut_single_dim.argtypes = [C_FLOAT_P, C_INT_P, C_FLOAT_P, C_FLOAT_P,
                                    C_INT_P, ct.c_int, ct.c_int, C_FLOAT_P,
                                    ct.c_bool, ct.c_double, C_FLOAT_P, ct.c_double,
                                    ct.c_bool, ct.c_bool]

# KEEP
def get_distances(data, centers):
    """
    Finds the squared Euclidean distances between each data point in data and
    each center in centers.
    """
    distances = np.zeros((data.shape[0], centers.shape[0]))
    for i in range(centers.shape[0]):
        distances[:,i] = np.linalg.norm(data - centers[i], axis=1) ** 2
    return distances

# KEEP
def get_best_cut_dim(data, data_count, valid_data, centers, valid_centers,
                     distances_pointer, dist_order_pointer,
                     n, k, dim, ratio, check_imbalance, imbalance_factor,
                     func, float_p, int_p, depth_factor, cuts_row):
    """
    Calls the C function that finds the cut in data (across dimension dim)
    with the smallest cost.
    """

    start = time.time()
    data_f = np.asarray(data[valid_data, dim], dtype=np.float64)
    data_p = data_f.ctypes.data_as(float_p)

    data_count_f = np.asarray(data_count[valid_data], dtype=np.int32)
    data_count_p = data_count_f.ctypes.data_as(int_p)

    centers_f = np.asarray(centers[valid_centers,dim], dtype=np.float64)
    centers_p = centers_f.ctypes.data_as(float_p)

    # print(cuts_row)
    bool_cut_left = bool(cuts_row[0])
    bool_cut_right = bool(cuts_row[1])

    r = np.zeros(1, dtype=np.float64)
    r[0] = ratio
    r_p = r.ctypes.data_as(float_p)
    imb_fac = ct.c_double(imbalance_factor)

    ans = np.zeros(4, dtype=np.float64)
    ans_p = ans.ctypes.data_as(float_p)
    end = time.time()
    func(data_p, data_count_p, centers_p, distances_pointer,
         dist_order_pointer, n, k, r_p, check_imbalance, imb_fac,
         ans_p, depth_factor, bool_cut_left, bool_cut_right)
    return ans

# KEEP
def best_cut(data, data_count, valid_data, centers, valid_centers, distances,
                ratio, check_imbalance, imbalance_factor, depth_factor,
                cuts_matrix):
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
        ans = get_best_cut_dim(data, data_count, valid_data, centers, valid_centers,
                               distances_p, dist_order_p, n, k, i, ratio,
                               check_imbalance, imbalance_factor,
                               LIB2.best_cut_single_dim,
                               C_FLOAT_P, C_INT_P, depth_factor,
                               cuts_matrix[i])
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
                distances, valid_centers, valid_data, ratio_type,
                check_imbalance, imbalance_factor, depth_factor, cuts_matrix,
                treat_redundances):
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
    ratio = np.inf

    dim, cut, cost, terminal = best_cut(data, data_count, valid_data, centers,
                              valid_centers, distances, ratio, check_imbalance,
                              imbalance_factor, depth_factor, cuts_matrix)
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

    left_centers = left_valid_centers.sum()
    right_centers = right_valid_centers.sum()
    left_data = left_valid_data.sum()
    right_data = right_valid_data.sum()

    if treat_redundances:
        cuts_matrix[node.feature,0] += 1
    node.left = build_tree(data, data_count, centers,
                            distances, left_valid_centers, left_valid_data,
                            ratio_type, check_imbalance, imbalance_factor, depth_factor,
                            cuts_matrix, treat_redundances)
    if treat_redundances:
        cuts_matrix[node.feature,0] -= 1
        cuts_matrix[node.feature,1] += 1
    node.right = build_tree(data, data_count, centers,
                            distances, right_valid_centers, right_valid_data,
                            ratio_type, check_imbalance, imbalance_factor, depth_factor,
                            cuts_matrix, treat_redundances)
    if treat_redundances:
        cuts_matrix[node.feature,1] -= 1
    return node

# KEEP
def fit_tree(data, centers, depth_factor, ratio_type=None,
             check_imbalance=False, imbalance_factor=1, treat_redundances=False):
    """
    Calculates the distances between all data and all centers from an
    unrestricted partition and finds a tree that induces an explainable
    partition based on the unrestricted one.
    """
    k, d = centers.shape
    unique_data, data_count = np.unique(data, axis=0, return_counts=True)
    n = unique_data.shape[0]
    valid_centers = np.ones(k, dtype=bool)
    valid_data = np.ones(n, dtype=bool)
    distances = get_distances(unique_data, centers)
    # CHANGED
    cuts_matrix = np.zeros((d,2), dtype=int)
    return build_tree(unique_data, data_count, centers,
                        distances, valid_centers, valid_data, ratio_type,
                        check_imbalance, imbalance_factor, depth_factor,
                        cuts_matrix, treat_redundances) # CHANGED
