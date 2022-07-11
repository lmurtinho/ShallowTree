#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:17:38 2021

@author: lucasmurtinho
"""

import numpy as np
import ctypes as ct

C_FLOAT_P = ct.POINTER(ct.c_float)
C_INT_P = ct.POINTER(ct.c_int)

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