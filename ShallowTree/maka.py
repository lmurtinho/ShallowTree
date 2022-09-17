#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:17:38 2021

@author: lucasmurtinho
"""

import numpy as np
from ExKMC.Tree import Node
import time
import random

def get_best_cut_makarychev(data, data_count, valid_data, centers, valid_centers,
                     n, k,phi_data, phi_centers):
    start = time.time()
    dim = len(data[0])
    
    phi_data = phi_data[valid_data]
    valid_n = data.shape[0] #linhas 
    data_count = data_count[valid_data]
    valid_k = centers.shape[0] #linhas

    phi_centers = phi_centers[valid_centers]

    ##### ALGORITMO MAKARYCHEV
    #para cada dimensao temos uma ordenacao dos centros ainda nao separados
    #dada essa ordenacao, temos que a uniao dos cortes que separam os centros nessa ordenacao eh [c1,cn[
    #seja cij o j-esimo centro (ordenado) na dimensao i
    #At vai ser da forma [[1,c11,c1m],[2,c21,c2m],....,[d,cd1,cdm]], onde m eh o numero de centros nao separados ainda
    At = []
    
    for i in range(dim):
        #corte possivel se ele separa pelo menos 2 centros
        #se tem algum centro a direita, ele separa
        # o corte que nao tem centro a direita eh o last_center
        At.append([i])
        phi_centers_dim = phi_centers[:,i]
        phi_centers_dim_sort = np.argsort(phi_centers_dim)
        last_phi_center = phi_centers_dim[phi_centers_dim_sort[-1]]
        # for j in range(valid_k):
        #     if(centers_dim[j] < last_center):
        #         At[-1].append([centers_dim[j]])
        first_phi_center = phi_centers_dim[phi_centers_dim_sort[0]]
        if(last_phi_center > first_phi_center):
            At[-1].append(first_phi_center)
            At[-1].append(last_phi_center)
    total_length =0
    for i in range(dim):
        if(len(At[i])==3):
            total_length += At[i][2] - At[i][1]

    rand = random.uniform(0,total_length)
    # print(total_length)
    # print(rand)
    # print(At)
    auxiliar_length = rand
    best_dim = -1
    best_cut = -1
    for i in range(dim):
        if(len(At[i])==3):
            auxiliar_length = auxiliar_length -(At[i][2] - At[i][1])
            if(auxiliar_length<0):
                auxiliar_length+=At[i][2] - At[i][1]
                best_cut = At[i][1] + auxiliar_length
                best_dim = At[i][0]
                # print('dim',best_dim)
                # print(best_cut)
                break

    if(best_dim ==-1):
        #in which case the draw gives total_length. 
        #As the interval is open, I define that it will be the same as when the draw gives 0. 
        #This happens with probability 0
        for i in range(dim):
            if(len(At[i])==3):
                best_dim = At[0]
                best_cut = At[1]

    # Dt = 0
    # for i in range(valid_k):
    #     for j in range(i+1,valid_k):
    #         dist = np.linalg.norm((centers[i]-centers[j]),ord = 1)
    #         if(dist>Dt):
    #             Dt = dist

    # Bt =[]
    # print("Dt = ",Dt)
    # print("k=",k)
    # for i in range(dim):
    #     centers_dim = centers[:,i]
    #     order_dim_index = np.argsort(centers_dim)
        
    #     for j in range(valid_k):
    #         count = 0 #quantidade de centros a uma distancia menor que Dw/k*3
    #         idx_1 = ordem_dim_index[j]
    #         w = j+1
    #         idx2 = ordem_dim_index[w]
    #         while(np.linalg.norm((centers[idx1]-centers[idx2]),ord = 1)<= Dt/(k**3))
    #         while(np.linalg.norm((centers[idx1]-centers[idx2]),ord = 1)<= Dt/(k**3))

    #         for w in range(j+1,valid_k):
    #             #percorrer os pontos depois dele na ordem crescente dessa dim
                 
    #             if():
    #                 count += 1
    #         if(count > 0):
    #             Bt.append([i,centers_dim[j]])

    # Ct = []

    

    # for i in range(len(At)):
    #     if At[i] not in Bt:
    #         Ct.append(At[i])

    # print("At=",At)
    # # print("Bt=",Bt)
    # # print("Ct=",Ct)

    # rand_index = random.randint(0,len(At)-1)
    # best_dim = Ct[rand_index][0]
    # best_cut = Ct[rand_index][1]

    end = time.time()
    
    return best_dim,best_cut

def best_cut_makarychev(data, data_count, valid_data, centers, valid_centers,phi_data, phi_centers,cuts_matrix):
    """
    Finds the best cut across any dimension of data.
    """
    dim = centers.shape[1]
    best_cut = -np.inf
    best_dim = -1
    best_cost = np.inf

    n = valid_data.sum()
    k = valid_centers.sum()

    terminal = False
    
    ans = get_best_cut_makarychev(data, data_count, valid_data, centers, valid_centers,
                            n, k,phi_data, phi_centers)
    best_dim, best_cut = ans
    if best_cut == -np.inf:
        terminal = True
    return best_dim, best_cut, terminal

def build_tree_makarychev(data, data_count, centers,  cur_height,
 valid_centers, valid_data, phi_data, phi_centers,cuts_matrix ):
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
    

    curr_data = data[valid_data]
    curr_centers = centers[valid_centers]
    curr_phi_data = phi_data[valid_data]
    curr_phi_centers = phi_centers[valid_centers]
    dim, cut,terminal = best_cut_makarychev(data, data_count, valid_data, centers,
                              valid_centers,phi_data, phi_centers,cuts_matrix)
    #here the dim and cut are in embedded space, so we need to find the corresponding cut in the original space

    if terminal:
        node.value = np.argmax(valid_centers)
        return node

    highest_center_value_below = -np.inf 
    highest_data_value_below = -np.inf 
    smallest_data_value_over = np.inf 
    smallest_center_value_over = np.inf
    highest_center_value_below_idx = -1 
    highest_data_value_below_idx = -1 
    smallest_data_value_over_idx = -1 
    smallest_center_value_over_idx = -1
    
    #here I'm heavily using that the embedding function is increasing over R. 
    #I'm finding which is the largest value below the cutoff and which is the smallest above 
    #and defining the cutoff in original space to be the midpoint of the two. 
    #That is, the cut that represents an equivalent result in terms of separation
    for i in range(k):
        # print(i,dim)
        if(curr_phi_centers[i,dim]<=cut): #below
            if(curr_phi_centers[i,dim]>highest_center_value_below):
                highest_center_value_below = curr_phi_centers[i,dim]
                highest_center_value_below_idx = i
        else: #over
            if(curr_phi_centers[i,dim] < smallest_center_value_over):
                smallest_center_value_over = curr_phi_centers[i,dim] 
                smallest_center_value_over_idx = i

    for i in range(n):
        if(curr_phi_data[i,dim]<=cut): #below
            if(curr_phi_data[i,dim]>highest_data_value_below):
                highest_data_value_below = curr_phi_data[i,dim]
                highest_data_value_below_idx = i
        else: #over
            if(curr_phi_data[i,dim] < smallest_data_value_over):
                smallest_data_value_over = curr_phi_data[i,dim] 
                smallest_data_value_over_idx = i

    if(highest_data_value_below > highest_center_value_below):
        original_highest_below = curr_data[highest_data_value_below_idx,dim]
    else:
        original_highest_below = curr_centers[highest_center_value_below_idx,dim]

    if(smallest_center_value_over > smallest_data_value_over):
        original_smallest_over = curr_data[smallest_data_value_over_idx,dim]
    else:
        original_smallest_over = curr_centers[smallest_center_value_over_idx,dim]

    original_cut = (original_smallest_over+original_highest_below)/2


    node.feature = dim
    node.value = original_cut

    n = data.shape[0]
    data_below = 0
    left_valid_data = np.zeros(n, dtype=bool)
    right_valid_data = np.zeros(n, dtype=bool)
    for i in range(n):
        if valid_data[i]:
            if data[i,dim] <= original_cut:
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
            if centers[i, dim] <= original_cut:
                left_valid_centers[i] = True
                centers_below += 1
            else:
                right_valid_centers[i] = True

    left_centers = left_valid_centers.sum()
    right_centers = right_valid_centers.sum()
    left_data = left_valid_data.sum()
    right_data = right_valid_data.sum()
    cuts_matrix[node.feature,0] += 1


    node.left = build_tree_makarychev(data, data_count, centers, cur_height + 1,
                            left_valid_centers, left_valid_data,phi_data,phi_centers,cuts_matrix)
    cuts_matrix[node.feature,0] -= 1
    cuts_matrix[node.feature,1] += 1
    node.right = build_tree_makarychev(data, data_count, centers,  cur_height + 1,
                             right_valid_centers, right_valid_data,phi_data,phi_centers,cuts_matrix)

    cuts_matrix[node.feature,1] -= 1

    return node

def signal(x):
    if(x>0):
        return 1
    elif(x<0):
        return -1
    else:
        return 0

def embedding_dim (data_dim,center_dim, arg_sort_data, arg_sort_center):
    #data_dim = projection of data in a specific dimension
    #center_dim = projection of centes in a specific dimension
    #arg_sort_data = pointers to ordered positions in data vector
    #ard_sort_center = pointers to orderes positions in center vector
    valid_k = len(center_dim)
    valid_n = len(data_dim)
    phi_k =  np.zeros(valid_k)
    phi_n = np.zeros(valid_n)
    for i in range(1,valid_k):
        phi_k[arg_sort_center[i]] = phi_k[arg_sort_center[i-1]]+((center_dim[arg_sort_center[i]]-center_dim[arg_sort_center[i-1]])**2)/2
    for i in range(valid_n):
        idx = -1
        small_dist = np.inf 
        for j in range(valid_k):
            curr_dist = abs(center_dim[arg_sort_center[j]]-data_dim[arg_sort_data[i]])
            if(curr_dist<small_dist):
                idx = j
                small_dist = curr_dist

        sgn = signal(data_dim[arg_sort_data[i]] - center_dim[arg_sort_center[idx]])
        phi_n[arg_sort_data[i]] = phi_k[arg_sort_center[idx]] + sgn*(data_dim[arg_sort_data[i]]-center_dim[arg_sort_center[idx]])**2
    return phi_n,phi_k 

def embedding(data,centers):
    #data = data points d-dimensional
    #center = centers points d-dimensional
    valid_k = len(centers)
    valid_n = len(data)
    dim = len(data[0])
    phi_n = np.zeros((valid_n,dim))
    phi_k = np.zeros((valid_k,dim))
    for i in range(dim):
        centers_dim = centers[:,i]
        arg_sort_center = np.argsort(centers_dim)
        data_dim = data[:,i]
        arg_sort_data = np.argsort(data_dim)
        aux_n,aux_k = embedding_dim(data_dim,centers_dim, arg_sort_data, arg_sort_center)
        phi_n[:,i] = aux_n 
        phi_k[:,i] = aux_k 
    return phi_n,phi_k

def fit_tree_makarychev(data, centers):
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
    phi_data, phi_centers = embedding(unique_data,centers)
    cuts_matrix = np.zeros((d,2), dtype=int)
    return build_tree_makarychev(unique_data, data_count, centers,  0,
                         valid_centers, valid_data,phi_data,phi_centers,cuts_matrix)
