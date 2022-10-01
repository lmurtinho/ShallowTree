from ExKMC.Tree import Tree, Node
import numpy as np
from sklearn.cluster import KMeans
import numpy


class RandomThresholdTree(Tree):
    """
    Tree constructor for explainable k-means clustering
    using MAkarychev's algorithm.
    :param k: Number of clusters.
    :param depth_factor: Weight of penalty term to disincentivize deep trees.
    :param random_state: Random seed generator for kmeans.
    """

    def __init__(self, k, random_state=None):
        super().__init__(k, random_state=random_state)
        self.base_tree = 'RandomThreshold'
        self.rdm = np.random.default_rng(seed=int(self.random_state))

    def fit(self, x_data, centers=None):
        kmeans = KMeans(self.k, verbose=self.verbose, 
                        random_state=self.random_state)
        if centers is None:
            if self.verbose > 0:
                print('Finding %d-means' % self.k)
            kmeans.fit(x_data)
        else:
            kmeans.cluster_centers_ = centers
        
        y = np.array(kmeans.predict(x_data), dtype=np.int32)
        centers = np.array(kmeans.cluster_centers_, dtype=np.float64)
        self.tree = self._fit_tree(x_data, centers)
        self._feature_importance = np.zeros(x_data.shape[1])
        self.__fill_stats__(self.tree, x_data, y)

        return self
    
    def _fit_tree(self, data, centers):
        """
        Calculates the distances between all data and all centers from an
        unrestricted partition and finds a tree that induces an explainable
        partition based on the unrestricted one.
        """
        k, _ = centers.shape
        unique_data, data_count = np.unique(data, axis=0, return_counts=True)
        n = unique_data.shape[0]
        valid_centers = np.ones(k, dtype=bool)
        valid_data = np.ones(n, dtype=bool)
        phi_data, phi_centers = embedding(unique_data,centers)
        return self._build_tree(unique_data, data_count, centers,  0,
                                valid_centers, valid_data,phi_data,phi_centers)
        
    def _build_tree(self, data, data_count, centers, cur_height,
                    valid_centers, valid_data, phi_data, phi_centers):
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

        curr_data = data[valid_data]
        curr_centers = centers[valid_centers]
        curr_phi_data = phi_data[valid_data]
        curr_phi_centers = phi_centers[valid_centers]
        dim, cut,terminal = self._best_cut(data, data_count, valid_data,
                                valid_centers,phi_data, phi_centers)
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
        
        # the embedding function is increasing over R. 
        # we find which is the largest value below the cutoff and which is the smallest above 
        # and define the cutoff in original space to be the midpoint of the two; 
        # i.e., the cut that represents an equivalent result in terms of separation
        for i in range(k):
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

        node.left = self._build_tree(data, data_count, centers, cur_height + 1,
                                left_valid_centers, left_valid_data,phi_data,phi_centers)
        node.right = self._build_tree(data, data_count, centers,  cur_height + 1,
                                right_valid_centers, right_valid_data,phi_data,phi_centers)
        return node
   
    def _best_cut(self, data, data_count, valid_data, valid_centers, phi_data, phi_centers):
        """
        Finds the best cut across any dimension of data.
        """
        best_cut = -np.inf
        best_dim = -1

        n = valid_data.sum()
        k = valid_centers.sum()

        terminal = False
        
        ans = self._get_best_cut(data, data_count, valid_data, valid_centers,
                                    n, k,phi_data, phi_centers)
        best_dim, best_cut = ans
        if best_cut == -np.inf:
            terminal = True
        return best_dim, best_cut, terminal

    def _get_best_cut(self, data, data_count, valid_data, valid_centers,
                        n, k,phi_data, phi_centers):
        dim = len(data[0])
        
        phi_data = phi_data[valid_data]
        data_count = data_count[valid_data]

        phi_centers = phi_centers[valid_centers]

        ##### ALGORITHM
        # unseparated centers are sorted for each dimension
        # given the order, the union of cuts that separate centers is [c1,cn[
        # let c_{ij} be the j-th center (in order) in dimension i
        # At will be [[1,c_{11},c_{1m}],[2,c_{21,c2m}],....,[d,c_{d1},c_{dm}]], 
        # where m is the number of unseparated centers
        At = []
        
        for i in range(dim):
            # cut is possible if separates at least 2 centers
            # (i.e., there is a center to the right of it)
            # cut with no center to the right is last_center
            At.append([i])
            phi_centers_dim = phi_centers[:,i]
            phi_centers_dim_sort = np.argsort(phi_centers_dim)
            last_phi_center = phi_centers_dim[phi_centers_dim_sort[-1]]
            first_phi_center = phi_centers_dim[phi_centers_dim_sort[0]]
            if(last_phi_center > first_phi_center):
                At[-1].append(first_phi_center)
                At[-1].append(last_phi_center)
        total_length =0
        for i in range(dim):
            if(len(At[i])==3):
                total_length += At[i][2] - At[i][1]

        auxiliar_length = self.rdm.uniform(0,total_length)
        best_dim = -1
        best_cut = -1
        for i in range(dim):
            if(len(At[i])==3):
                auxiliar_length = auxiliar_length -(At[i][2] - At[i][1])
                if(auxiliar_length<0):
                    auxiliar_length+=At[i][2] - At[i][1]
                    best_cut = At[i][1] + auxiliar_length
                    best_dim = At[i][0]
                    break

        if(best_dim ==-1):
            #in which case the draw gives total_length. 
            #As the interval is open, I define that it will be the same as when the draw gives 0. 
            #This happens with probability 0
            for i in range(dim):
                if(len(At[i])==3):
                    best_dim = At[0]
                    best_cut = At[1]
        return best_dim,best_cut

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


def signal(x):
    if(x>0):
        return 1
    elif(x<0):
        return -1
    else:
        return 0