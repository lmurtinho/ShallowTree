from ExKMC.Tree import Tree, Node
import numpy as np
from sklearn.cluster import KMeans
import ctypes as ct
import pathlib

libfile = pathlib.Path(__file__).parent / "lib_best_cut.so"
LIB = ct.CDLL(libfile)
    
C_FLOAT_P = ct.POINTER(ct.c_float)
C_INT_P = ct.POINTER(ct.c_int)

LIB.best_cut_single_dim.restype = ct.c_void_p
LIB.best_cut_single_dim.argtypes = [C_FLOAT_P, C_INT_P, C_FLOAT_P, 
                                     C_FLOAT_P, C_INT_P, ct.c_int, 
                                     ct.c_int, C_FLOAT_P, ct.c_double, 
                                     ct.c_bool, ct.c_bool]

class ShallowTree(Tree):
    """
    Shallow tree constructor for explainable k-means clustering.
    :param k: Number of clusters.
    :param depth_factor: Weight of penalty term to disincentivize deep trees.
    :param random_state: Random seed generator for kmeans.
    """

    def __init__(self, k, depth_factor=0.03, random_state=None):
        super().__init__(k, random_state=random_state)
        self.depth_factor = depth_factor
        self.base_tree = 'Shallow' if depth_factor else 'ExGreedy'

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
        self.tree = self._fit_tree(x_data, centers, self.depth_factor)
        self._feature_importance = np.zeros(x_data.shape[1])
        self.__fill_stats__(self.tree, x_data, y)

        return self
    
    def _fit_tree(self, data, centers, depth_factor):
        """
        Calculates the distances between all data and all centers from an
        unrestricted partition and finds a tree that induces an explainable
        partition based on the unrestricted one.
        """
        k, d = centers.shape
        unique_data, data_count = np.unique(data, axis=0, 
                                            return_counts=True)
        n = unique_data.shape[0]
        valid_centers = np.ones(k, dtype=bool)
        valid_data = np.ones(n, dtype=bool)
        distances = get_distances(unique_data, centers)
        cuts_matrix = np.zeros((d,2), dtype=int)
        return self._build_tree(unique_data, data_count, centers,
                                distances, valid_centers, valid_data,
                                depth_factor, cuts_matrix)
    
    def _build_tree(self, data, data_count, centers,
                    distances, valid_centers, valid_data,
                    depth_factor, cuts_matrix):
        """
        Builds a tree that induces an explainable partition (from 
        axis-aligned cuts) of the data, based on the centers provided 
        by an unrestricted partition.
        """
        node = Node()
        k = valid_centers.sum()
        n = valid_data.sum()
        if k == 1:
            node.value = np.argmax(valid_centers)
            return node

        dim, cut, _, terminal = self._best_cut(data, data_count, valid_data, 
                                         centers, valid_centers, distances, 
                                         depth_factor, cuts_matrix)
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
        node.left = self._build_tree(data, data_count, centers,
                                     distances, left_valid_centers, 
                                     left_valid_data, depth_factor, 
                                     cuts_matrix)
        cuts_matrix[node.feature,0] -= 1
        cuts_matrix[node.feature,1] += 1
        node.right = self._build_tree(data, data_count, centers,
                                      distances, right_valid_centers, 
                                      right_valid_data, depth_factor, 
                                      cuts_matrix)
        cuts_matrix[node.feature,1] -= 1
        return node
    
    def _best_cut(self, data, data_count, valid_data, centers, 
                    valid_centers, distances, depth_factor, cuts_matrix):
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

        for i in range(dim):
            if len(np.unique(data[valid_data,i])) == 1:
                continue
            ans = self._get_best_cut_dim(data, data_count, valid_data, 
                                        centers, valid_centers, distances, 
                                        n, k, i, LIB.best_cut_single_dim, 
                                        depth_factor, cuts_matrix[i])
            cut, cost = ans
            if cost < best_cost:
                best_cut = cut
                best_dim = i
                best_cost = cost
        if best_cut == -np.inf:
            terminal = True
        return best_dim, best_cut, best_cost, terminal
    
    def _get_best_cut_dim(self, data, data_count, valid_data, centers, 
                            valid_centers, distances, n, k, dim, func, 
                            depth_factor, cuts_row):
        """
        Calls the C function that finds the cut in data (across dimension 
        dim) with the smallest cost.
        """

        full_dist_mask = np.outer(valid_data, valid_centers)
        distances_f = np.asarray(distances[full_dist_mask], 
                                    dtype=np.float64)
        distances_pointer = distances_f.ctypes.data_as(C_FLOAT_P)

        dist_shape = distances_f.reshape(n, k)
        dist_order = np.argsort(dist_shape, axis=1)
        dist_order_f = np.asarray(dist_order, 
                                    dtype=np.int32).reshape(n*k)
        dist_order_pointer = dist_order_f.ctypes.data_as(C_INT_P)

        data_f = np.asarray(data[valid_data, dim], 
                            dtype=np.float64)
        data_p = data_f.ctypes.data_as(C_FLOAT_P)

        data_count_f = np.asarray(data_count[valid_data], 
                                    dtype=np.int32)
        data_count_p = data_count_f.ctypes.data_as(C_INT_P)

        centers_f = np.asarray(centers[valid_centers,dim], 
                                dtype=np.float64)
        centers_p = centers_f.ctypes.data_as(C_FLOAT_P)

        bool_cut_left = bool(cuts_row[0])
        bool_cut_right = bool(cuts_row[1])

        ans = np.zeros(2, dtype=np.float64)
        ans_p = ans.ctypes.data_as(C_FLOAT_P)
        func(data_p, data_count_p, centers_p, distances_pointer,
            dist_order_pointer, n, k, ans_p, depth_factor, 
            bool_cut_left, bool_cut_right)
        return ans

def get_distances(data, centers):
    """
    Finds the squared Euclidean distances between each data point in 
    data and each center in centers.
    """
    distances = np.zeros((data.shape[0], centers.shape[0]))
    for i in range(centers.shape[0]):
        distances[:,i] = np.linalg.norm(data - centers[i], axis=1) ** 2
    return distances

# if __name__ == '__main__':
#     import joblib, sys
#     from sklearn.cluster import KMeans
#     data_name = sys.argv[1]
#     seed = int(sys.argv[2]) if len(sys.argv) >= 3 else None
#     folder = '/home/lmurtinho_local/shallow_decision_trees/results/data'
#     data_dict = joblib.load(f'{folder}/{data_name}.joblib')
#     data = data_dict['data']
#     k = data_dict['k']
#     km = KMeans(k, random_state=seed)
#     km.fit(data)
#     st = ShallowTree(k=k)
#     st.fit(data, km)
#     print(st.score(data))
#     t = Tree(k)
#     t.fit(data, km)
#     print(t.score(data))