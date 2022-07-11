from ExKMC.Tree import Tree, Node
import numpy as np
from sklearn.cluster import KMeans
from find_cut import best_cut

class ShallowTree(Tree):

    def fit(self, x_data, kmeans=None, depth_factor=0.03):
        if kmeans is None:
            if self.verbose > 0:
                print('Finding %d-means' % self.k)
            kmeans = KMeans(self.k, verbose=self.verbose, 
                        random_state=self.random_state, 
                        n_init=1, max_iter=40)
            kmeans.fit(x_data)
        y = np.array(kmeans.predict(x_data), dtype=np.int32)

        centers = np.array(kmeans.cluster_centers_, dtype=np.float64)
        self.tree = self._fit_tree(x_data, centers, depth_factor)
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

        dim, cut, _, terminal = best_cut(data, data_count, valid_data, 
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

def get_distances(data, centers):
    """
    Finds the squared Euclidean distances between each data point in 
    data and each center in centers.
    """
    distances = np.zeros((data.shape[0], centers.shape[0]))
    for i in range(centers.shape[0]):
        distances[:,i] = np.linalg.norm(data - centers[i], axis=1) ** 2
    return distances

if __name__ == '__main__':
    import joblib, sys
    from sklearn.cluster import KMeans
    data_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    folder = '/home/lmurtinho_local/shallow_decision_trees/results/data'
    data_dict = joblib.load(f'{folder}/{data_name}.joblib')
    data = data_dict['data']
    k = data_dict['k']
    km = KMeans(k, random_state=seed)
    km.fit(data)
    st = ShallowTree(k=k)
    st.fit(data, km)
    print(st.score(data))
    t = Tree(k)
    t.fit(data, km)
    print(t.score(data))