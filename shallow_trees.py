from ExKMC.Tree import Tree
import numpy as np
from sklearn.cluster import KMeans
from find_cut import build_tree

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
        # CHANGED
        cuts_matrix = np.zeros((d,2), dtype=int)
        return build_tree(unique_data, data_count, centers,
                            distances, valid_centers, valid_data,
                            depth_factor, cuts_matrix) # CHANGED

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