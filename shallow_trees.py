from ExKMC.Tree import Tree
import numpy as np
import ctypes as ct
from sklearn.cluster import KMeans
from find_cut import fit_tree

class ShallowTree(Tree):

    def fit(self, x_data, kmeans=None, depth_factor=0.03, treat_redundances=True):
        if kmeans is None:
            if self.verbose > 0:
                print('Finding %d-means' % self.k)
            kmeans = KMeans(self.k, verbose=self.verbose, 
                        random_state=self.random_state, 
                        n_init=1, max_iter=40)
            kmeans.fit(x_data)
        y = np.array(kmeans.predict(x_data), dtype=np.int32)

        centers = np.array(kmeans.cluster_centers_, dtype=np.float64)
        self.tree = fit_tree(x_data, centers, depth_factor)
        self._feature_importance = np.zeros(x_data.shape[1])
        self.__fill_stats__(self.tree, x_data, y)

        return self

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