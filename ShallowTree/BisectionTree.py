from ExKMC.Tree import Tree, Node
import numpy as np

class BisectionTree(Tree):
    """
    Explainable k-means clustering without the need of a previous partition of the data.
    :param k: Number of clusters.
    """

    def __init__(self, k):
        super().__init__(k)
        self.base_tree = 'Bisection'
        self.k = k
    
    def fit(self, data):
        cuts = self.get_cuts(data)
        self.tree = self.fill_tree(cuts)
    
    def get_best_cut(self, data):
        order = np.argsort(data, 0)
        _, d = data.shape
        best_gain = best_dim = best_val = -np.inf
        for i in range(d):
            val, gain = self.get_best_for_dim(data, order[:,i], i)
            # print(i, val, gain)
            if gain > best_gain:
                best_dim = i
                best_val = val
                best_gain = gain
        return best_dim, best_val, best_gain

    def get_best_for_dim(self, data, order, dim):
        vals, indices = np.unique(data[order, dim], return_index=True)
        if len(vals) <= 1: # cannot separate data on this dimension
            return -np.inf, -np.inf
        indices = indices[1:] - 1
        counts = indices + 1
        cs = data[order].cumsum(0)[indices]
        css = np.square(cs)
        cssm = css / counts[:,None]
        rev_cs = data.sum(0) - cs
        rev_css = np.square(rev_cs)
        rev_counts = len(data) - counts
        rev_cssm = rev_css / rev_counts[:,None]
        gains = (cssm + rev_cssm).sum(1)
        idx = gains.argmax()
        val = vals[idx]
        return val, gains[idx]

    def get_cuts(self, data):
        n, _ = data.shape
        vars_ = np.zeros(self.k)
        mask = np.zeros(n, dtype=int)
        check = np.zeros(self.k, dtype=int)
        check[0] = 1
        lengths = np.zeros(self.k)
        lengths[0] = len(data)
        cuts = []
        n_clusters = 1
        vars_[0] = data.var(axis=0).sum() * len(data)
        while n_clusters < self.k:
            # print('find cluster', n_clusters)
            best_k = best_val = best_gain = best_dim = -np.inf
            best_var0 = best_var1 = -np.inf
            for i in range(n_clusters):
                if check[i]:
                    # print('cluster', i)
                    cur_data = data[mask==i]
                    dim, val, _ = self.get_best_cut(cur_data)
                    d0 = data[(mask==i) & (data[:,dim] <= val)]
                    var0 = d0.var(0).sum() * len(d0)
                    d1 = data[(mask==i) & (data[:,dim] > val)]
                    var1 = d1.var(0).sum() * len(d1)
                    gain = vars_[i] - var0 - var1
                    # print(gain, vars_[i], var0, var1)
                    if gain > best_gain:
                        best_gain = gain
                        best_k = i
                        best_dim = dim
                        best_val = val
                        best_var0 = var0
                        best_var1 = var1
                    # print()
            # print(best_k, best_dim, best_val)
            cuts.append((best_k, best_dim, best_val))
            new_mask = (mask == best_k) & (data[:,best_dim] > best_val)
            mask[new_mask] = n_clusters
            # print(mask)
            vars_[best_k] = best_var0
            vars_[n_clusters] = best_var1
            check[n_clusters] = 1
            n_clusters += 1
        return cuts

    def fill_tree(self, cuts):
        node = Node()
        node.samples = 0
        for i in range(len(cuts)):
            c, dim, cut = cuts[i]
            to_split = self.find_node(node, c)
            if to_split is None:
                print(f'node {c} not found')
            to_split.feature = dim
            to_split.value = cut
            to_split.left = Node()
            to_split.left.samples = c
            to_split.right = Node()
            to_split.right.samples = i+1
            to_split.samples = None
        return self.fix_tree(node)

    def find_node(self, node, c):
        if node.samples == c:
            return node
        if node.left:
            left = self.find_node(node.left, c)
            if isinstance(left, Node):
                return left
        if node.right:
            right = self.find_node(node.right, c) 
            if isinstance(right, Node):
                return right
        return None
    
    def fix_tree(self, node):
        nodes = [node]
        while nodes:
            n = nodes.pop()
            if n.left:
                nodes.append(n.left)
            if n.right:
                nodes.append(n.right)
            if n.is_leaf():
                n.value = n.samples
        return node