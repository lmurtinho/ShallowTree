# ShallowTree

Implementation of the ExShallow algoritm from [Shallow decision trees for explainable k-means clustering](https://arxiv.org/abs/2112.14718), the goal of which is to find a decision tree inducing a partition of the data that is both good (in terms of the k-means cost) and explainable (in terms of being defined by a shallow tree).

## Installation

```
pip install ShallowTree
```

## Example

```python
from ShallowTree.ShallowTree import ShallowTree
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# test on the iris dataset
iris = load_iris()
data = iris.data
k = len(iris.target_names)

# create a tree that will partition the data into k clusters
tree = ShallowTree(k)

# define a KMeans instance and feed it to the tree
km = KMeans(k, random_state=1)
km.fit(data)

tree.fit(data, km)

# alternatively, fit the tree immediately;
# kmeans will run internally
tree.fit(data)

# return the score of the ttree fit and compare
# it to the score of the k-means partition
tree_score = tree.score(data)
km_score = -km.score(data)

print(tree_score / km_score)

# Construct the tree, and return cluster labels
prediction = tree.fit_predict(X)

# Tree plot saved to filename
tree.plot('filename')
```

## Citation

```bash
@article{laber2021shallow,
    title={Shallow decision trees for explainable $k$-means clustering},
    author={Laber, Eduardo and Murtinho, Lucas and Oliveira, Felipe},
    journal={arXiv preprint arXiv:2112.14718},
    year={2021}
}
```

## Contact
* [Lucas Murtinho](mailto:lucas.murtinho@gmail.com)
