import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import hdbscan
from joblib import Memory
from sklearn.datasets import fetch_openml
#import umap

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

#normalize pixel values
X = X / 255.0

X = X[:500]
y = y[:500]

clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
clusterer.fit(X)


clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
plt.show()

clusterer.condensed_tree_.plot()
plt.show()
