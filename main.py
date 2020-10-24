import sys

from blanket_clusterer import BlanketClusterer

__author__ = "Konstantin Bogdanoski"
__copyright__ = "Copyright 2020, BlanketClusterer"
__credits__ = ["Konstantin Bogdanoski", "Prof. PhD. Dimitar Trajanov", "MSc. Kostadin Mishev"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Konstantin Bogdanoski"
__email__ = "konstantin.b@live.com"
__status__ = "Production"

# Clusterer is called with arguments
# argv[1] - Clustering type (KMeans, Agglomerative, ...)
# argv[2] - Number of clusters
# argv[3] - Embeddings (model - Word2Vec, Node2Vec)
# argv[4] - Names (key - value)
# argv[5] - Number of items in algorithms
# argv[6] - Max depth
# argv[7] - Output path
# argv[8] - Group-names (left_border - right_border) - Optional

# If the user does not add a file containing the names of the group (_group_names_),
# the names of the clusters will be extracted based on the _Most common word_ algorithm

if __name__ == "__main__":
    clustering_type = sys.argv[1]
    num_clusters = sys.argv[2]
    embeddings = sys.argv[3]
    names = sys.argv[4]
    items_in_cluster = sys.argv[5]
    max_depth = sys.argv[6]
    output_path = sys.argv[7]
    group_names = None
    if len(sys.argv) == 9:
        group_names = sys.argv[8]

    print("[INFO] Beginning clustering")
    clusterer = BlanketClusterer(n_clusters=num_clusters, clustering_type=clustering_type,
                                 embeddings=embeddings, names=names, items_in_cluster=items_in_cluster,
                                 max_depth=max_depth, output_path=output_path,
                                 group_names=group_names)
    clusterer.clusterize()
    print("[INFO] Clustering completed")
