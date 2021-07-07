import csv

from algorithms.agglomer_clustering import AgglomerClustering
from algorithms.birch_clustering import BirchClustering
from algorithms.dbscan_clustering import DBSCANClustering
from algorithms.kmeans_clustering import KMeansClustering

__author__ = "Konstantin Bogdanoski"
__copyright__ = "Copyright 2020, BlanketClusterer"
__credits__ = ["Konstantin Bogdanoski", "Prof. PhD. Dimitar Trajanov", "MSc. Kostadin Mishev"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Konstantin Bogdanoski"
__email__ = "konstantin.b@live.com"
__status__ = "Production"


def validate_constructor(n_clusters,
                         clustering_type,
                         embeddings,
                         names,
                         items_in_cluster,
                         max_depth, group_names=None):
    if int(n_clusters) <= 0:
        raise ValueError("Invalid number of clusters")
    if clustering_type not in ["k-means", "agglomerative", "dbscan", "birch"]:
        raise ValueError("Invalid clustering type\nAllowed values: ['k-means', 'agglomerative', 'dbscan', 'birch']")
    if embeddings is None:
        raise ValueError("No embeddings specified")
    if names is None:
        raise ValueError("No names .csv file specified")
    file = open(names, "r")
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        if "key" not in row or "value" not in row:
            raise ValueError("Names are not in specified format\n"
                             "File must start with the following line:\n"
                             "key,value\n"
                             "and must be a .csv file")
        break
    if group_names is not None:
        file = open(group_names, "r")
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            if "key" not in row or "value" not in row:
                raise ValueError("Group names are not in specified format\n"
                                 "File must start with the following line:\n"
                                 "key,value\n"
                                 "and must be a .csv file")
            break

    if int(items_in_cluster) <= 5:
        raise ValueError("Number of items in clusters must be greater than 5")
    if not 0 < int(max_depth) <= 6:
        raise ValueError("Invalid argument for max depth, choose in range 1-6")
    return


class BlanketClusterer:
    """
    BlanketClusterer

    Read more in the :ref:`UserGuide <blanket_clusterer>`.

    Parameters
    ----------
    :param n_clusters
        :type Integer, default=10
        The number of clusters to be generated together with centroids.

    :param clustering_type
        :type String, default="k-means"
        Type of algorithms to be done.
        Allowed types:
        `k-means`, `agglomerative`

        The types are used from scikit-learn

    :param embeddings
        path to embeddings model on file system.
        it needs to be imported for the algorithm to operate.

        Model must contain key-value pairs, where value is a
        matrix embedding. The key is needed for the name extraction.

        Allowed models:
        `Word2Vec`,

    :param names
        path to `.csv` file containing the key-value pairs of
        names. The keys must be the same as the
        keys in the embedding model.

    :param group_names
        path to `.csv` file containing key-value pairs of group names.
        They must be in a specific format
        'left_boundary-right_boundary, generic_name'
        Example:
        '001-009, Example name'

        IMPORTANT:
        If you add group-names, the clusters will also be colorized based
        on the prevailing group

        This format is crucial if you want to name
        the clusters with specific names
    """

    def __init__(self, n_clusters=10,
                 clustering_type="k-means",
                 embeddings=None,
                 names=None,
                 items_in_cluster=20,
                 max_depth=6,
                 output_path="./output.json",
                 group_names=None):
        validate_constructor(n_clusters, clustering_type, embeddings, names,
                             items_in_cluster, max_depth, group_names)
        self.n_clusters = int(n_clusters)
        self.clustering_type = clustering_type
        self.embeddings = embeddings
        self.output_path = output_path
        self.names = names
        self.max_depth = int(max_depth)
        self.items_in_cluster = int(items_in_cluster)
        self.group_names = group_names

    def clusterize(self):
        if self.clustering_type == "k-means":
            kmeans = KMeansClustering(self.n_clusters, self.embeddings, self.names, self.items_in_cluster,
                                      self.max_depth, self.output_path, self.group_names)
            kmeans.clusterize()
        elif self.clustering_type == "agglomerative":
            agglomer = AgglomerClustering(self.n_clusters, self.embeddings, self.names, self.items_in_cluster,
                                          self.max_depth, self.output_path, self.group_names)
            agglomer.clusterize()
        elif self.clustering_type == "dbscan":
            dbs = DBSCANClustering(self.n_clusters, self.embeddings, self.names, self.items_in_cluster,
                                   self.max_depth, self.output_path, self.group_names)
            dbs.clusterize()
        elif self.clustering_type == "birch":
            brch = BirchClustering(self.n_clusters, self.embeddings, self.names, self.items_in_cluster,
                                   self.max_depth, self.output_path, self.group_names)
            brch.clusterize()
