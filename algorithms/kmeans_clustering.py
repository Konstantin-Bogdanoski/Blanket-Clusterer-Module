import math as math

import nltk
from nltk.cluster import KMeansClusterer

from .generic_clustering import GenericClustering

__author__ = "Konstantin Bogdanoski"
__copyright__ = "Copyright 2020, BlanketClusterer"
__credits__ = ["Konstantin Bogdanoski", "Prof. PhD. Dimitar Trajanov", "MSc. Kostadin Mishev"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Konstantin Bogdanoski"
__email__ = "konstantin.b@live.com"
__status__ = "Production"


class KMeansClustering(GenericClustering):
    def clusterize_cluster(self, this_cluster):
        """
        Function used to clusterize a algorithms
        :param this_cluster:
            Cluster needed to be clusterized
        :return:
            Dictionary of new clusters
        """
        if (len(this_cluster)) <= self.items_in_cluster:
            return this_cluster
        if (len(this_cluster)) > (self.items_in_cluster * self.items_in_cluster):
            kmeans = KMeansClusterer(self.n_clusters, distance=nltk.cluster.util.cosine_distance)
        else:
            kmeans = KMeansClusterer(math.ceil(len(this_cluster) / self.items_in_cluster),
                                     distance=nltk.cluster.util.cosine_distance)
        labels = kmeans.cluster(this_cluster, assign_clusters=True)
        this_clusters = {}
        n = 0
        for item in labels:
            if item in this_clusters:
                this_clusters[item].append(this_cluster[n])
            else:
                this_clusters[item] = [this_cluster[n]]
            n += 1
        return this_clusters
