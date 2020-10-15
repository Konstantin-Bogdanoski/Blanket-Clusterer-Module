import math as math

from sklearn.cluster import AgglomerativeClustering

from .generic_clustering import GenericClustering

__author__ = "Konstantin Bogdanoski"
__copyright__ = "Copyright 2020, BlanketClusterer"
__credits__ = ["Konstantin Bogdanoski", "Prof. PhD. Dimitar Trajanov", "MSc. Kostadin Mishev"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Konstantin Bogdanoski"
__email__ = "konstantin.b@live.com"
__status__ = "Development"


class AgglomerClustering(GenericClustering):
    def clusterize_cluster(self, this_cluster):
        """
        Function used to clusterize a cluster
        :param this_cluster:
            Cluster needed to be clusterized
        :return:
            Dictionary of new clusters
        """
        if (len(this_cluster)) <= self.items_in_cluster:
            return this_cluster
        if (len(this_cluster)) > (self.items_in_cluster * self.items_in_cluster):
            agglomerative = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='cosine', linkage='complete')
        else:
            agglomerative = AgglomerativeClustering(n_clusters=(math.ceil(len(this_cluster) / self.items_in_cluster)),
                                                    affinity='cosine', linkage='complete')
        agglomerative.fit(this_cluster)
        labels = agglomerative.fit_predict(this_cluster)
        this_clusters = {}
        n = 0
        for item in labels:
            if item in this_clusters:
                this_clusters[item].append(this_cluster[n])
            else:
                this_clusters[item] = [this_cluster[n]]
            n += 1
        return this_clusters
