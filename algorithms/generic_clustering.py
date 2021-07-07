import copy
import csv
import pickle
from datetime import datetime

import numpy as np
from rake_nltk import Rake

__author__ = "Konstantin Bogdanoski"
__copyright__ = "Copyright 2020, BlanketClusterer"
__credits__ = ["Konstantin Bogdanoski", "Prof. PhD. Dimitar Trajanov", "MSc. Kostadin Mishev"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Konstantin Bogdanoski"
__email__ = "konstantin.b@live.com"
__status__ = "Production"


class GenericClustering:
    """
    Parameters
    ----------
    :param n_clusters
        :type Integer, default=10
        The number of clusters to be generated together with centroids.

    :param embeddings
        path to embeddings on file system.
        it needs to be imported for the algorithm to operate.

        Embeddings must contain key-value pairs, where value is a
        matrix embedding. The key is needed for the name extraction.

    :param names
        path to `.csv` file containing the key-value pairs of
        names. The keys must be the same as the
        keys in the embeddings.

    :param group_names
        path to `.csv` file containing key-value pairs of group names.
        They must be in a specific format
        'left_boundary-right_boundary, generic_name'
        Example:
        '001-009, Example name'

        This format is crucial if you want to name
        the clusters with specific names

    Attributes
    ----------
    vectors
        vectors extracted from the embeddings

    X
        Vectors represented in the required format for using clustering algorithms

    Y
        Key-Value dictionary representing the codes and names for the codes

    Z
        Key-Value dictionary representing the group_codes and names for those ranges

    nums_count
        Key-Value information for group-names
        NONE if no group-names are provided

    color_constant
        Constant that is needed for extracting the color weight for the cluster/code
    """

    def __init__(self,
                 n_clusters,
                 embeddings=None,
                 names=None,
                 items_in_cluster=10,
                 max_depth=6,
                 output="output.json",
                 group_names=None):
        self.n_clusters = n_clusters
        self.embeddings = embeddings
        self.names = names
        self.group_names = group_names
        with open(embeddings, 'rb') as read:
            self.vectors = pickle.load(read)

        self.reversed_vectors = dict()
        for key, value in self.vectors.items():
            self.reversed_vectors[hash(np.array2string(value))] = key

        self.arrays = []
        for item in self.vectors.values():
            self.arrays.append(list(item))

        self.total_missing_items = 0
        self.items_in_cluster = items_in_cluster
        self.max_depth = max_depth
        self.output_path = output
        self.X = None
        self.Y = dict()
        self.Z = dict()
        self.coverages = dict()
        self.nums_count = dict()
        self.color_constant = None

    def extract_nums_count(self):
        """
        Function used to extract the group_names keys
        """
        if self.group_names is None:
            return
        with open(self.group_names, mode="r") as file:
            reader = csv.DictReader(file, delimiter=",")
            for line in reader:
                key = line['key'].strip()
                self.nums_count[key] = 0

        self.color_constant = 360 / len(self.nums_count)

    def extract_names(self):
        """
        Function used to extract the key-value names of the embeddings
        """
        file = open(self.names, "r")
        reader = csv.DictReader(file, delimiter=',')
        self.Y = dict()
        for line in reader:
            code = line['key'].strip().upper()
            description = line['value'].strip()
            self.Y[code] = description
        return

    def extract_group_names(self):
        """
        File used to extract the key-value group_names
        """
        if self.group_names is None:
            return
        file = open(self.group_names, "r")
        reader = csv.DictReader(file, delimiter=",")
        for line in reader:
            code = line['key'].strip().upper()
            description = line['value'].strip()
            self.Z[code] = description
        return

    def extract_coverages(self):
        """
        Function that calculates the percentage for the color of the algorithms
        Returns 0 if no group-names are given
        """
        if self.group_names is None:
            return
        self.coverages = copy.deepcopy(self.nums_count)
        i = 1
        for key in self.coverages:
            self.coverages[key] = self.color_constant * i
            i += 1

    def extract_vectors_as_array(self):
        """
        Function used to extract the vectors in the needed format for the algorithms
        """
        self.X = list()
        for line in self.vectors:
            self.X.append(np.array(self.vectors[line]).tolist())
        self.X = np.array(self.X, dtype=float)
        return

    def get_most_similar_key(self, array):
        hashed_array = hash(np.array2string(array))
        if hashed_array in self.reversed_vectors.keys():
            return self.reversed_vectors[hashed_array]
        else:
            # Find the most similar vector, and retrieve the corresponding key
            target_arrays = np.array(self.arrays)
            similarity = np.sqrt(
                np.sum((target_arrays / target_arrays.sum(axis=1)[:, np.newaxis] - array / np.sum(array)) ** 2, axis=1))
            similarity[np.isnan(similarity)] = similarity[0] + 1
            result = target_arrays[similarity.argmin()]
            hashed_result = hash(np.array2string(result))
            self.reversed_vectors[hashed_array] = self.reversed_vectors[hashed_result]
            self.total_missing_items += 1
            print(np.array2string(array))
            return self.reversed_vectors[hashed_array]

    def colorize_cluster(self, curr_cluster):
        """
        If no group_names are provided, no colorization technique will be done

        Colorize received cluster.
        Color value is retrieved based on the number of times
        a specific group_name is recieved

        :param curr_cluster:
            Cluster to be colorized
        :return final_coverage:
            float value for colorization
        """
        if self.group_names is None:
            return 0

        group_names = copy.deepcopy(self.nums_count)

        all_keys = []
        for value in curr_cluster:
            value = np.array(value, dtype=np.float32)
            key = self.get_most_similar_key(value)
            all_keys.append(str(key))

        for code in all_keys:
            for key in self.nums_count:
                borders = key.split("-")
                left_border = str(borders[0])
                right_border = str(borders[1])
                code = str(code)
                if left_border <= code <= right_border:
                    group_names[key] += 1
                    break

        dict_count = {}
        for key in sorted(group_names, key=group_names.get, reverse=True):
            dict_count[key] = self.nums_count[key]

        final_coverage = self.coverages[list(dict_count)[0]]
        return final_coverage

    def colorize_code(self, code):
        """
        If no group_names are provided, no specific colorization will be made
        :param code:
            Code representing the key needed to be colorized
        :return coverage:
            float value for colorization
        """
        if self.group_names is None:
            return 0
        embedding = np.array(code, dtype=np.float32)
        code = self.get_most_similar_key(embedding)
        coverage = copy.deepcopy(self.color_constant)
        final_coverage = 0
        for key in self.nums_count:
            borders = key.split("-")
            left_border = str(borders[0])
            right_border = str(borders[1])
            code = str(code)
            if left_border <= code <= right_border:
                final_coverage = coverage
                break
            coverage += self.color_constant
        return final_coverage

    def name_cluster(self, currCluster):
        """
        Name the cluster, based on the frequency of the words in the values of the names
        (Takes most common words in the cluster and generates a max of 6 words name)
        :param currCluster:
            Cluster to be named
        :return name:
            String representing the name for the cluster
        """
        r = Rake()
        all_descriptions = str()
        name = str()
        for lname in currCluster:
            embed_name = np.array(lname, dtype=np.float32)
            key_name = self.get_most_similar_key(embed_name)
            if key_name in self.Y:
                all_descriptions = (all_descriptions + " " + self.Y[key_name])
        r.extract_keywords_from_text(all_descriptions)
        num_words = 0
        if len(r.get_ranked_phrases()) > 0:
            for word in r.get_ranked_phrases()[0].split():
                if num_words >= 4:
                    break
                name = name + " " + word
                num_words = num_words + 1
        return name.strip()

    def longest_subsequence(self, curr_cluster):
        """
        If no group_names are provided, no search for longest subsequence will be done

        Function used to retrieve the longest subsequence of codes
        Needed to extract the names of the clusters

        :param curr_cluster:
            Cluster to be named
        :return name:
            String name representing the name for the cluster
        """
        if self.group_names is None:
            return ""
        all_keys = []
        for item in curr_cluster:
            value = np.array(item, dtype=np.float32)
            key = self.get_most_similar_key(value)
            all_keys.append(str(key))
        return self.count_sequences(all_keys)

    def count_sequences(self, arr):
        """
        Function used to count all of the sequences.
        (Counts the frequency of codes in each group_name)

        :param arr:
            List, representing the codes in the cluster
        :return name:
            String name representing the final name for the cluster
        """
        if self.group_names is None:
            return ""

        nums_count = copy.deepcopy(self.nums_count)
        for code in arr:
            for key in nums_count:
                borders = key.split("-")
                left_border = str(borders[0])
                right_border = str(borders[1])
                code = str(code)
                if left_border <= code <= right_border:
                    nums_count[key] += 1
                    break

        dict_count = {}
        for key in sorted(nums_count, key=nums_count.get, reverse=True):
            dict_count[key] = nums_count[key]
        name_keys = []
        num_keys = 0
        # The main group_names, must contain at least 60% of all the items in the cluster
        cluster_size = (60 * len(arr)) / 100
        for item in dict_count:
            if num_keys < cluster_size and len(name_keys) < 3:
                name_keys.append(item)
                num_keys += int(dict_count[item])
        name = list()
        for key in name_keys:
            for word in self.Y[key].split(" "):
                name.append(word)
            if name_keys.index(key) != (len(name_keys) - 1):
                name.append("/")
        final_name = ""
        for word in name:
            final_name += word + " "
        return final_name.strip()

    def print_cluster(self, this_cluster, cluster_names, code_id, missing_data, out):
        """
        Function used to print the cluster into file
        :param this_cluster:
            cluster to be printed
        :param cluster_names:
            names to be used
        :param code_id:
            id of the code
        :param missing_data:
            list of all missing values
            (values for which names are not found in self.Y)
        :param out:
            output stream
        """
        last = this_cluster[-1]
        for entry in this_cluster:
            value = np.array(entry, dtype=np.float32)
            key = str(self.get_most_similar_key(value)).upper()
            out.write('{ \n "id": "' + str(key) + '",')
            if key in self.Y:
                code_coverage = self.colorize_code(entry)
                out.write('\n "label": "' + self.Y[key] + '"')
                if self.group_names is None:
                    out.write('\n}')
                else:
                    out.write(',\n "coverage": ' + str(code_coverage) + '\n}')
                if code_id in cluster_names:
                    cluster_names[code_id] = cluster_names[code_id] + "," + self.Y[key]
                else:
                    cluster_names[code_id] = self.Y[key]
            else:
                out.write('\n "label": "NAME NOT FOUND"\n}')
                missing_data.append(key)
            if not ((entry == last).all()):
                out.write(",")
        out.flush()
        return

    def print_cluster_end(self, coverage, out):
        """
        Function needed to print the end of the cluster (after printing all other entities and subclusters in it)
        :param coverage:
            Coverage value of cluster - Color
        :param out:
            output stream
        """
        out.write(']\n')
        if self.group_names is None:
            out.write("},\n")
        else:
            out.write(',\n "coverage": ' + str(coverage) + "},")
        out.flush()
        return

    def clusterize_cluster(self, this_cluster):
        """
        Function used to clusterize a cluster
        Implement it for each clustering algorithm
        :param this_cluster:
            Cluster needed to be clusterized
        :return:
            Dictionary of new clusters
        """
        return None

    def run(self):
        """
        Function that clusters the dataset

        Outputs the result into a FoamTree compatible JSON object
        """
        # First Iteration
        start_time = datetime.now()
        clusters = self.clusterize_cluster(self.X)
        cluster_names = {}

        # Second, Third, Fourth, Fifth and Sixth Iteration
        missing_data = []
        out = open(self.output_path, "w+")
        out.write('{ \n "groups": [')
        iter0 = 0
        total_clusters = 0
        for entry in clusters:
            cluster_name = self.longest_subsequence(clusters[entry])
            if cluster_name == "":
                cluster_name = self.name_cluster(clusters[entry])
            out.write('\n { "id":"' + str(iter0) + '",')
            out.write('\n "label": "' + cluster_name + '",')
            out.write('\n "groups": [')
            code_id = str(iter0)
            if len(clusters[entry]) > self.items_in_cluster and self.max_depth > 1:
                cluster1 = clusters[entry]
                cluss1 = self.clusterize_cluster(cluster1)
                iter1 = 0
                for i1 in cluss1:
                    cluster_name = self.longest_subsequence(cluss1[i1])
                    if cluster_name == "":
                        cluster_name = self.name_cluster(cluss1[i1])
                    out.write('\n { "id":"' + (str(iter0) + "." + str(iter1)) + '",')
                    out.write('\n "label": "' + cluster_name + '",')
                    out.write('\n "groups": [')
                    id1 = (str(iter0) + "." + str(iter1))
                    if len(cluss1[i1]) > self.items_in_cluster and self.max_depth > 2:
                        cluster2 = cluss1[i1]
                        cluss2 = self.clusterize_cluster(cluster2)
                        iter2 = 0
                        for i2 in cluss2:
                            cluster_name = self.longest_subsequence(cluss2[i2])
                            if cluster_name == "":
                                cluster_name = self.name_cluster(cluss2[i2])
                            out.write('\n { "id":"' + (str(iter0) + "." + str(iter1) + "." + str(iter2)) + '",')
                            out.write('\n "label": "' + cluster_name + '",')
                            out.write('\n "groups": [')
                            id2 = (str(iter0) + "." + str(iter1) + "." + str(iter2))
                            if len(cluss2[i2]) > self.items_in_cluster and self.max_depth > 3:
                                cluster3 = cluss2[i2]
                                cluss3 = self.clusterize_cluster(cluster3)
                                iter3 = 0
                                for i3 in cluss3:
                                    cluster_name = self.name_cluster(cluss3[i3])
                                    out.write('\n { "id":"' + (
                                            str(iter0) + "." + str(iter1) + "." + str(iter2) + "." + str(iter3)) + '",')
                                    out.write('\n "label": "' + cluster_name + '",')
                                    out.write('\n "groups": [')
                                    id3 = (str(iter0) + "." + str(iter1) + "." + str(iter2) + "." + str(iter3))
                                    if len(cluss3[i3]) > self.items_in_cluster and self.max_depth > 4:
                                        cluster4 = cluss3[i3]
                                        cluss4 = self.clusterize_cluster(cluster4)
                                        iter4 = 0
                                        for i4 in cluss4:
                                            cluster_name = self.name_cluster(cluss4[i4])
                                            out.write('\n { "id":"' + (
                                                    str(iter0) + "." + str(iter1) + "." + str(iter2) + "."
                                                    + str(iter3) + "." + str(iter4)) + '",')
                                            out.write('\n "label": "' + cluster_name + '",')
                                            out.write('\n "groups": [')
                                            id4 = (str(iter0) + "." + str(iter1) + "." + str(iter2) + "." + str(
                                                iter3) + "." + str(iter4))
                                            if len(cluss4[i4]) > self.items_in_cluster and self.max_depth > 5:
                                                cluster5 = cluss4[i4]
                                                cluss5 = self.clusterize_cluster(cluster5)
                                                iter5 = 0
                                                for i5 in cluss5:
                                                    cluster_name = self.name_cluster(cluss5[i5])
                                                    out.write(
                                                        '\n { "id":"' + (str(iter0) + "." + str(iter1) + "." + str(
                                                            iter2) + "." + str(iter3) + "." + str(iter4) + "." + str(
                                                            iter5)) + '",')
                                                    id5 = (str(iter0) + "." + str(iter1) + "." + str(iter2) + "."
                                                           + str(iter3) + "." + str(iter4) + "." + str(iter5))
                                                    out.write('\n "label": "' + cluster_name + '",')
                                                    out.write('\n "groups": [')
                                                    self.print_cluster(cluss5[i5], cluster_names, id5,
                                                                       missing_data, out)
                                                    iter5 = int(iter5) + 1
                                                    total_clusters += 1
                                                    coverage5 = self.colorize_cluster(cluss5[i5])
                                                    print("[INFO] Writing cluster: ", total_clusters, end="\r")
                                                    self.print_cluster_end(coverage5, out)
                                            else:
                                                self.print_cluster(cluss4[i4], cluster_names, id4, missing_data, out)
                                                total_clusters += 1
                                            iter4 = int(iter4) + 1
                                            coverage4 = self.colorize_cluster(cluss4[i4])
                                            print("[INFO] Writing cluster: ", total_clusters, end="\r")
                                            self.print_cluster_end(coverage4, out)
                                    else:
                                        self.print_cluster(cluss3[i3], cluster_names, id3, missing_data, out)
                                        total_clusters += 1
                                    iter3 = int(iter3) + 1
                                    coverage3 = self.colorize_cluster(cluss3[i3])
                                    print("[INFO] Writing cluster: ", total_clusters, end="\r")
                                    self.print_cluster_end(coverage3, out)
                            else:
                                self.print_cluster(cluss2[i2], cluster_names, id2, missing_data, out)
                                total_clusters += 1
                            iter2 = int(iter2) + 1
                            coverage2 = self.colorize_cluster(cluss2[i2])
                            print("[INFO] Writing cluster: ", total_clusters, end="\r")
                            self.print_cluster_end(coverage2, out)
                    else:
                        self.print_cluster(cluss1[i1], cluster_names, id1, missing_data, out)
                        total_clusters += 1
                    iter1 = int(iter1) + 1
                    coverage1 = self.colorize_cluster(cluss1[i1])
                    print("[INFO] Writing cluster: ", total_clusters, end="\r")
                    self.print_cluster_end(coverage1, out)
            else:
                self.print_cluster(clusters[entry], cluster_names, code_id, missing_data, out)
                total_clusters += 1
            iter0 = int(iter0) + 1
            coverage = self.colorize_cluster(clusters[entry])
            self.print_cluster_end(coverage, out)
        out.write("]}")
        out.flush()
        out.close()
        print("[INFO] Done clustering")
        total_time = datetime.now() - start_time
        print("[INFO] Time: " + str(total_time.seconds) + " seconds")
        print("[INFO] Total number of clusters: " + str(total_clusters))

    def clusterize(self):
        """
        Main function used to implement the clusterization

        Output is Writing in a file specified in the output path
        """
        self.extract_names()
        self.extract_group_names()
        self.extract_nums_count()
        self.extract_coverages()
        self.extract_vectors_as_array()
        self.run()
