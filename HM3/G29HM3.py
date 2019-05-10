from pyspark import SparkContext, SparkConf
import argparse
import os
from math import sqrt, inf
import random
from pyspark.mllib.linalg import Vectors
from VectorInput import readVectorsSeq
from typing import List


def argparser() -> (str, int, int):
    # initial settings to accept incoming dataset and k number of partitions
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', help='path/to/file.txt', required=True)
    parser.add_argument('-k', '--n_of_clusters', help='number of clusters', required=True)
    parser.add_argument('-iter', '--n_of_iterations', help='number of iterations', required=True)
    args = vars(parser.parse_args())

    path = os.path.join(os.getcwd(), args['filename'])
    if not os.path.isfile(path):
        raise EnvironmentError('Path/to/file.txt is not right')
    k = int(args['n_of_clusters'])
    iterations = int(args['n_of_iterations'])
    return path, k, iterations


def conf_spark_env() -> SparkContext:
    # defining Spark context
    spark_conf = SparkConf(True).setAppName('G29HM3').setMaster('local')
    return SparkContext(conf=spark_conf)


# def partition(P: List[Vectors.dense], S: List[Vectors.dense], WP: List[int]) -> (List[List[Vectors.dense]], List[List[int]]):
#     """
#     Partitions P in S different clusters.
#     :param P: List of Vectors.dense points
#     :param S: List of Vectors.dense centroids that has been provided, one for cluster
#     :return: Cluster list of list of Vectors.dense, divided per clusters. The first element of each cluster
#     is the centroid
#     """
#     cP = [*P]
#     clusters = list(map(lambda x: [x], S))
#     weights = [[] for _ in clusters]
#     # TODO: check if it is possible to remove this costy call
#     # cP.remove(S)
#
#     for p_idx, p in enumerate(cP):
#         min_dist = inf
#         r = -1
#         for i, s in enumerate(S):
#             temp = WP[i]*Vectors.squared_distance(p, s)
#             if temp < min_dist:
#                 r = i
#                 min_dist = temp
#         clusters[r].append(p)
#         weights[r].append(WP[p_idx])
#     return clusters, weights


def partition_distances(
        P: List[Vectors.dense],
        S: List[Vectors.dense],
        wp: List[int],
        partition: bool
) -> (
        List[List[Vectors.dense]],
        List[List[int]],
        List[float]
):
    # Creating sum_of_distances: we need to take the nearest point of S per each point of P\S.
    # We calculate the value to use it afterward.
    lenP = len(P)
    # defining initial cluster so we can keep it and return
    C = [[] for _ in S]
    WC = [[] for _ in S]

    distances = [0.0 for _ in range(lenP)]
    for i, p in enumerate(P):
        dist = inf
        r = -1
        # TODO: simplify iterations remembering old computed values of distances for previous centers
        for j, s in enumerate(S):
            temp = wp[i] * sqrt(Vectors.squared_distance(p, s))
            if temp < dist:
                dist = temp
                r = j

        # Here I know which point is nearer to, so we construct the partitioning directly
        if not partition:
            distances[i] = dist  # distance of every point from the nearest of S
        else:
            C[r].append(p)
            WC[r].append(wp[i])

    return C, WC, distances


def select_c(
        S: List[Vectors.dense],
        P_minus_S: List[Vectors.dense],
        wp: List[int]) -> int:
    """
    Select the right index of P_minus_S to be returned by the algorithm in page 16 of Clustering-2-1819.pdf file
    :param S: List of centroids already found
    :param P_minus_S: P\S
    :param wp: weights of P_minus_one
    :return: r is the index of new centroid; C and WC are the partitions with the points already found
    """

    _, _, distances = partition_distances(P_minus_S, S, wp, partition=False)

    sum_of_distances = sum(distances)

    pis = [wp[i]*dist / sum_of_distances for i, dist in enumerate(distances)]

    x = random.random()
    r = -1
    left_sum = 0
    right_sum = 0
    toggled = False
    for i, pi_j in enumerate(pis):
        # taking a step-forward weight we are sure we are not surpassing x
        if not toggled and left_sum + pi_j <= x:
            left_sum += pi_j
        else:
            if not toggled:  # selecting index of right point (slide 16)
                r = i
                toggled = True
                right_sum = left_sum
            right_sum += pi_j

    assert left_sum <= x <= right_sum

    return r


def initialize(
        P: List[Vectors.dense],
        WP: List[int], k: int
) -> (
        List[Vectors.dense]
):
    P_and_WP = [P, WP]
    bounded = list(zip(*P_and_WP))
    random.shuffle(bounded)
    # create shuffled copies of P and WP so we can pop the first element easier
    cP, cWP = list(zip(*bounded))
    cP = list(cP)
    cWP = list(cWP)

    S = [cP.pop()]  # picking last element for S, since P is now shuffled so last element is a random one
    for _ in range(k)[1:]:
        r = select_c(S, cP, cWP)
        S.append(cP.pop(r))
    # C, WC, _ = partition_distances(cP, S, WP, partition=True)
    # return S, C, WC
    return S


def centroid(P: List[Vectors.dense], WP: List[Vectors.dense]) -> Vectors.dense:
    """
    Calculates the perfect centroid coordinates
    :param P:
    :return:
    """
    lenP = len(P)
    summa = Vectors.dense([0 for _ in range(len(P[0]))])
    for i, p in enumerate(P):
        summa += WP[i]*p
    c_opt = summa/(sum(WP)*lenP)
    return c_opt


def kmeansPP(P: List[Vectors.dense], WP: List[int], K: int, iterations: int) -> List[Vectors.dense]:
    # S, C, WC = initialize(P, [1 for _ in range(len(P))], K)
    S = initialize(P, [1 for _ in range(len(P))], K)
    # C = None
    for iter in range(iterations):
        C, WC, _ = partition_distances(P, S, WP, partition=True)
        S_new = []
        for i, c in enumerate(C):
            S_new.append(centroid(c, WC[i]))
        S = S_new
        # C, WC, _ = partition_distances(P, S, WP, partition=True)

    return S


def KmeansObj(P: List[Vectors.dense], S: List[Vectors.dense]) -> float:
    """

    :param P: Points of dataset
    :param S: list of centroids
    :return: average distance of a point from its centroid center
    """
    WP = [1 for _ in P]
    return sum(partition_distances(P, S, WP, partition=False)[2])/len(P)


if __name__ == '__main__':
    # sc = conf_spark_env()
    path, k, iterations = argparser()
    coords = readVectorsSeq(path)
    S = kmeansPP(coords, [1 for i in range(len(coords))], 10, 1)
    print(S)
    print(KmeansObj(coords, S))

