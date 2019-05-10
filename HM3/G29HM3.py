from pyspark import SparkContext, SparkConf
import argparse
import os
from math import sqrt, inf
import random
from pyspark.mllib.linalg import Vectors
from VectorInput import readVectorsSeq
from typing import List
import time


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


def partition(P: List[Vectors.dense], S: List[Vectors.dense], WP: List[int]) -> (List[List[Vectors.dense]], List[List[int]], float):
    """
    Partitions P in S different clusters.
    :param P: List of Vectors.dense points
    :param S: List of Vectors.dense centroids that has been provided, one for cluster
    :return: Cluster list of list of Vectors.dense, divided per clusters. The first element of each cluster
    is the centroid
    """
    clusters = list(map(lambda x: [x], S))
    # giving weight 1 because centroids doens't have weight
    weights = [[1] for _ in clusters]
    # TODO: check if it is possible to remove this costy call
    # cP.remove(S)
    distances = [inf for _ in P]
    for p_idx, p in enumerate(P):
        min_dist = inf
        r = -1
        for i, s in enumerate(S):
            temp = WP[i]*sqrt(Vectors.squared_distance(p, s))
            if temp < min_dist:
                r = i
                min_dist = temp
        # assert r > -1
        distances[p_idx] = min_dist
        clusters[r].append(p)
        weights[r].append(WP[p_idx])
    return clusters, weights, distances


def update_distances(
        P: List[Vectors.dense],
        S: List[Vectors.dense],
        wp: List[int],
        distances: List[float]
) -> List[float]:
    # Creating sum_of_distances: we need to take the nearest point of S per each point of P\S.
    # We calculate the value to use it afterward.
    # defining initial cluster so we can keep it and return

    for i, p in enumerate(P):
        temp = wp[i] * sqrt(Vectors.squared_distance(p, S[-1]))
        if temp < distances[i]:
            distances[i] = temp

    # assert inf not in distances
    return distances


def select_c(
        S: List[Vectors.dense],
        P: List[Vectors.dense],
        wp: List[int],
        distances: List[float]
) -> (int, List[float]):
    """
    Select the right index of P_minus_S to be returned by the algorithm in page 16 of Clustering-2-1819.pdf file
    :param S: List of centroids already found
    :param P_minus_S: P\S
    :param wp: weights of P_minus_one
    :return: r is the index of new centroid; C and WC are the partitions with the points already found
    """
    distances = update_distances(P, S, wp, distances)

    sum_of_distances = sum(distances)
    # assert len(wp) == len(distances)

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

    # assert left_sum <= x <= right_sum

    return r, distances


def initialize(
        P: List[Vectors.dense],
        WP: List[int], k: int
) -> (
        List[Vectors.dense],
        List[float]
):
    P_and_WP = [P, WP]
    bounded = list(zip(*P_and_WP))
    random.shuffle(bounded)
    # create shuffled copies of P and WP so we can pop the first element easier
    cP, cWP = list(map(list, zip(*bounded)))
    # cP = list(cP)
    # cWP = list(cWP)

    S = [cP[-1]]  # picking last element for S, since P is now shuffled so last element is a random one
    # cWP.pop()
    distances = [inf for _ in cP]
    for _ in range(k)[1:]:
        # assert len(cP) == len(cWP)
        r, distances = select_c(S, cP, cWP, distances)
        S.append(cP[r])
        # cWP.pop(r)
    # C, WC, _ = update_distances(cP, S, WP, partition=True)
    # return S, C, WC
    return S, distances


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
    c_opt = summa/(sum(WP))
    return c_opt


def kmeansPP(P: List[Vectors.dense], WP: List[int], K: int, iterations: int) -> (List[Vectors.dense]):
    # S, C, WC = initialize(P, [1 for _ in range(len(P))], K)
    S, distances = initialize(P, [1 for _ in range(len(P))], K)
    # print('Average_distance before iteration cicle: {}'.format(sum(distances) / len(P)))
    # C = None
    for iter in range(iterations):
        C, WC, distances = partition(P, S, WP)
        # assert len(C[0]) == len(WC[0])
        S_new = []
        for i, c in enumerate(C):
            S_new.append(centroid(c, WC[i]))
        S = S_new
        # C, WC, _ = update_distances(P, S, WP, partition=True)
        # print('Average_distance in iteration cicle: {}'.format(sum(distances) / len(P)))

    return S


def KmeansObj(P: List[Vectors.dense], S: List[Vectors.dense]) -> float:
    """

    :param P: Points of dataset
    :param S: list of centroids
    :return: average distance of a point from its centroid center
    """
    WP = [1 for _ in P]
    return sum(partition(P, S, WP)[2]) / len(P)


if __name__ == '__main__':
    # sc = conf_spark_env()
    path, k, iterations = argparser()
    coords = readVectorsSeq(path)
    start = time.time()
    S = kmeansPP(coords, [1 for i in range(len(coords))], k, iterations)
    # print(S)
    avg_dist = KmeansObj(coords, S)
    end = time.time()
    print('K: {k}\nIterations: {i}\nAverage distance from centers: {d:.2f}\nFound in: {s:.4f}s'.format(
        d=avg_dist,
        s=end-start,
        k=k,
        i=iterations))

