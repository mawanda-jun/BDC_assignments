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
    """
    Initial settings to accept incoming dataset and k number of partitions
    :return: path/to/data_file, k number of clusters, iterations for Lloyd algorithm
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', help='path/to/file.txt', required=True)
    parser.add_argument('-k', '--n_of_clusters', help='number of clusters', required=True)
    parser.add_argument('-iter', '--n_of_iterations', help='number of iterations', required=True)
    args = vars(parser.parse_args())

    path = os.path.join(os.getcwd(), args['filename'])
    if not os.path.isfile(path):
        raise EnvironmentError('{} does not exists. Please double check full path.'.format(path))
    k = int(args['n_of_clusters'])
    iterations = int(args['n_of_iterations'])
    return path, k, iterations


def conf_spark_env() -> SparkContext:
    """
    Specify spark configuration. In this homework it is not needed though.
    :return: SparkContext
    """
    # defining Spark context
    spark_conf = SparkConf(True).setAppName('G29HM3').setMaster('local')
    return SparkContext(conf=spark_conf)


def partition(
        P: List[Vectors.dense],
        S: List[Vectors.dense],
        WP: List[int],
        calc_dist: bool = True
) -> (
        List[List[Vectors.dense]],
        List[List[int]],
        float
):
    """
    Partitions P in S different clusters.
    :param P: List of Vectors.dense points
    :param S: List of Vectors.dense centroids that has been provided, one for cluster
    :param WP: weights of each point separated in lists
    :param calc_dist: boolean to calculate the minimum distance of each point from its nearest centroid
    :return: Vectors.dense of points, divided per clusters; list of the weights for each point in each cluster;
             average distance of each point from its closest centroid
    """
    # initialize clusters, weights and distances lists
    clusters = list(map(lambda x: [x], S))
    # giving weight 1 because centroids does not have weight
    weights = [[1] for _ in clusters]
    distances = [inf for _ in P]

    for p_idx, p in enumerate(P):
        min_dist = inf
        r = -1
        for i, s in enumerate(S):
            # check to which centroid the point is nearer to
            temp = WP[i]*sqrt(Vectors.squared_distance(p, s))
            if temp < min_dist:
                r = i
                min_dist = temp
        if calc_dist:
            # remember calculated distance
            distances[p_idx] = min_dist
        clusters[r].append(p)
        weights[r].append(WP[p_idx])
    return clusters, weights, distances


def update_distances(
        P: List[Vectors.dense],
        S: List[Vectors.dense],
        wp: List[int],
        distances: List[float]
) -> (
        List[float]
):
    """
    Updated distance wrt the last appended element in S
    :param P: List of points
    :param S: List of centroids. S[-1] is the last one, to which the list of distances is not updated yet
    :param wp: weight of each point
    :param distances: list of distances that has already been calculated until S[-1] element
    :return: list of distances updated, each one referred to a point to its closest centroid
    """
    for i, p in enumerate(P):
        temp = wp[i] * sqrt(Vectors.squared_distance(p, S[-1]))
        if temp < distances[i]:
            distances[i] = temp

    return distances


def select_c(
        S: List[Vectors.dense],
        P: List[Vectors.dense],
        wp: List[int],
        distances: List[float]
) -> (
        int,
        List[float]
):
    """
    Select the right index of P_minus_S to be returned by the algorithm in page 16 of Clustering-2-1819.pdf file
    :param S: List of centroids already found
    :param P: List of points
    :param wp: weights of P
    :param distances: list of distances that has already been calculated until S[-1] element
    :return: r is the index of new centroid; distances is the list of distances updated
    """
    # update distances before start
    distances = update_distances(P, S, wp, distances)

    sum_of_distances = sum(distances)
    # list of probabilities for non-center points of being chosen as next center
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

    return r, distances


def initialize(
        P: List[Vectors.dense],
        WP: List[int],
        k: int
) -> (
        List[Vectors.dense],
        List[float]
):
    """
    ASSIGNMENT 1.1
    Smart initialization for Lloyd using kmeans++ algorithm.
    :param P: list of points
    :param WP: list of weights, one for each point
    :param k: number of clusters
    :return: list of centroids; list of distances that has already been calculated until S[-1] element
    """
    P_and_WP = [P, WP]
    bounded = list(zip(*P_and_WP))
    random.shuffle(bounded)
    # create shuffled copies of P and WP so we can select a random element easily
    cP, cWP = list(map(list, zip(*bounded)))

    S = [cP[-1]]  # picking last element for S, since P is now shuffled so last element is a random one
    # we are not going to delete the picked object. This behaviour is not going to modify the result until the number
    # of clusters is much lesser than the number of points.
    distances = [inf for _ in cP]
    for _ in range(k)[1:]:
        r, distances = select_c(S, cP, cWP, distances)
        S.append(cP[r])

    return S, distances


def centroid(
        P: List[Vectors.dense],
        WP: List[Vectors.dense]
) -> (
        Vectors.dense
):
    """
    Calculates the centroid coordinates
    :param P: points of one cluster
    :return: coordinates of the centroid
    """
    summa = Vectors.dense([0 for _ in range(len(P[0]))])

    for i, p in enumerate(P):
        summa += WP[i]*p
    c_opt = summa/(sum(WP))
    return c_opt


def kmeansPP(
        P: List[Vectors.dense],
        WP: List[int],
        K: int,
        iterations: int
) -> (
        List[Vectors.dense]
):
    """
    ASSIGNMENT 1.2. Iterating with Lloyd algorithm.
    :param P: list of points to be divided into clusters
    :param WP: weights, one for each point
    :param K: number of clusters
    :param iterations: number of iterations
    :return: list of clusters centroids
    """
    # initialization of clusters points and relative centroids
    S, _ = initialize(P, [1 for _ in range(len(P))], K)

    # iterations of Lloyd algorithm
    for iter in range(iterations):
        C, WC, _ = partition(P, S, WP, calc_dist=False)
        S_new = []
        for i, c in enumerate(C):
            S_new.append(centroid(c, WC[i]))
        S = S_new

    return S


def KmeansObj(P: List[Vectors.dense], S: List[Vectors.dense]) -> float:
    """
    ASSIGNMENT 2
    Calculate average distance from each point to its cluster relative centroid
    :param P: Points of dataset
    :param S: list of centroids
    :return: average distance from each point to its cluster relative centroid
    """
    # setting all weights to 1 as is required by the assignment
    WP = [1 for _ in P]
    return sum(partition(P, S, WP)[2]) / len(P)


if __name__ == '__main__':
    path, k, iterations = argparser()
    # import point coordinates from the dataset
    coords = readVectorsSeq(path)
    start = time.time()
    S = kmeansPP(coords, [1 for i in range(len(coords))], k, iterations)
    # print(S)
    avg_dist = KmeansObj(coords, S)
    end = time.time()
    print('k={k}, iter={i}, time={s:.3f}s : [{d:.2f}]'.format(
        d=avg_dist,
        s=end-start,
        k=k,
        i=iterations))
