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


def partition(P: List[Vectors.dense], S: List[Vectors.dense]) -> List[List[Vectors.dense]]:
    """
    Partitions P in S different clusters.
    :param P: List of Vectors.dense points
    :param S: List of Vectors.dense centroids that has been provided, one for cluster
    :return: Cluster list of list of Vectors.dense, divided per clusters
    """
    cP = [*P]
    clusters = list(map(lambda x: [x], S))
    # TODO: check if it is possible to remove this costy call
    cP.remove(S)

    for p in cP:
        min_dist = -inf
        r = -1
        for i, s in enumerate(S):
            temp = Vectors.squared_distance(p, s)
            if temp < min_dist:
                r = i
        clusters[r].append(p)
    return clusters


def select_c(
        S: List[Vectors.dense],
        P_minus_S: List[Vectors.dense],
        wp: List[int]) -> int:
    """
    Select the right index of P_minus_S to be returned by the algorithm in page 16 of Clustering-2-1819.pdf file
    :param S: List of centroids already found
    :param P_minus_S: P\S
    :param wp: weights of P_minus_one
    :return:
    """

    # Creating sum_of_distances: we need to take the nearest point of S per each point of P\S.
    # We calculate the value to use it afterward.
    sum_of_distances = 0
    for i, p in enumerate(P_minus_S):
        d_q = inf
        for s in S:
            temp = Vectors.squared_distance(p, s)
            if temp < d_q:
                d_q = temp
        sum_of_distances += wp[i]*sqrt(d_q)

    # Creating probability distribution
    pis = []
    for i, p in enumerate(P_minus_S):
        d_p = inf
        for s in S:
            # TODO: extract a list from the same iteration above and just iter over weights, not to calculate them again
            temp = Vectors.squared_distance(p, s)
            if temp < d_p:
                d_p = temp
        #  pis keeps the same order of P_minus_S,
        #  so then we can use the same index to be returned and used to select P[r]
        pis.append(wp[i] * sqrt(d_p) / sum_of_distances)

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


def kmPP(P: List[Vectors.dense], WP: List[int], k: int) -> List[Vectors.dense]:
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
    return S


# def kmeansPP(P: List[Vectors.dense], WP: List[int], K: int, iterations: int) -> List[Vectors.dense]:



if __name__ == '__main__':
    # sc = conf_spark_env()
    path, k, iterations = argparser()
    coords = readVectorsSeq(path)
    # print(kmeansPP(coords, [1 for i in range(len(coords))], 10, 1))

