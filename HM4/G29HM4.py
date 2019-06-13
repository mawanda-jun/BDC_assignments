import random
import sys
import math
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vectors
from functools import partial
from typing import List
from timeit import default_timer as timer


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
    distances = [math.inf for _ in P]

    for p_idx, p in enumerate(P):
        min_dist = math.inf
        r = -1
        for i, s in enumerate(S):
            # check to which centroid the point is nearer to
            temp = WP[i]*math.sqrt(Vectors.squared_distance(p, s))
            # temp = WP[i]*np.linalg.norm(p-s)
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
        temp = wp[i] * math.sqrt(Vectors.squared_distance(p, S[-1]))
        # temp = wp[i] * np.linalg.norm(p - S[-1])
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
    distances = [math.inf for _ in cP]
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


# --------------------------------------------------------------
# --------------------------------------------------------------

def smallest_distance(el: Vectors.dense, centers: List[Vectors.dense]):
    min = math.inf
    for center in centers:
        temp = math.sqrt(Vectors.squared_distance(el, center))
        # temp = np.linalg.norm(el - center)
        if temp < min:
            min = temp
    return min


def compute_weights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        # mindist = math.sqrt(point.squared_distance(centers[0]))
        mindist = np.linalg.norm(point - centers[0])
        for i in range(1, len(centers)):
            # if math.sqrt(point.squared_distance(centers[i])) < mindist:
                # mindist = math.sqrt(point.squared_distance(centers[i]))
            # temp = np.linalg.norm(centers[i] - point)
            temp = math.sqrt(point.squared_distance(centers[i]))
            if temp < mindist:
                mindist = temp
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights


def f2(k, L, iterations, partition):
    points = [vector for vector in iter(partition)]
    weights = np.ones(len(points))
    centers = kmeansPP(points, weights, k, iterations)
    final_weights = compute_weights(points, centers)
    return [(vect, weight) for vect, weight in zip(centers, final_weights)]


def rdd_iterate(rdd, chunk_size=1000000):
    indexed_rows = rdd.zipWithIndex().cache()
    count = indexed_rows.count()
    print("Will iterate through RDD of count {}".format(count))
    start = 0
    end = start + chunk_size
    while start < count:
        print("Grabbing new chunk: start = {}, end = {}".format(start, end))
        chunk = indexed_rows.filter(lambda r: r[1] >= start and r[1] < end).collect()
        for row in chunk:
            yield row[0]
        start = end
        end = start + chunk_size


def MR_kmedian(pointset, k, L, iterations):
    times = np.empty(3)
    # ---------- ROUND 1 ---------------
    start = timer()
    coreset = pointset.mapPartitions(partial(f2, k, L, iterations))
    coreset.count()
    end = timer()
    times[0] = end - start
    # ---------- ROUND 2 ---------------
    start = timer()
    centersR1 = []
    weightsR1 = []
    for pair in coreset.collect():
    # for pair in rdd_iterate(coreset):
        centersR1.append(pair[0])
        weightsR1.append(pair[1])
    centers = kmeansPP(centersR1, weightsR1, k, iterations)
    end = timer()
    times[1] = end - start
    # ---------- ROUND 3 --------------------------
    start = timer()
    len = pointset.count()
    sum = pointset.map(lambda el: smallest_distance(el, centers)).reduce(lambda x, y: x+y)
    obj = sum / len
    end = timer()
    times[2] = end - start
    return obj, times


def f1(line):
    return Vectors.dense([float(coord) for coord in line.split(" ") if len(coord) > 0])


def main(argv):
    # Avoided controls on input..
    dataset = argv[1]
    k = int(argv[2])
    L = int(argv[3])
    iterations = int(argv[4])
    conf = SparkConf().setAppName('G29HM4-python')
    sc = SparkContext(conf=conf)
    pointset = sc.textFile(dataset).map(f1).repartition(L).cache()
    N = pointset.count()
    print("Number of points is : " + str(N))
    print("Number of clusters is : " + str(k))
    print("Number of parts is : " + str(L))
    print("Number of iterations is : " + str(iterations))
    obj, times = MR_kmedian(pointset, k, L, iterations)
    print("T1 is : {}".format(times[0]))
    print("T2 is : {}".format(times[1]))
    print("T3 is : {}".format(times[2]))
    print("Objective function is : " + str(obj))


if __name__ == '__main__':
    if (len(sys.argv) != 5):
        print("Usage: <pathToFile> k L iter")
        sys.exit(0)
    main(sys.argv)

