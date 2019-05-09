from pyspark import SparkContext, SparkConf
import argparse
import os
from VectorInput import readVectorsSeq


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
    k = int(args['k'])
    iterations = int(args['iter'])
    return path, k, iterations


def conf_spark_env() -> SparkContext:
    # defining Spark context
    spark_conf = SparkConf(True).setAppName('G29HM3').setMaster('local')
    return SparkContext(conf=spark_conf)


if __name__ == '__main__':
    sc = conf_spark_env()
    path, k, iterations = argparser()