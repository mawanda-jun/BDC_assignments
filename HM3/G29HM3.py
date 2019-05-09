from pyspark import SparkContext, SparkConf
from time import time
import argparse
from operator import add
from random import random
import os

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
iter = int(args['iter'])

# defining Spark context
spark_conf = SparkConf(True).setAppName('G29HM3').setMaster('local')
sc = SparkContext(conf=spark_conf)


if __name__ == '__main__':
    print('ciao')