from pyspark import SparkContext, SparkConf
from time import time
import argparse
from operator import add
from random import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_path', help='path/to/file.txt', required=True)
parser.add_argument('-k', '--k', help='number of partitions', required=True)
args = vars(parser.parse_args())

# K = 3  # number of partitions
path = os.path.join(os.getcwd(), args['file_path'])
if not os.path.isfile(path):
    raise EnvironmentError('Path/to/file.txt is not right')
K = int(args['k'])

spark_conf = SparkConf(True).setAppName('G29HM2').setMaster('local')
sc = SparkContext(conf=spark_conf)


def word_count_per_doc(document):
    pairs_dict = {}  # dictionary to store (key, value) pairs
    for word in document.split(' '):
        if word not in pairs_dict.keys():
            pairs_dict[word] = 1
        else:
            pairs_dict[word] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]


def assign_random_keys(couple):
    """
    Assigns a random key value in range [0, K) to couple. There is no need to return it
    because the groupBy method takes care of the link.
    K is general parameter for the number of partitions
    :param couple: couple (word, count) per document
    :return: random int in range [0, K)
    """
    return int(K*random())


def word_count_1():
    # assignment 1: the Improved Word count 1 algorithm described in class the using reduceByKey method
    # reduceByKey, da quello che ho capito, prende la lista di coppie passata da flatMap, e le raggruppa per chiave.
    # e ti chiede cosa fare dei valori delle chiavi: in questo caso dobbiamo sommarle.
    # Di conseguenza, i valori che prende "add" in input sono il valore della chiave della coppia che esiste gia' e
    # il valore della chiave della coppia che vogliamo inserire.

    # per vedere la documentazione dei metodi, pigia "CTRL" + "Q"
    wc_in_doc = docs\
        .flatMap(word_count_per_doc)\
        .reduceByKey(add)\
        .count()
    return wc_in_doc


def gather_pairs(x):
    """
    Generate a new stream as a list
    :param x:   x[0] is rand_key is the key with which we partitioned the data (our 'x' of the slide)
                x[1] is couples, a list of the couples that are present
    :return: yield a new couple with
    """
    couples = x[1]
    new_couples = {}
    # new_couples = {
    #   word: count
    # }
    for word, count in couples:
        if word in new_couples.keys():
            new_couples[word] += count
        else:
            new_couples[word] = count
    for word, count in new_couples.items():
        yield word, count


def gather_pairs_partitions(couples):
    return gather_pairs(('', couples))


def word_count_2():
    wc_in_doc = docs\
        .flatMap(word_count_per_doc)\
        .groupBy(assign_random_keys)\
        .flatMap(gather_pairs)\
        .reduceByKey(add)\
        .count()

    return wc_in_doc


def word_count_2_with_partition():
    wc_in_doc = docs\
        .flatMap(word_count_per_doc)\
        .mapPartitions(gather_pairs_partitions)\
        .reduceByKey(add)\

    return wc_in_doc


def print_stats(k=K):
    start = time()
    print(
        '\n#--------------------------------------------------#\nWord count 1 with k={k}: {c}'.format(c=word_count_1(),
                                                                                                      k=k))
    end = time()
    print('Time elapsed: {}\n#--------------------------------------------------#\n'.format(end - start))

    start = time()
    print(
        '\n#--------------------------------------------------#\nWord count 2 with k={k}: {c}'.format(c=word_count_2(),
                                                                                                      k=k))
    end = time()
    print('Time elapsed: {}\n#--------------------------------------------------#\n'.format(end - start))

    wc_partition = word_count_2_with_partition()

    start = time()
    words_in_wc_partition = wc_partition.count()
    print(
        '\n#--------------------------------------------------#\nWord count 2 with k={k} with mapPartition: {c}'.format(
            c=words_in_wc_partition,
            k=k))
    end = time()
    print('Time elapsed: {}\n#--------------------------------------------------#\n'.format(end - start))

    print('Average length of distinct words: {}'.format(
        wc_partition
        .map(lambda x: len(x[0]))
        .reduce(add) / words_in_wc_partition
    ))


if __name__ == '__main__':
    # load partitions
    docs = sc.textFile(path).repartition(K).cache()
    a = docs.count()
    print('Printing stats for k={} partitions as in input...'.format(K))
    print_stats()
    input('\nPress any key to exit...')
