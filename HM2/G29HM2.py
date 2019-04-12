from pyspark import SparkContext, SparkConf
from time import time
import argparse
from operator import add
from random import random
import os

# initial settings to accept incoming dataset and k number of partitions
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_path', help='path/to/file.txt', required=True)
parser.add_argument('-k', '--n_of_partitions', help='number of partitions', required=True)
args = vars(parser.parse_args())

path = os.path.join(os.getcwd(), args['file_path'])
if not os.path.isfile(path):
    raise EnvironmentError('Path/to/file.txt is not right')
K = int(args['k'])

# defining Spark context
spark_conf = SparkConf(True).setAppName('G29HM2').setMaster('local')
sc = SparkContext(conf=spark_conf)


def word_count_per_doc(document):
    """
    scan the document and add the new word in the dictionary if it's a new one,
    otherwise increment the count of that word by one
    :param document:
    :return:
    """
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
    because the groupBy method takes care of the link, however it is needed because of the method signature.
    K is general parameter for the number of partitions
    :param couple: couple (word, count) per document
    :return: random int in range [0, K)
    """
    return int(K*random())


def gather_pairs(x):
    """
    <gather_pairs> receives the pair (random_key, iterator) and iterates over the iterator to find and count how many
    time distinct words occurs. It then generates a new stream of couples (word, count)
    :param x:   x[0] is rand_key is the key with which we partitioned the data (our 'x' of the slide)
                x[1] is couples, a list of the couples that are present
    :return: yield a new couple with a word and its count in <x>
    """
    couples = x[1]
    new_couples = {}
    for word, count in couples:
        if word in new_couples.keys():
            new_couples[word] += count
        else:
            new_couples[word] = count
    for word, count in new_couples.items():
        yield word, count


def gather_pairs_partitions(couples):
    """
    This method has the same behaviour of <gather_pairs>, but it receives an iterator over which we iterate as above.
    :param couples: iterator made up of couples (word, count).
    :return: a stream of couples in which every word is unique, and its count.
    """
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


# ASSIGNMENT 1
def word_count_1():
    """
    Assignment 1: the Improved Word count 1 algorithm described in class the using reduceByKey method.
    flatMap produce the new RDD applying word_count_per_doc function.
    The reduceByKey is the transformation that aggregate data corresponding to the key (word) with the help
    of an associative reduce function (add).
    The count function count the number of distinct word in all documents.
    :return: a count of the distinct words in documents
    """
    wc_in_doc = docs\
        .flatMap(word_count_per_doc)\
        .reduceByKey(add)\
        .count()
    return wc_in_doc


# ASSIGNMENT 2.1
def word_count_2():
    """
    assignment 2.1: the Improved Word count 2 algorithm where random keys take K possible values, where K is the value
    given in input.
    -   flatMap produce the new RDD applying word_count_per_doc function.
    -   the groupBy aggregate data using a random number created using the function "assign_random_key".
    -   flatMap produces a new stream of RDD objects made up of (word, count), where <word> are unique per each partition.
    -   The reduceByKey is the transformation that aggregate data corresponding to the key (word) with the help
        of an associative reduce function (add).
    -   the count function count the number of distinct word in all documents.
    :return: the count of distinct words
    """
    wc_in_doc = docs\
        .flatMap(word_count_per_doc)\
        .groupBy(assign_random_keys)\
        .flatMap(gather_pairs)\
        .reduceByKey(add)\
        .count()
    return wc_in_doc


# ASSIGMENT 2.2
def word_count_2_with_partition():
    """
    assignment 2.2: the Improved Word count 2 algorithm that exploits the subdivision of docs into K parts and access
    each partition separately.
    -   flatMap produce the new RDD applying word_count_per_doc function.
    -   flatMap produces a new stream of RDD objects made up of (word, count), where <word> are unique per each partition.
    -   The reduceByKey is the transformation that aggregate data corresponding to the key (word) with the help
    -   of an associative reduce function (add).
    We decided not to return any count but the RDD because we wanted to keep it to obtain the average length of words
    as required in the assignment 3
    :return: a RDD object
    """
    wc_in_doc = docs\
        .flatMap(word_count_per_doc)\
        .mapPartitions(gather_pairs_partitions)\
        .reduceByKey(add)\

    return wc_in_doc

# ASSIGNMENT 3 IS IN THE <print_stats> method


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

    # ASSIGNMENT 3: calculate the average length of distinct words in all the documents
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
