from pyspark import SparkContext, SparkConf
from time import time
from operator import add

import os

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


def count_values_per_key(list_of_pairs, input_pair):
    if not list_of_pairs:
        return [input_pair]
    found = False
    for pair in list_of_pairs:
        if pair[0] == input_pair[0]:
            pair[1] += input_pair[1]
            found = True
    if not found:
        list_of_pairs.append(input_pair)
    return list_of_pairs


def main():
    K = 3  # of parts

    docs = sc.textFile(os.path.join('HM2', 'dataset.txt')).repartition(K)

    # assignment 1: the Improved Word count 1 algorithm described in class the using reduceByKey method
    wc_per_pair = docs\
        .flatMap(word_count_per_doc)\
        .reduceByKey(add)\
        .collect()
    print(wc_per_pair)


if __name__ == '__main__':
    main()
    # start = time()
    # wordcountpairs = docs.flatMap(f1).groupByKey().map(count_values_per_key)
    # print(docs.count())
    # end = time()
    # print('Elapsed time: {}'.format(end-start))

    # input('Press any key to continue...')
    # flatMap(f1) implements the Map phase
    # groupByKey().map(count_values_per_key) implements the Reduce phase
