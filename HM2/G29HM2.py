from pyspark import SparkContext, SparkConf
from time import time
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


def count_values_per_key(pair_1, pair2_):
    word1, occurrences1 = pair_1[0], list(pair_1[1])
    word2, occurrences2 = pair_2[0], list(pair_2[1])
    sum_o = 0

    for o in occurrences1:
        sum_o += o
    return (word1, sum_o)


def main():
    K = 3  # of parts

    docs = sc.textFile(os.path.join('HM2', 'dataset.txt')).repartition(K)

    # MAP PHASE 1
    wordcountpairs = docs\
        .flatMap(word_count_per_doc)\
        .reduceByKey()
    print(wordcountpairs)


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
