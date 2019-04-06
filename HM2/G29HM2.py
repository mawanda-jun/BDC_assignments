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


def count_values_per_key(value, new_value):
    return value + new_value


def main():
    K = 3  # of parts

    docs = sc.textFile(os.path.join('HM2', 'dataset.txt')).repartition(K)

    # assignment 1: the Improved Word count 1 algorithm described in class the using reduceByKey method
    # reduceByKey, da quello che ho capito, prende la lista di coppie passata da flatMap, e le raggruppa per chiave.
    # e ti chiede cosa fare dei valori delle chiavi: in questo caso dobbiamo sommarle.
    # Di conseguenza, i valori che prende "add" in input sono il valore della chiave della coppia che esiste gia' e
    # il valore della chiave della coppia che vogliamo inserire.

    # per vedere la documentazione dei metodi, pigia "CTRL" + "Q"
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
