def f1(document):
    pairs_dict = {}  # dictionary to store (key, value) pairs
    for word in document.split(' '):
        if word not in pairs_dict.keys():
            pairs_dict[word] = 1
        else:
            pairs_dict[word] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]


def f2(pair):
    word, occurrences = pair[0], list(pair[1])
    sum_o = 0
    for o in occurrences:
        sum_o += o
    return (word, sum_o)


# flatMap(f1) implements the Map phase
# groupByKey().map(count_values_per_key) implements the Reduce phase
wordcountpairs = docs.flatMap(f1).groupByKey().map(f2)
