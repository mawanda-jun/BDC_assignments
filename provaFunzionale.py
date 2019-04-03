import timeit

arr = range(100000)


def lambdas():
    return list(map(lambda x: x * x, map(lambda x: x * x, arr)))


def comprehension():
    return [x * x for x in [x * x for x in arr]]


if __name__ == '__main__':
    print('Time for lambdas: {l}\nTime for list comprehension: {c}'.format(
        l=timeit.timeit(list(map(lambda x: x * x, map(lambda x: x * x, arr))), number=100000, globals=globals()),
        c=timeit.timeit([x * x for x in [x * x for x in arr]], number=100000, globals=globals())
    ))