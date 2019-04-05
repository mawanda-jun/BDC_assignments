from pyspark import SparkContext, SparkConf

# Import the Dataset
lNumbers = []

with open('dataset.txt', 'r') as f:
    lines = f.readlines()
    for n in lines:
        if len(n[:-1]) > 0:
            lNumbers.append(float(n[:-1]))

print("The locally loaded list of numbers is: ", lNumbers)

# Spark Setup
conf = SparkConf().setAppName('G29HM1').setMaster('local')
sc = SparkContext(conf=conf)

# Create a parallel collection
dNumbers = sc.parallelize(lNumbers)

# calculate max value with reduce function
max_value_r = dNumbers.reduce(lambda x, y: x if x > y else y)

# calculate max value with RDD function
max_value_m = dNumbers.max()

# calculate normalized version of dNumbers
dMax = dNumbers.max()
dNormalized = dNumbers.map(lambda x: x / dMax)

# retrieve dNormalized stats
stats = dNormalized.stats()

# calculate number of values > .5
n_mag_5 = dNormalized.filter(lambda x: x > 0.5).count()

print("Max value with reduce method: {}".format(max_value_r))
print("Max value with max method: {}".format(max_value_m))
print("dNormalized, dNumber normalized version: {}".format(dNormalized.collect()))
print("N of elements: {n}\nMean: {m}\nStDev: {sd}\nMax: {max}\nMin: {min}".format(
    n=stats.n, m=stats.m2, sd=stats.mu, max=stats.maxValue, min=stats.minValue
))
print('N of elements > .5: {}'.format(n_mag_5))
