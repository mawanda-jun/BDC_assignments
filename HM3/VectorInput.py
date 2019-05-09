from pyspark.mllib.linalg import Vectors


def readVectorsSeq(filename):
    file = open(filename, 'r')
    vector_list = []
    for row in file.readlines():
        vector_list.append(Vectors.dense([float(num_str) for num_str in row.split()]))
    return vector_list


if __name__ == '__main__':
    vector = readVectorsSeq('covtype10K.data')
    print(vector[1])
