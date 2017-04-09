# coding=utf-8

import random
import operator
import sys
import unittest
import math
from itertools import izip

"""
Discrete Hopfield Auto Net
A well-known iterative auto-associative net (1982)
Features
Activation function is discrete
Symmetric weights,  wij = wji
No self-connections,  wii = 0
Asynchronous activation updating
only one neuron updates at a time and the update is broad-casted to all other neurons
Continuously taking input signals x
Guaranteed to converge in theory
"""

DEBUG = True

# INPUT PARAM: FILENAME
# RETURN: LIST OF INPUT * OUTPUT {-1,1}
def readFile(filename):
    # OPEN THE FILE
    if DEBUG:
        f = open('train.txt','r')
    else:
        f = open(filename, 'r')
    # THE FIRST LINE IS AN EMPTY LINE
    f.readline()
    # THE NEXT LINE NEEDS TO BE READ IN AS THE INPUT DIMENSIONS
    input_dim = [int(s) for s in f.readline().split() if s.isdigit()]
    # THE NEXT LINE IN THE FILE NEEDS TO BE READ IN AS THE OUTPUT DIMENSIONS
    output_dim = [int(s) for s in f.readline().split() if s.isdigit()]
    # INIT DATA MEMORY AS A SIMPLE LIST
    matrixline = list()
    matrixbody = list()
    matrixcontainer = list()
    # WHILE READLINE WILL PARSE THROUGH ALL THE REST OFF THE LINES IN THE FILE
    f.readline()
    for x in range (0, output_dim[0]):
        for x in range(0, 10):
            # READ THE ACTUAL LINE
            line = f.readline()
            # STRIP ALL OF THE NEWLINE CHARS
            line = line.strip('\n')
            # ITERATE THROUGH ALL CHARS IN EVERY LINE
            for z in line:
                # CHECK FOR A 'O'
                if z == 'O':
                    # CONVERT 'O' TO 1
                    matrixline.append(1)
                # ELSE WE READ A SPACE
                elif z == ' ':
                    # CONVERT SPACE TO -1
                    matrixline.append(-1)
            matrixbody.append(matrixline)
            # RESET LINE
            matrixline = []
        # SKIP THE NEXT EMPTY LINE
        f.readline()
        # ADD TO CONTAINER
        matrixcontainer.append(matrixbody)
        # RESET BODY
        matrixbody = []
    return matrixcontainer


def training():
    if not DEBUG:
        in_file = raw_input("Enter training file filename:")
    else:
        in_file = None
    training_data = readFile(in_file)
    return training_data


def testing():
    return None


def ttflag():
    ttflag = raw_input("whats hoppin? enter 1 to train, anything else to test : ")
    return ttflag


def transpose(matrix):
    """
    transposed = []
    for i in range(4):
...     # the following 3 lines implement the nested listcomp
...     transposed_row = []
...     for row in matrix:
...         transposed_row.append(row[i])
...     transposed.append(transposed_row)
    """
    # SIMPLE ZIP AND UNPACK
    return zip(*matrix)


def transpose_matrices(matrixcontainer):
    # INIT RETURN VAR
    # NUMBER TO INIT
    count = matrixcontainer.__len__()
    # CONTAINER FOR THE TRANSPOSED MATRICES
    transposed_container = list()
    # LOOP FOR CONTAINER.LEN
    for z in range(0, count):
        # FEED THE TRANSPOSE FUNCTION AND APPEND TO CONTAINER
        args = matrixcontainer[z]
        print (args)
        print ('\n')
        print zip(*args)
        print ("new")

    # RETURN WEIGHT MATRIX
    return transposed_container


def add_matrices(c, d):
    return [[a+b for a, b in izip(row1, row2)] for row1, row2 in izip(c, d)]


def no_self_connections(weight_matrix):
    # SET DIAGONALS TO ZERO
    count = weight_matrix.__len__()
    count2 = weight_matrix[0].__len__()
    for x in range(0, count):
        for x in range(0, count2):
            # SET DIAGONALS TO ZERO HERE
            weight_matrix[count2][count2] == 0
    return weight_matrix

def converged():
    # IF NO CHANGE OR MAX EPOCHS
    return False


def main():
    # TRAINING
    if (ttflag() == '1'):
        matrixcontainer = training()
        transposedcontainer = transpose_matrices(matrixcontainer)
        # THE FOLLOWING LINE NEEDS TO BE LOOPED
        weight_matrix = add_matrices(transposedcontainer, matrixcontainer)
        weight_matrix = no_self_connections(weight_matrix)

    # TESTING
    else:
        matrixcontainer = testing()


if __name__ == '__main__':
    main()






"""
Testing of Hopfield Auto Net
Use same activation function as iterative auto-associative net
y = f (yin) = -1  if yin <  0
y = f (yin) = y  if yin = 0
y = f (yin) = 1 if yin > 0
"""

"""
Testing of Hopfield Auto Net
1. assume the net has all trained weights (weight matrix W)
2. for a given test pattern x do
3. { set yi = xi, i = 1, 2, … n do step 4 (randomly for each neuron i)
4. yin_i = xi + [y1w1i + y2w2i +…+ ynwni] yi = f (yin_i), i = 1, 2, … n broadcast yi to all other neurons
5. If (no changes on activation) then converged else set xi = yi and goto step 3 }
"""