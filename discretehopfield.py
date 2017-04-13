# coding=utf-8
# alex cameron & eli grady
# neural nets 2017 USD

import random
import operator
import sys
import unittest
import math
from itertools import *

"""
Discrete Hopfield Auto Net
A well-known iterative auto-associative net (1982)
Features
Activation function is discrete
Symmetric weights,  wij = wji
No self-connections,  wii = 0
Asynchronous activation updating
Only one neuron updates at a time and the update is broad-casted to all other neurons
Continuously taking input signals x
Guaranteed to converge in theory
"""

# SAVES TIME TYPING
DEBUG = True

# INPUT PARAM: FILE PATH TO DATA
# RETURN: LIST OF X*Z*Z [[[{-1,1}*]]]
# FUNCTIONS PROPERLY FOR BOTH TRAINING AND TESTING
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
    # RETURN THE COMPILED LIST OF MATRICES
    return matrixcontainer


def training():
    # CHECK FOR DEBUG OVERRIDE
    if not DEBUG:
        in_file = raw_input("Enter training file filename:")
    else:
        in_file = None
    # INIT THE TRAINING DATA
    training_data = readFile(in_file)
    # SEND TRAINING DATA TO MAIN FOR WEIGHT MATRIX CONSTRUCTION
    return training_data


def testing():
    # CHECK FOR DEBUG OVERRIDE
    return None


def ttflag():
    ttflag = raw_input("waz hoppin ... enter 1 to train, 2 to test, anything else to quit : ")
    return ttflag


def saveweightmatrix():
    swmflag = raw_input("waz hoppin ... enter the name of the file to save the trained weight matrix to : ")
    return swmflag


def saveresults():
    resflag = raw_input("waz hoppin ... enter the name of the file to save the results of testing to : ")
    return saveresults()


def epochs():
    epochs = raw_input("waz hoppin ... enter the max number of epochs : ")
    return epochs()


def transpose_matrices(matrixcontainer):
    return list(map(list, zip(*matrixcontainer)))


def add_matrices(c, d):
    return [[a+b for a, b in izip(row1, row2)] for row1, row2 in izip(c, d)]


def no_self_connections(weightmatrix):
    # SET DIAGONALS TO ZERO
    count = weightmatrix.__len__()
    for x in range(0, count, 1):
        weightmatrix[x][x] = 0
    return weightmatrix


def converged():
    # IF NO CHANGE OR MAX EPOCHS
    return False


def main():
    # TRAINING
    if (ttflag() == '1'):
        # READ IN THE INPUT FILE AND CONVERT TO BINARY
        matrixcontainer = training()
        # PROMPT THE USER FOR THE PATH TO SAVE THE WEIGHT MATRIX
        weightmatrixpath = saveweightmatrix()
        # TRANSPOSE THE MATRICES FOR WEIGHT MATRIX CONSTRUCTION
        transposedcontainer = []
        # SET LOOP COUNTER
        looper = matrixcontainer.__len__()
        # CREATE TEMP MATRIX FOR CONVERSIONS
        t_c = None
        # LOOP THROUGH MATRIX CONTAINER AND TRANSPOSE EACH N*N MATRIX
        for x in range(0, looper, 1):
            t_c = transpose_matrices(matrixcontainer[x])
            # APPEND TO CONTAINER
            transposedcontainer.append(t_c)

        '''
        print("BEFORE TRANSPOSING:")
        print (matrixcontainer)
        print ("AFTER TRANSPOSING:")
        print (transposedcontainer)


        # ^^^ TRANSPOSE IS EITHER NOT WORKING OR THE DATA IS JUST SYMMETRIC ^^^
        # ^^^ THE TRANSPOSED MATRIX APPEARS TO BE THE SAME AS THE ORIGINAL BUT THE METHOD WORKS ^^^
        # ^^^ THE LINES BELOW PROVE THE TRANSPOSE METHOD WORKS ^^^

        testmatrix = [[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]

        testmatrix = transpose_matrices(testmatrix)

        print (" THIS PROVES THE TRANSPOSE METHOD WORKS")

        print(testmatrix)

        # WILL PRINT A CORRECTLY TRANSPOSED MATRIX

        '''

        # ADD THE MATRICES TOGETHER FOR WEIGHT MATRIX CONSTRUCTION
        weightcontainer = []
        w_c = None
        for x in range(0, looper, 1):
            w_c = add_matrices(transposedcontainer[x], matrixcontainer[x])
            weightcontainer.append(w_c)
        # LOOP THROUGH N MATRICES TO BUILD THE FINAL COMPILED WEIGHT MATRIX
        for x in range(1, looper, 1):
            weightcontainer[0] = add_matrices(weightcontainer[0], weightcontainer[x])
        weightmatrix = weightcontainer[0]
        # SET SELF CONNECTIONS OF WEIGHT MATRIX TO 0
        weightmatrix = no_self_connections(weightmatrix)
        # OUTPUT AND SAVE THE FINAL WEIGHT MATRIX
        print ("testing complete, below is the weight matrix: ")
        print (weightmatrix)
        # SAVE THE WEIGHT MATRIX TO COMPLETE TRAINING
        f = open(weightmatrixpath, "a+")
        f.write(str(weightmatrix))
        f.close()
        # TRAINING COMPLETE

    # TESTING
    else:
        matrixcontainer = testing()
        # PROMPT THE USER TO GIVE US THE FILE WHERE THE TRAINING DATA IS SAVED
        # PROMPT THE USER TO GIVE US THE FILE WHERE THE TESTING DATA IS SAVED
        # RETURN THE RESULTS OF THE TESTING


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