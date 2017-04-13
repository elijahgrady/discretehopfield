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
def readFile(filename, option):
    # OPEN THE FILE
    if DEBUG & ~option:
        f = open('train.txt', 'r')
    elif DEBUG & option:
        f = open('test.txt', 'r')
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
        in_file = raw_input("waz hoppin ... enter training file : ")
    else:
        in_file = None
    # INIT THE TRAINING DATA
    training_data = readFile(in_file, False)
    # SEND TRAINING DATA TO MAIN FOR WEIGHT MATRIX CONSTRUCTION
    return training_data


def testing():
    # CHECK FOR DEBUG OVERRIDE
    if not DEBUG:
        in_file = raw_input("waz hoppin ... enter testing file : ")
        print("\n")
    else:
        in_file = None
    testing_data = readFile(in_file, True)
    return testing_data


def ttflag():
    ttflag = raw_input("waz hoppin ... enter 1 to train, 2 to test, anything else to quit : ")
    print("\n")
    return ttflag


def saveweightmatrix(weightmatrix):
    weightmatrixpath = raw_input("waz hoppin ... enter the name of the file to save the trained weight matrix to : ")
    f = open(weightmatrixpath, "a+")
    for x in range(weightmatrix.__len__()):
        for y in range(weightmatrix[0].__len__()):
            f.write(str(weightmatrix[x][y]) + "\n")
    f.close()
    return None


def savetestingresults(testingcontainer):
    resflag = raw_input("waz hoppin ... enter the name of the file to save the results of testing to : ")
    # OUTPUT THE TESTING IN THE SAME FORMAT AS THE INPUT DATA
    # DO THE WORK HERE
    # NO NEED TO RETURN ANY OBJECTS
    return None


def epochs():
    epochs = raw_input("waz hoppin ... enter the max number of epochs : ")
    return int(epochs)


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

def read_weight_matrix(dimensions):
    # INIT MATRIX TO RETURN
    testingweightmatrix = [[0 for x in range(dimensions)] for y in range(dimensions)]
    # PROMPT USER FOR PATH TO WEIGHT MATRIX
    filepath = raw_input("waz hoppin ... enter the filename of a saved weight matrix : ")
    # OPEN THE FILE
    f = open(filepath, "a+")
    for x in range(dimensions):
        for y in range(dimensions):
            testingweightmatrix[x][y] = int(f.readline())
    return testingweightmatrix


def converged():
    # IF NO CHANGE
    return False

class HopfieldNeuron:
    def __init__(self):
        pass

class HopfieldNet:
    def __init__(self):
        pass

# INPUT: YIN {int}*
# OUTPUT: ACTIVATED Y {-1, 0, 1}*
def activation(yin):
    if yin < 0:
        y = -1 # y = f(yin) = -1
    if yin == 0:
        y = yin # f(yin) = y
    if yin > 0:
        y = 1 # y = f(yin) = 1
    return y

def hopfield_testing_algorithm(testingcontainer, weightmatrix, maxepochs):
    # MAKE A COPY OF THE TESTING DATA TO MANIPULATE
    T = testingcontainer[:]
    # ASSUME THE NET HAS BEEN TRAINED WITH WEIGHT MATRIX W
    W = weightmatrix[:]
    # INIT A CURRENT EPOCH COUNTER FOR LOOPING
    thisepoch = 1
    # RUN UNTIL MAX EPOCHS IS REACHED
    while thisepoch <= maxepochs:
        # OR IF CONVERGENCE IS REACHED
        while converged() is not True:
            # INCREMENT EPOCH COUNTER
            thisepoch += 1
            # INITIALIZE THE NEURONS
    """
    2. for a given test pattern x do
    3. { set yi = xi, i = 1, 2, … n
       do step 4 (randomly for each neuron i)
    4. yin_i = xi + [y1w1i + y2w2i +…+ ynwni]
       yi = f (yin_i), i = 1, 2, … n
       broadcast yi to all other neurons
    5. If (no changes on activation) then converged else set xi = yi and goto step 3 }


    """
    return T


def main():
    print ("\n")
    # TRAINING
    testortrain = ttflag()
    while testortrain == '1' :
        # READ IN THE INPUT FILE AND CONVERT TO BINARY
        matrixcontainer = training()
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
        print ("\n")
        # SAVE THE WEIGHT MATRIX TO COMPLETE TRAINING
        saveweightmatrix(weightmatrix)
        print ("weights have been saved")
        # TRAINING COMPLETE
        # PROMPT USER TO TRAIN, TEST, OR QUIT
        testortrain = ttflag()
        # LOOP IF 1, JUMP TO TESTING IF 2, QUIT FOR ALL ELSE

    # TESTING
    while testortrain == '2':
        # READ THE DATA FROM THE TESTING FILE AS 3D MATRIX
        # CONVERT TO BIPOLAR
        testingcontainer = testing()
        # PROMPT THE USER TO GIVE US THE FILE WHERE THE TRAINING DATA IS SAVED
        testingweightmatrix = read_weight_matrix(testingcontainer[0].__len__())
        # PRINT TO SCREEN THE MATRIX READ IN
        print ("The weight matrix below has been initialized for testing")
        print (testingweightmatrix)
        # PROMPT USER TO GIVE THE MAXIMUM NUMBER OF EPOCHS
        maxepochs = epochs()
        print ("Testing ... ")
        # FEED ALL THE DATA TO THE TESTING ALGORITHM
        patterns = hopfield_testing_algorithm(testingcontainer, testingweightmatrix, maxepochs)
        # SAVE RESULTS AND OUTPUT TO FILE
        print ("Tested")
        savetestingresults(patterns)
        # PROMPT USER FOR LOOP
        testortrain = ttflag()
        # TRAIN IF 1, TESTING IF 2, QUIT FOR ALL ELSE

if __name__ == '__main__':
    main()
