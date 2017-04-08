# coding=utf-8

import random
import operator
import sys
import unittest
import math

class MatrixError(Exception):
    """ An exception class for Matrix """
    pass


class Matrix(object):
    """ A simple Python matrix class with
    basic operations and operator overloading """

    def __init__(self, m, n, init=True):
        if init:
            self.rows = [[0] * n for x in range(m)]
        else:
            self.rows = []
        self.m = m
        self.n = n

    def __getitem__(self, idx):
        return self.rows[idx]

    def __setitem__(self, idx, item):
        self.rows[idx] = item

    def __str__(self):
        s = '\n'.join([' '.join([str(item) for item in row]) for row in self.rows])
        return s + '\n'

    def __repr__(self):
        s = str(self.rows)
        rank = str(self.getRank())
        rep = "Matrix: \"%s\", rank: \"%s\"" % (s, rank)
        return rep

    def reset(self):
        """ Reset the matrix data """
        self.rows = [[] for x in range(self.m)]

    def transpose(self):
        """ Transpose the matrix. Changes the current matrix """

        self.m, self.n = self.n, self.m
        self.rows = [list(item) for item in zip(*self.rows)]

    def getTranspose(self):
        """ Return a transpose of the matrix without
        modifying the matrix itself """

        m, n = self.n, self.m
        mat = Matrix(m, n)
        mat.rows = [list(item) for item in zip(*self.rows)]

        return mat

    def getRank(self):
        return (self.m, self.n)

    def __eq__(self, mat):
        """ Test equality """

        return (mat.rows == self.rows)

    def __add__(self, mat):
        """ Add a matrix to this matrix and
        return the new matrix. Doesn't modify
        the current matrix """

        if self.getRank() != mat.getRank():
            raise MatrixError, "Trying to add matrices of varying rank!"

        ret = Matrix(self.m, self.n)

        for x in range(self.m):
            row = [sum(item) for item in zip(self.rows[x], mat[x])]
            ret[x] = row

        return ret

    def __sub__(self, mat):
        """ Subtract a matrix from this matrix and
        return the new matrix. Doesn't modify
        the current matrix """

        if self.getRank() != mat.getRank():
            raise MatrixError, "Trying to add matrixes of varying rank!"

        ret = Matrix(self.m, self.n)

        for x in range(self.m):
            row = [item[0] - item[1] for item in zip(self.rows[x], mat[x])]
            ret[x] = row

        return ret

    def __mul__(self, mat):
        """ Multiple a matrix with this matrix and
        return the new matrix. Doesn't modify
        the current matrix """

        matm, matn = mat.getRank()

        if (self.n != matm):
            raise MatrixError, "Matrices cannot be multipled!"

        mat_t = mat.getTranspose()
        mulmat = Matrix(self.m, matn)

        for x in range(self.m):
            for y in range(mat_t.m):
                mulmat[x][y] = sum([item[0] * item[1] for item in zip(self.rows[x], mat_t[y])])

        return mulmat

    def __iadd__(self, mat):
        """ Add a matrix to this matrix.
        This modifies the current matrix """

        # Calls __add__
        tempmat = self + mat
        self.rows = tempmat.rows[:]
        return self

    def __isub__(self, mat):
        """ Add a matrix to this matrix.
        This modifies the current matrix """

        # Calls __sub__
        tempmat = self - mat
        self.rows = tempmat.rows[:]
        return self

    def __imul__(self, mat):
        """ Add a matrix to this matrix.
        This modifies the current matrix """

        # Possibly not a proper operation
        # since this changes the current matrix
        # rank as well...

        # Calls __mul__
        tempmat = self * mat
        self.rows = tempmat.rows[:]
        self.m, self.n = tempmat.getRank()
        return self

    def save(self, filename):
        open(filename, 'w').write(str(self))

    @classmethod
    def _makeMatrix(cls, rows):

        m = len(rows)
        n = len(rows[0])
        # Validity check
        if any([len(row) != n for row in rows[1:]]):
            raise MatrixError, "inconsistent row length"
        mat = Matrix(m, n, init=False)
        mat.rows = rows

        return mat

    @classmethod
    def makeRandom(cls, m, n, low=0, high=10):
        """ Make a random matrix with elements in range (low-high) """

        obj = Matrix(m, n, init=False)
        for x in range(m):
            obj.rows.append([random.randrange(low, high) for i in range(obj.n)])

        return obj

    @classmethod
    def makeZero(cls, m, n):
        """ Make a zero-matrix of rank (mxn) """

        rows = [[0] * n for x in range(m)]
        return cls.fromList(rows)

    @classmethod
    def makeId(cls, m):
        """ Make identity matrix of rank (mxm) """

        rows = [[0] * m for x in range(m)]
        idx = 0

        for row in rows:
            row[idx] = 1
            idx += 1

        return cls.fromList(rows)

    @classmethod
    def readStdin(cls):
        """ Read a matrix from standard input """

        print 'Enter matrix row by row. Type "q" to quit'
        rows = []
        while True:
            line = sys.stdin.readline().strip()
            if line == 'q': break

            row = [int(x) for x in line.split()]
            rows.append(row)

        return cls._makeMatrix(rows)

    @classmethod
    def readGrid(cls, fname):
        """ Read a matrix from a file """

        rows = []
        for line in open(fname).readlines():
            row = [int(x) for x in line.split()]
            rows.append(row)

        return cls._makeMatrix(rows)

    @classmethod
    def fromList(cls, listoflists):
        """ Create a matrix by directly passing a list
        of lists """

        # E.g: Matrix.fromList([[1 2 3], [4,5,6], [7,8,9]])

        rows = listoflists[:]
        return cls._makeMatrix(rows)

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
    input_data = []
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
                    input_data.append(1)
                # ELSE WE READ A SPACE
                elif z == ' ':
                    # CONVERT SPACE TO -1
                    input_data.append(-1)
        # SKIP THE NEXT EMPTY LINE
        f.readline()
    return input_data


def training():
    if not DEBUG:
        in_file = raw_input("Enter training file filename:")
    else:
        in_file = None
    training_data = readFile(in_file)
    print(training_data)
    print len(training_data)


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
    return list(zip(*matrix))

def main():
    if (ttflag() == '1'):
        datalist = training()
    else:
        datalist = testing()
    matrix = Matrix.fromList(datalist)
    print (matrix)

if __name__ == '__main__':
    main()



'''
Training Hopfield Auto Net
Hebb rule,
for a given set of patterns {s1, s2, …, sp} in bipolar,
weight matrix = s1T s1 + s2T s2 + … + spT sp
wii = O diagonals
'''



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