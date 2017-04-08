# coding=utf-8
import math
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

def readFile(filename):
    f = open(filename, 'r')
    # THE FIRST LINE IS AN EMPTY LINE
    f.readline()
    input_dim = [int(s) for s in f.readline().split() if s.isdigit()]
    input_dim_sqrt = [int(math.sqrt(input_dim[0]))]
    print (input_dim_sqrt[0])
    output_dim = [int(s) for s in f.readline().split() if s.isdigit()]
    print (output_dim[0])
    # SKIP THE NEXT EMPTY LINE
    x = f.readline()
    # INIT DATA MEMORY AS DICT
    input_data = []
    while(f.readline()):
        line = f.readline()
        line = line.strip('\n')
        for z in line:
            if z == 'O':
                input_data.append(1)
            else:
                input_data.append(-1)
    print (input_data)


def main():
    in_file = raw_input("Enter training file filename:")
    print ("This")
    readFile(in_file)

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