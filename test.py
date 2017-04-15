import numpy as np
##
# hstack and vstack concatenate two arrays
#
# a = np.array([[1, 2, 3],
#                 [1, 2, 3]])
# b = np.array([[4, 5, 6],
#                 [7, 8, 9]])
# print 'a = \n', a[:2, :2]
# print 'b = \n', b
# print 'b transpose: \n', b.T
# c = np.hstack((a, b))
# d = np.vstack((a, b))
# print 'c = \n', c
# print 'd = \n', d

##
# test maxpooling
#
input = np.array([
        [[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],
       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],
       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
h_in = input.shape[0]
w_in = input.shape[1]
c = input.shape[2]
# print c


s = 1
k = 2
h_out = 2
w_out = 2

temp = np.zeros([h_out, w_out, c])

for i in range(h_out):
    for j in range(w_out):
        print 'input row = %d, column = %d' % (i, j)
        print input[(i*s) : (k + i*s), (j*s) : (k + j*s), :]
        temp[i, j, :] = np.amax(input[(i*s) : (k + i*s), (j*s) : (k + j*s), :], axis=(0, 1))
        print 'temp = \n', temp[i, j, :]

print 'temp = \n', temp
print temp.shape

print 'input\n'
print "%d %d %d" % (input[0,0,0], input[0,1,0], input[0,2,0])
print "%d %d %d" % (input[1,0,0], input[1,1,0], input[1,2,0])
print "%d %d %d" % (input[2,0,0], input[2,1,0], input[2,2,0])

print 'output\n'
print "%d %d" % (temp[0,0,0], temp[0,1,0])
print "%d %d" % (temp[1,0,0], temp[1,1,0])
