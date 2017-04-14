import numpy as np
##
# hstack and vstack concatenate two arrays
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

input = np.array([
                [0, 0, 4, 0],
                [0, 2, 5, 0],
                [0, 0, 1, 0],
                [0, 0, 2, 0]])
s = 2
k = 2
h_out = 2
w_out = 2

temp = np.zeros([h_out, w_out])

for i in range(h_out):
    for j in range(w_out):
        temp[i, j] = np.amax(input[(i*s) : (k + i*s), (j*s) : (k + j*s)], axis=(0, 1))

print 'temp = \n', temp
