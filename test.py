import numpy as np
import numpy.matlib
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

"""
 test maxpooling forward
 also can be used to test
 backward
"""
# input = np.array([
#         [[ 0,  1,  2],
#         [ 3,  4,  5],
#         [ 6,  7,  8]],
#        [[ 9, 10, 11],
#         [12, 13, 14],
#         [15, 16, 17]],
#        [[18, 19, 20],
#         [21, 22, 23],
#         [24, 25, 26]]])
# h_in = input.shape[0]
# w_in = input.shape[1]
# c = input.shape[2]
# s = 1
# k = 2
# h_out = 2
# w_out = 2
# temp = np.zeros([h_out, w_out, c])

# for i in range(h_out):
#     for j in range(w_out):
#         print 'input row = %d, column = %d' % (i, j)
#         print input[(i*s) : (k + i*s), (j*s) : (k + j*s), :]
#         temp[i, j, :] = np.amax(input[(i*s) : (k + i*s), (j*s) : (k + j*s), :], axis=(0, 1))
#         print 'temp = \n', temp[i, j, :]
#
# print 'output = \n', temp
# print temp.shape
#
# print 'input\n', input
# print '\nchannel 0:'
# print "%d %d %d" % (input[0,0,0], input[0,1,0], input[0,2,0])
# print "%d %d %d" % (input[1,0,0], input[1,1,0], input[1,2,0])
# print "%d %d %d" % (input[2,0,0], input[2,1,0], input[2,2,0])
# print '\nchannel 1:'
# print "%d %d %d" % (input[0,0,1], input[0,1,1], input[0,2,1])
# print "%d %d %d" % (input[1,0,1], input[1,1,1], input[1,2,1])
# print "%d %d %d" % (input[2,0,1], input[2,1,1], input[2,2,1])
# print '\nchannel 2:'
# print "%d %d %d" % (input[0,0,2], input[0,1,2], input[0,2,2])
# print "%d %d %d" % (input[1,0,2], input[1,1,2], input[1,2,2])
# print "%d %d %d" % (input[2,0,2], input[2,1,2], input[2,2,2])
# #
# print '\noutput\n'
# print "%d %d" % (temp[0,0,0], temp[0,1,0])
# print "%d %d" % (temp[1,0,0], temp[1,1,0])
#
# output_diff = np.array([
# [[1, 2, 3],
# [4, 5, 6]],
#
# [[7, 8, 9],
# [10, 11, 12]]
# ])
"""
 test maxpooling backward
 can use the input from forward or this
 simple input from backward
"""
# input = np.array([
#         [0, 8, 14],
#         [2, 7, 7],
#         [0, 1, 5]])
# output = np.array([
#         [8, 14],
#         [7, 7]])
# output_diff = np.array([
# [4, 4],
# [3, 3]
# ])
# h_out = output.shape[0]
# w_out = output.shape[1]
# k = 2
# s = 1
# # print c
# input_od = np.zeros(input.shape)
# temp_od = np.zeros(input.shape)
# #
# for i in range(h_out):
#     for j in range(w_out):
#         index = np.argmax(input[(i*s) : (k + i*s), (j*s) : (k + j*s), :])
#         print 'index', index
#         input_od[(i*s + (index/k)%k), (j*s + index%k), :] = output_diff[i, j, :]
#         temp_od[(i*s) : (k + i*s), (j*s) : (k + j*s)] += input[(i*s) : (k + i*s), (j*s) : (k + j*s), :] >= temp[i, j]
#
# print '\nderivatives of this layer\n', np.minimum(temp_od, 1)
# print '\nderivatives\n', input_od

"""
 test relu layer forward
"""
# x = np.array([[1, -2, 0],
#               [-3, 5, 6]])
# y = np.maximum(0, x)
# print y

"""
 test IP layer forward
"""

# input = np.array([8, 14, 7, 7, 14, 14, 11, 11]).reshape((8,1))
# w = np.array([
# [1, 0.5, 1, 2, -1, 1, 1, -1],
# [0.5, 1, 2, 1, -0.5, 0.5, -1, 1],
# [0.25, -1, 1, 1, 0.5, -1, -1, -1]
# ])
# b = np.array([0, 0, 0]).reshape(3, 1)
#
# output = np.dot(w, input) + b
#
# print 'output\n', output

"""
 test relu layer backward
"""
#input_od = np.zeros(output.shape)
# zeros = np.zeros(output.shape)
# input_od = np.array(np.greater(output, zeros), dtype=int)
# print 'input_od\n', input_od

# a = np.array([
#     [1, 2],
#     [3, 4]])
#
# b = np.matlib.repmat(a, 1, 4)
# print 'b = \n', b

"""
 test write output to a file
"""
f = open('out.txt', 'w')
for i in range(10):
 print >> f, 'i = ', i
f.close()
