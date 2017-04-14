import numpy as np
##
# hstack and vstack concatenate two arrays
a = np.array([[1, 2, 3],
                [1, 2, 3]])
b = np.array([[4, 5, 6],
                [7, 8, 9]])
print 'a = \n', a.shape
print 'b = \n', b.shape
c = np.hstack((a, b))
d = np.vstack((a, b))
print 'c = \n', c
print 'd = \n', d
