print("Difficulty Level Beginner")
# 1 Import the numpy package under the name np
import numpy as np

# 2 Create a null vector of size 10
vector = np.zeros(10)
print(vector)

# 3 Create a vector with values ranging from 10 to 49

vector1 = np.arange(10, 49)
print(vector1)

# 4 Find the shape of previous array in question 3

print(vector1.shape)

# 5 Print the type of the previous array in question 3

print(vector1.dtype)

# 6 Print the numpy version and the configuration

print(np.__version__, np.show_config())

# 7 Print the dimension of the array in question 3

print(vector1.ndim)

# 8 Create a boolean array with all the True values

vector2=np.array(range(1,10), dtype="bool")
print(vector2)


# 9 Create a two dimensional array

vector3=np.zeros((2,2))
print(vector3)

# 10 Create a three dimensional array

vector4=np.ones((3,3))
print((vector4))

print("Difficulty Level Easy")

# 11 Reverse a vector (first element becomes last)

# Changing variable names as they are getting too boring!
reverse_vector = np.array(range(1, 10))
print("Original: ", reverse_vector)

reverse_vector[0], reverse_vector[len(reverse_vector) - 1] = reverse_vector[len(reverse_vector) - 1], reverse_vector[0]

print("Modified: ", reverse_vector)

# 12 Create a null vector of size 10 but the fifth value which is 1

null_vector = np.zeros(10)
print(null_vector)
null_vector[5]=1 
print(null_vector) 

# 13 Create a 3x3 identity matrix

matrix=np.identity(3)
print(matrix)

# 14 arr = np.array([1, 2, 3, 4, 5])
# Convert the data type of the given array from int to float

arr = np.array([1, 2, 3, 4, 5])
arr=arr.astype(np.float32)
print(arr)


# 15 arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])  arr2 = np.array([[0., 4., 1.],  [7., 2., 12.]]) Multiply arr1 with arr2

arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])

print(np.multiply(arr1,arr2))

# 16 arr1 = np.array([[1., 2., 3.],      [4., 5., 6.]])  arr2 = np.array([[0., 4., 1.],            [7., 2., 12.]]) Make an array by comparing both the arrays provided above
arr1 = np.array([[1., 2., 3.],      [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],      [7., 2., 12.]])
print(np.greater(arr1,arr2))

# 17 Extract all odd numbers from arr with values(0-9)

arr = np.arange(9)
filtered_array = filter(lambda val: val % 2 == 0, arr)
list(filtered_array)

# 18 Replace all odd numbers to -1 from previous array

arr[arr % 2 == 0] = -1
print(arr)

# 19 arr = np.arange(10) Replace the values of indexes 5,6,7 and 8 to 12
arr = np.arange(10)
print(arr)
arr[5] = arr[6] = arr[7] = arr[8] = 12
print(arr)


# 20 reate a 2d array with 1 on the border and 0 inside

arr = np.ones((8, 8))
arr[1:-1, 1:-1] = 0
print(arr)

print("Difficulty Level Medium")

# arr2d = np.array([[1, 2, 3], [4, 5, 6],   [7, 8, 9]]) Replace the value 5 to 12

arr2d = np.array([[1, 2, 3],

               [4, 5, 6], 

                    [7, 8, 9]])
arr2d[1, 1] = 12
print(arr2d)

# 22 arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]) Convert all the values of 1st array to 64

arr3d = np.array([ [ [1, 2, 3], [4, 5, 6] ], [ [7, 8, 9], [10, 11, 12] ] ])

arr3d[0][:] = 64
arr3d[1][:] = 64

print(arr3d)

# 23 Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

arr = np.arange(9).reshape(3, 3)
arr2 = arr[0] #TODO
print(arr2)

# 24 Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

arr = np.arange(9).reshape(3, 3)
val = arr[1][1] #TODO
print(val)

# 25 Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

arr = np.arange(9).reshape(3, 3)
arr3 = arr[0:-1, -1]

print(arr3)

# 26 Create a 10x10 array with random values and find the minimum and maximum values


arr = np.arange(100).reshape(10, 10)
maxInColumns = np.amax(arr, axis=1)
val = max(maxInColumns)
print(val)

# 27 a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8]) Find the common items between a and b


a = np.array([1,2,3,2,3,4,3,4,5,6]) 

b = np.array([7,2,10,2,7,4,9,4,9,8])

print(np.intersect1d(a, b))


# 28 a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# Find the positions where elements of a and b match

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

print(np.where(a == b))

# 29 names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# Find all the values from array data where the values from array names are not equal to Will

import random

name = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  
# data = np.random.randn(7,4) # --> This gives floating point numbers
data = np.random.randint(7, size=random.randrange(1, 10))
print(data)
print(np.where(name[data] != "Will"))


# names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# Find all the values from array data where the values from array names are not equal to Will and Joe

import random

name = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  
# data = np.random.randn(7,4) # --> This gives floating point numbers
data = np.random.randint(7, size=random.randrange(1, 10))
print(data)
print(np.where((name[data] != "Joe") & (name[data] != "Will")))

print("Difficulty Level Hard")

# 31 Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

arr = np.random.uniform(low=1.0, high=15.0, size=(5,3))
print(arr)

# 32 Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

arr_klo = np.random.uniform(low=1.0, high=16.0, size=(2,2,4))
print(arr_klo)

# 33 Swap axes of the array you created in Question 32


np.swapaxes(arr_klo, 0, 1)
print(arr_klo)

# 34 Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0
import math

arr = np.arange(10)
print(arr)
new_arr = filter(lambda x: x if math.sqrt(x) > 0.5 else 0, arr)
print(np.asarray(list(new_arr)))


# 35 Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

arr1 = np.random.randint(12, size=random.randrange(0, 12))
arr2 = np.random.randint(12, size=random.randrange(0, 12))
print(arr1)
print(arr2)
print(np.where(arr1 == arr2))


# 36 names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# Find the unique names and sort them out!

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print(np.unique(names)) # --> returns sorted unique values

 # 37 a = np.array([1,2,3,4,5]) b = np.array([5,6,7,8,9])
# From array a remove all items present in array b

a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
val = np.intersect1d(a, b)

a = filter(lambda x: x if x not in val else None, a)
print(np.asarray(list(a)))

# 38 ollowing is the input NumPy array delete column two and insert following new column in its place.
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]])
# newColumn = numpy.array([[10,10,10]])

sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]]) 
sampleArray[:, 1] = 10
print(sampleArray)


# 39 x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# Find the dot product of the above two matrix


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])

a=np.dot(x, y)

print(a)

# 40 enerate a matrix of 20 random values and find its cumulative sum

matrix = np.random.randint(20, size=(random.randint(0, 5), random.randint(0, 5)))
b = np.cumsum(matrix)
print (b)