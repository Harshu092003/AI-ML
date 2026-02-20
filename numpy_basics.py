import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr)  # [1 2 3 4 5]

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", matrix)
# [[1 2 3]
#  [4 5 6]]

print("Shape:", matrix.shape)      # (2, 3)
print("Dimensions:", matrix.ndim) # 2
print("Data type:", matrix.dtype) # int64 (may vary)

nums = np.array([10, 20, 30])
print("Original:", nums)           # [10 20 30]
print("Add 5:", nums + 5)          # [15 25 35]
print("Multiply by 2:", nums * 2)  # [20 40 60]
print("Square:", nums ** 2)        # [100 400 900]

print("Zeros:\n", np.zeros((2, 3)))
# [[0. 0. 0.]
#  [0. 0. 0.]]

print("Ones:\n", np.ones((2, 3)))
# [[1. 1. 1.]
#  [1. 1. 1.]]

print("Range:", np.arange(0, 10, 2))   # [0 2 4 6 8]
print("Random numbers:", np.random.rand(5))  # random output

data = np.array([1, 2, 3, 4, 5])
print("Sum:", np.sum(data))    # 15
print("Mean:", np.mean(data))  # 3.0
print("Max:", np.max(data))    # 5
print("Min:", np.min(data))    # 1
print("Std Dev:", np.std(data))# ~1.41

arr2 = np.array([10, 20, 30, 40, 50])
print("First:", arr2[0])     # 10
print("Last:", arr2[-1])     # 50
print("Slice:", arr2[1:4])   # [20 30 40]

reshaped = np.arange(1, 7).reshape(2, 3)
print("Reshaped:\n", reshaped)
# [[1 2 3]
#  [4 5 6]]

a = np.array([1, 2, 3])
b = np.array([10])
print("Broadcast Add:", a + b)  # [11 12 13]
