import numpy as np

#creating arrays
array = np.zeros(10)
array = np.ones(10)
array = np.full(10, 2.5)
array = np.array([1, 2, 3, 4, 5])
array = np.arange(3, 10)
array = np.linspace(0, 100, 11)

#multidimenaitonsal array
array = np.zeros((5, 2))
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#randomly genrated arrays
np.random.seed(42)
array = np.random.rand()
array = np.random.rand(10)



def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]

    n = u.shape[0]

    result = 0.0
    
    for i in range(n):
        result += u[i]*v[i]

    return result




#displaying result
print(array)