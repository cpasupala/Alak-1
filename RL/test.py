import numpy as np
A = [1,2,3]
B = [4,5,6]
print(np.array(np.meshgrid(A,B)).T.reshape(-1,2))

