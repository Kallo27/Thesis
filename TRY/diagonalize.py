# import sympy 
from sympy import * 
M = Matrix([[3, -2,  4, -2],
            [5,  3, -3, -2],
            [5, -2,  2, -2],
            [5, -2, -3,  3]])
  
print("Matrix : {} ".format(M))
   
# Use sympy.diagonalize() method 
P, D = M.diagonalize()  
      
print("Diagonal of a matrix : {}".format(D))  