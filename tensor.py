import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pauli matrices and identity

s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])
I = np.matrix([[1, 0],[0,1]])

# tensor product

s1_I = np.kron(s1, I)
I_s1 = np.kron(I, s1)
s3_I = np.kron(s3, I)
I_s3 = np.kron(I, s3)
s3_s3 = np.kron(s3, s3)

# objective function parameters

biases = np.array([1,1])
coupling_strengths = -2

# QPU anneal parameters

data = pd.read_excel(r'.\DWAVE_2000Q_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

# hamiltonians

h0 = (s1_I + I_s1)
hf = (biases.item(0) * s3_I + biases.item(1) * I_s3) + (coupling_strengths * s3_s3)

def H(t): 
    return - A.item(t) / 2 * h0 + B.item(t) / 2 * hf
    
# eigenvalues and eigenvectors


E = np.empty([4, 1])

for i in range (0, len(s)):
    EigValues = np.linalg.eigvalsh(H(i))
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    E = np.append(E, EigValues)

for i in range (0, 4): 
    E = np.delete(E, 0)

print(len(E))
print(E[0], E[4])

e0 = np.empty([4, 1])

for i in range(0, len(E), 4):
    e0 = np.append(e0, E.item(i))

for i in range (0, 4): 
    e0 = np.delete(e0, 0)
    
e1 = np.empty([4, 1])

for i in range(1, len(E), 4):
    e1 = np.append(e1, E.item(i))

for i in range (0, 4): 
    e1 = np.delete(e1, 0)
    
e2 = np.empty([4, 1])

for i in range(2, len(E), 4):
    e2 = np.append(e2, E.item(i))

for i in range (0, 4): 
    e2 = np.delete(e2, 0)

e3 = np.empty([4, 1])

for i in range(3, len(E), 4):
    e3 = np.append(e3, E.item(i))

for i in range (0, 4): 
    e3 = np.delete(e3, 0)
    
plt.grid()
plt.plot(s, A, c = 'red')
plt.plot(s, B, c = 'blue')
plt.plot(s, e0, c = 'green')
plt.plot(s, e1, c = 'green')
plt.plot(s, e2, c = 'green')
plt.plot(s, e3, c = 'green')
plt.show()
plt.grid()