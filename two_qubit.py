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

biases = np.array([-1, 1])
coupling_strengths = 1

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

e = np.empty([4, 1])
E = np.empty([4, 4])

for i in range (0, len(s)):
    EigValues, EigVectors = np.linalg.eig(H(i))
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    EigVectors = EigVectors[:,permute]
    e = np.append(e, EigValues)
    E = np.append(E, EigVectors, axis = 0)
    
df = pd.DataFrame(EigVectors, columns=['e0', 'e1', 'e2', 'e3'])
df.to_excel('final_eigenvectors.xlsx', index=False)

for i in range (0, 4): 
    e = np.delete(e, 0)
    E = np.delete(E, 0, 0)

e0 = np.empty([4, 1])
e1 = np.empty([4, 1])
e2 = np.empty([4, 1])
e3 = np.empty([4, 1])

for i in range(0, len(e), 4):
    e0 = np.append(e0, e.item(i))

for i in range(1, len(e), 4):
    e1 = np.append(e1, e.item(i))

for i in range(2, len(e), 4):
    e2 = np.append(e2, e.item(i))
    
for i in range(3, len(e), 4):
    e3 = np.append(e3, e.item(i))
   
for i in range (0, 4): 
    e0 = np.delete(e0, 0)
    e1 = np.delete(e1, 0)
    e2 = np.delete(e2, 0)
    e3 = np.delete(e3, 0)
    
for i in range (0, len(e0)):
    e1[i] = e1[i] - e0[i]
    e2[i] = e2[i] - e0[i]    
    e3[i] = e3[i] - e0[i]
    e0[i] = e0[i] - e0[i]

 
# drawing

print('The final eigenvectors are saved in the Excel file named "final_eigenvectors.xlsx": the eigenvector e0 is the one corresponding to the ground state of the Hamiltonian.')

print("The final eigenvalues are:")
print(EigValues)

plt.grid()
plt.plot(s, e0, c = 'black')
plt.plot(s, e1, c = 'yellow')
plt.plot(s, e2, c = 'purple')
plt.plot(s, e3, c = 'green')
plt.xlabel("s = t/tA")
plt.ylabel("Energy (GHz)")
plt.show()