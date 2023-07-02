import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pauli matrices and identity

s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])
I = np.matrix([[1, 0],[0,1]])

# tensor product

s1_I_I_I = np.kron(np.kron(np.kron(s1, I), I), I)
I_s1_I_I = np.kron(np.kron(np.kron(I, s1), I), I)
I_I_s1_I = np.kron(np.kron(np.kron(I, I), s1), I)
I_I_I_s1 = np.kron(np.kron(np.kron(I, I), I), s1)

s3_I_I_I = np.kron(np.kron(np.kron(s3, I), I), I)
I_s3_I_I = np.kron(np.kron(np.kron(I, s3), I), I)
I_I_s3_I = np.kron(np.kron(np.kron(I, I), s3), I)
I_I_I_s3 = np.kron(np.kron(np.kron(I, I), I), s3)

s3_s3_I_I = np.kron(np.kron(np.kron(s3, s3), I), I)
s3_I_s3_I = np.kron(np.kron(np.kron(s3, I), s3), I)
s3_I_I_s3 = np.kron(np.kron(np.kron(s3, I), I), s3)
I_s3_s3_I = np.kron(np.kron(np.kron(I, s3), s3), I)
I_s3_I_s3 = np.kron(np.kron(np.kron(I, s3), I), s3)
I_I_s3_s3 = np.kron(np.kron(np.kron(I, I), s3), s3)

# objective function parameters

biases = np.array([1, 2, 3, 4])
coupling_strengths = np.matrix([[0, -7, -7, -7], 
                                [0, 0, -7, -7],
                                [0, 0, 0, -7], 
                                [0, 0, 0, 0]])

# QPU anneal parameters

data = pd.read_excel(r'.\Advantage_system6_2_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

h = 6.62607015e-25

A = A * h
B = B * h

# hamiltonians

h0 = (s1_I_I_I + I_s1_I_I + I_I_s1_I + I_I_I_s1)
hf = (biases.item(0) * s3_I_I_I + biases.item(1) * I_s3_I_I + biases.item(2) * I_I_s3_I + biases.item(3) * I_I_I_s3) + (coupling_strengths.item(0, 1) * s3_s3_I_I + coupling_strengths.item(0, 2) * s3_I_s3_I + coupling_strengths.item(0, 3) * s3_I_I_s3 + coupling_strengths.item(1, 2) * I_s3_s3_I + coupling_strengths.item(1, 3) * I_s3_I_s3 + coupling_strengths.item(2, 3) * I_I_s3_s3)

def H(t): 
    return - A.item(t) / 2 * h0 + B.item(t) / 2 * hf

# eigenvalues and eigenvectors

e = []

for i in range (0, len(s)):
    EigValues, EigVectors = np.linalg.eig(H(i))
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    EigVectors = EigVectors[:,permute]
    e = np.append(e, EigValues)
    
df = pd.DataFrame(EigVectors, columns=['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15'])
df.to_excel('final_eigenvec_four.xlsx', index=False)

e0 = []
e1 = []
e2 = []
e3 = []
e4 = []
e5 = []
e6 = []
e7 = []
e8 = []
e9 = []
e10 = []
e11 = []
e12 = []
e13 = []
e14 = []
e15 = []

for i in range(0, len(e) - 15, 16):
    e0 = np.append(e0, e.item(i))
    e1 = np.append(e1, e.item(i + 1))
    e2 = np.append(e2, e.item(i + 2))
    e3 = np.append(e3, e.item(i + 3))
    e4 = np.append(e4, e.item(i + 4))
    e5 = np.append(e5, e.item(i + 5))
    e6 = np.append(e6, e.item(i + 6))
    e7 = np.append(e7, e.item(i + 7))
    e8 = np.append(e8, e.item(i + 8))
    e9 = np.append(e9, e.item(i + 9))
    e10 = np.append(e10, e.item(i + 10))
    e11 = np.append(e11, e.item(i + 11))
    e12 = np.append(e12, e.item(i + 12))
    e13 = np.append(e13, e.item(i + 13))
    e14 = np.append(e14, e.item(i + 14))
    e15 = np.append(e15, e.item(i + 15))
    
for i in range (0, len(e0)):
    e1[i] = e1[i] - e0[i]
    e2[i] = e2[i] - e0[i]    
    e3[i] = e3[i] - e0[i]
    e4[i] = e4[i] - e0[i]
    e5[i] = e5[i] - e0[i]
    e6[i] = e6[i] - e0[i]
    e7[i] = e7[i] - e0[i]
    e8[i] = e8[i] - e0[i]
    e9[i] = e9[i] - e0[i]    
    e10[i] = e10[i] - e0[i]
    e11[i] = e11[i] - e0[i]
    e12[i] = e12[i] - e0[i]
    e13[i] = e13[i] - e0[i]
    e14[i] = e14[i] - e0[i]
    e15[i] = e15[i] - e0[i]
    e0[i] = e0[i] - e0[i]

print('The final eigenvectors are saved in the Excel file named "final_eigenvec_four.xlsx": the eigenvector e0 is the one corresponding to the ground state of the Hamiltonian.')

print("The final eigenvalues are:")
print(EigValues)

# minimum gap

minimum_gap = min(e1)
t_min = s.item(e1.argmin())

print('The band gap is', minimum_gap.real, 'Joule and occurs when s =', t_min)

# drawing

plt.grid()
plt.plot(s, e2, c = 'yellow')
plt.plot(s, e3, c = 'green')
plt.plot(s, e4, c = 'blue')
plt.plot(s, e5, c = 'brown')
plt.plot(s, e6, c = 'purple')
plt.plot(s, e7, c = 'grey')
plt.plot(s, e8, c = 'green')
plt.plot(s, e9, c = 'yellow')
plt.plot(s, e10, c = 'purple')
plt.plot(s, e11, c = 'green')
plt.plot(s, e12, c = 'blue')
plt.plot(s, e13, c = 'cyan')
plt.plot(s, e14, c = 'brown')
plt.plot(s, e15, c = 'grey')
plt.plot(s, e0, c = 'black')
plt.plot(s, e1, c = 'red')

plt.xlabel("s = t/tA")
plt.ylabel("Energy (J)")
plt.show()