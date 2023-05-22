import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pauli matrices and identity

s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])
I = np.matrix([[1, 0],[0,1]])

# tensor product

s1_I_I = np.kron(np.kron(s1, I), I)
I_s1_I = np.kron(np.kron(I, s1), I)
I_I_s1 = np.kron(np.kron(I, I), s1)

s3_I_I = np.kron(np.kron(s3, I), I)
I_s3_I = np.kron(np.kron(I, s3), I)
I_I_s3 = np.kron(np.kron(I, I), s3)

s3_s3_I = np.kron(np.kron(s3, s3), I)
s3_I_s3 = np.kron(np.kron(s3, I), s3)
I_s3_s3 = np.kron(np.kron(I, s3), s3)

s1_I_I_I = np.kron(np.kron(np.kron(s1, I), I), I)
I_s1_I_I = np.kron(np.kron(np.kron(I, s1), I), I)
I_I_s1_I = np.kron(np.kron(np.kron(I, I), s1), I)
I_I_I_s1 = np.kron(np.kron(np.kron(I, I), I), s1)

s3_I_I_I = np.kron(np.kron(np.kron(s3, I), I), I)
I_s3_I_I = np.kron(np.kron(np.kron(I, s3), I), I)
I_I_s3_I = np.kron(np.kron(np.kron(I, I), s3), I)
I_I_I_s3 = np.kron(np.kron(np.kron(I, I), I), s3)

s3_s3_I_I = np.kron(np.kron(np.kron(s3, s3), I), I)
I_s3_s3_I = np.kron(np.kron(np.kron(I, s3), s3), I)
I_I_s3_s3 = np.kron(np.kron(np.kron(I, I), s3), s3)
s3_I_I_s3 = np.kron(np.kron(np.kron(s3, I), I), s3)

# objective function parameters

biases3 = np.array([0.5, 0, 1])
coupling_strengths3 = np.matrix([[0, 1.5, 1.5], 
                                [0, 0, 1.5], 
                                [0, 0, 0]])

biases4 = np.array([0.5, 0, 1, 0.5])
coupling_strengths4 = np.matrix([[0, 1.5, 0, -3], 
                                [0, 0, 1.5, 0],
                                [0, 0, 0, 1.5], 
                                [0, 0, 0, 0]])
# QPU anneal parameters

data = pd.read_excel(r'.\DWAVE_2000Q_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

# hamiltonians

h03 = (s1_I_I + I_s1_I + I_I_s1)
hf3 = (biases3.item(0) * s3_I_I + biases3.item(1) * I_s3_I + biases3.item(2) * I_I_s3) + (coupling_strengths3.item(0, 1) * s3_s3_I + coupling_strengths3.item(0, 2) * s3_I_s3 + coupling_strengths3.item(1, 2) * I_s3_s3)

def H3(t): 
    return - A.item(t) / 2 * h03 + B.item(t) / 2 * hf3

h04 = (s1_I_I_I + I_s1_I_I + I_I_s1_I + I_I_I_s1)
hf4 = (biases4.item(0) * s3_I_I_I + biases4.item(1) * I_s3_I_I + biases4.item(2) * I_I_s3_I + biases4.item(3) * I_I_I_s3) + (coupling_strengths4.item(0, 1) * s3_s3_I_I+ coupling_strengths4.item(1, 2) * I_s3_s3_I + coupling_strengths4.item(2, 3) * I_I_s3_s3 + coupling_strengths4.item(0, 3) * s3_I_I_s3)

def H4(t): 
    return - A.item(t) / 2 * h04 + B.item(t) / 2 * hf4

# eigenvalues and eigenvectors

e_3 = []
e_4 = []

for i in range (0, len(s)):
    EigValues3, EigVectors3 = np.linalg.eig(H3(i))
    EigValues4, EigVectors4 = np.linalg.eig(H4(i))
    
    permute = EigValues3.argsort()
    EigValues3 = EigValues3[permute]
    EigVectors3 = EigVectors3[:,permute]
    permute = EigValues4.argsort()
    EigValues4 = EigValues4[permute]
    EigVectors4 = EigVectors4[:,permute]
    
    e_3 = np.append(e_3, EigValues3)
    e_4 = np.append(e_4, EigValues4)

df3 = pd.DataFrame(EigVectors3, columns=['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7'])
df3.to_excel('final_eigenvec_three.xlsx', index=False)

df4 = pd.DataFrame(EigVectors4, columns=['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15'])
df4.to_excel('final_eigenvec_four.xlsx', index=False)

e03 = []
e13 = []
e04 = []
e14 = []

for i in range(0, len(e_3) - 7, 8):
    e03 = np.append(e03, e_3.item(i))
    e13 = np.append(e13, e_3.item(i + 1))

for i in range(0, len(e_4) - 15, 16):
    e04 = np.append(e04, e_4.item(i))
    e14 = np.append(e14, e_4.item(i + 1))

assert(len(e03) == len (e04))

for i in range (0, len(e03)):
    e13[i] = e13[i] - e03[i]
    e03[i] = e03[i] - e03[i]
    e14[i] = e14[i] - e04[i]
    e04[i] = e04[i] - e04[i]

print('The final eigenvectors are saved in the Excel files named "final_eigenvec_three.xlsx" and "final_eigenvec_four.xlsx": the eigenvector e0 is the one corresponding to the ground state of the Hamiltonian.')

# minimum gap

minimum_gap3 = min(e13)
t_min3 = s.item(e13.argmin())
minimum_gap4 = min(e14)
t_min4 = s.item(e14.argmin())

print('The band gap for the 3 qubits (blue line) is', minimum_gap3.real, 'GHz and occurs when s =', t_min3)

print('The band gap for the 4 qubits (red line) is', minimum_gap4.real, 'GHz and occurs when s =', t_min4)

# drawing

plt.grid()
plt.plot(s, e03, c = 'black')
plt.plot(s, e04, c = 'black')
plt.plot(s, e13, c = 'blue')
plt.plot(s, e14, c = 'red')

plt.xlabel("s = t/tA")
plt.ylabel("Energy (GHz)")
plt.show()