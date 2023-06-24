import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pauli matrices and identity

s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])
I = np.matrix([[1, 0],[0,1]])

# tensor product

s1_I_I_I_I = np.kron(np.kron(np.kron(np.kron(s1, I), I), I), I)
I_s1_I_I_I = np.kron(np.kron(np.kron(np.kron(I, s1), I), I), I)
I_I_s1_I_I = np.kron(np.kron(np.kron(np.kron(I, I), s1), I), I)
I_I_I_s1_I = np.kron(np.kron(np.kron(np.kron(I, I), I), s1), I)
I_I_I_I_s1 = np.kron(np.kron(np.kron(np.kron(I, I), I), I), s1)

s3_I_I_I_I = np.kron(np.kron(np.kron(np.kron(s3, I), I), I), I)
I_s3_I_I_I = np.kron(np.kron(np.kron(np.kron(I, s3), I), I), I)
I_I_s3_I_I = np.kron(np.kron(np.kron(np.kron(I, I), s3), I), I)
I_I_I_s3_I = np.kron(np.kron(np.kron(np.kron(I, I), I), s3), I)
I_I_I_I_s3 = np.kron(np.kron(np.kron(np.kron(I, I), I), I), s3)

s3_s3_I_I_I = np.kron(np.kron(np.kron(np.kron(s3, s3), I), I), I)
s3_I_s3_I_I = np.kron(np.kron(np.kron(np.kron(s3, I), s3), I), I)
s3_I_I_s3_I = np.kron(np.kron(np.kron(np.kron(s3, I), I), s3), I)
s3_I_I_I_s3 = np.kron(np.kron(np.kron(np.kron(s3, I), I), I), s3)
I_s3_s3_I_I = np.kron(np.kron(np.kron(np.kron(I, s3), s3), I), I)
I_s3_I_s3_I = np.kron(np.kron(np.kron(np.kron(I, s3), I), s3), I)
I_s3_I_I_s3 = np.kron(np.kron(np.kron(np.kron(I, s3), I), I), s3)
I_I_s3_s3_I = np.kron(np.kron(np.kron(np.kron(I, I), s3), s3), I)
I_I_s3_I_s3 = np.kron(np.kron(np.kron(np.kron(I, I), s3), I), s3)
I_I_I_s3_s3 = np.kron(np.kron(np.kron(np.kron(I, I), I), s3), s3)

# objective function parameters

biases = np.array([7, 13/2, 6, 11/2, 15/2])
coupling_strengths = np.matrix([[0, 5/2, 5/2, 5/2, 5/2],
                                [0, 0, 5/2, 5/2, 5/2], 
                                [0, 0, 0, 5/2, 5/2], 
                                [0, 0, 0, 0, 5/2], 
                                [0, 0, 0, 0, 0]])

# QPU anneal parameters

data = pd.read_excel(r'.\Advantage_system6_2_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

h = 6.62607015e-25

A = A * h
B = B * h

# hamiltonians

h0 = (s1_I_I_I_I + I_s1_I_I_I + I_I_s1_I_I + I_I_I_s1_I + I_I_I_I_s1)
hf = (biases.item(0) * s3_I_I_I_I + biases.item(1) * I_s3_I_I_I + biases.item(2) * I_I_s3_I_I + biases.item(3) * I_I_I_s3_I + biases.item(4) * I_I_I_I_s3) + (coupling_strengths.item(0, 1) * s3_s3_I_I_I + coupling_strengths.item(0, 2) * s3_I_s3_I_I + coupling_strengths.item(0, 3) * s3_I_I_s3_I + coupling_strengths.item(0, 4) * s3_I_I_I_s3 + coupling_strengths.item(1, 2) * I_s3_s3_I_I + coupling_strengths.item(1, 3) * I_s3_I_s3_I + coupling_strengths.item(1, 4) * I_s3_I_I_s3 + coupling_strengths.item(2, 3) * I_I_s3_s3_I + coupling_strengths.item(2, 4) * I_I_s3_I_s3 + coupling_strengths.item(3, 4) * I_I_I_s3_s3)

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
    
df = pd.DataFrame(EigVectors, columns=['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'e21', 'e22', 'e23', 'e24', 'e25', 'e26', 'e27', 'e28', 'e29', 'e30', 'e31'])
df.to_excel('final_eigenvec_five.xlsx', index=False)

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
e16 = []
e17 = []
e18 = []
e19 = []
e20 = []
e21 = []
e22 = []
e23 = []
e24 = []
e25 = []
e26 = []
e27 = []
e28 = []
e29 = []
e30 = []
e31 = []

for i in range(0, len(e) - 31, 32):
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
    e16 = np.append(e16, e.item(i + 16))
    e17 = np.append(e17, e.item(i + 17))
    e18 = np.append(e18, e.item(i + 18))
    e19 = np.append(e19, e.item(i + 19))
    e20 = np.append(e20, e.item(i + 20))
    e21 = np.append(e21, e.item(i + 21))
    e22 = np.append(e22, e.item(i + 22))
    e23 = np.append(e23, e.item(i + 23))
    e24 = np.append(e24, e.item(i + 24))
    e25 = np.append(e25, e.item(i + 25))
    e26 = np.append(e26, e.item(i + 26))
    e27 = np.append(e27, e.item(i + 27))
    e28 = np.append(e28, e.item(i + 28))
    e29 = np.append(e29, e.item(i + 29))
    e30 = np.append(e30, e.item(i + 30))
    e31 = np.append(e31, e.item(i + 31))
    
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
    e16[i] = e16[i] - e0[i]
    e17[i] = e17[i] - e0[i]
    e18[i] = e18[i] - e0[i]
    e19[i] = e19[i] - e0[i]
    e20[i] = e20[i] - e0[i]
    e21[i] = e21[i] - e0[i]
    e22[i] = e22[i] - e0[i]
    e23[i] = e23[i] - e0[i]
    e24[i] = e24[i] - e0[i]
    e25[i] = e25[i] - e0[i]
    e26[i] = e26[i] - e0[i]
    e27[i] = e27[i] - e0[i]
    e28[i] = e28[i] - e0[i]
    e29[i] = e29[i] - e0[i]
    e30[i] = e30[i] - e0[i]
    e31[i] = e31[i] - e0[i]
    e0[i] = e0[i] - e0[i]

print('The final eigenvectors are saved in the Excel file named "final_eigenvec_five.xlsx": the eigenvector e0 is the one corresponding to the ground state of the Hamiltonian.')

print("The final eigenvalues are:")
print(EigValues)

# minimum gap

minimum_gap = min(e1)
t_min = s.item(e1.argmin())

print('The band gap is', minimum_gap.real, 'Joule and occurs when s =', t_min)

# drawing

plt.grid()
plt.plot(s, e0, c = 'black')
plt.plot(s, e1, c = 'yellow')
plt.plot(s, e2, c = 'purple')
plt.plot(s, e3, c = 'green')
plt.plot(s, e4, c = 'blue')
plt.plot(s, e5, c = 'red')
plt.plot(s, e6, c = 'brown')
plt.plot(s, e7, c = 'grey')
plt.plot(s, e8, c = 'black')
plt.plot(s, e9, c = 'yellow')
plt.plot(s, e10, c = 'purple')
plt.plot(s, e11, c = 'green')
plt.plot(s, e12, c = 'blue')
plt.plot(s, e13, c = 'red')
plt.plot(s, e14, c = 'brown')
plt.plot(s, e15, c = 'grey')
plt.plot(s, e16, c = 'black')
plt.plot(s, e17, c = 'yellow')
plt.plot(s, e18, c = 'purple')
plt.plot(s, e19, c = 'green')
plt.plot(s, e20, c = 'blue')
plt.plot(s, e21, c = 'red')
plt.plot(s, e22, c = 'brown')
plt.plot(s, e23, c = 'grey')
plt.plot(s, e24, c = 'black')
plt.plot(s, e25, c = 'yellow')
plt.plot(s, e26, c = 'purple')
plt.plot(s, e27, c = 'green')
plt.plot(s, e28, c = 'blue')
plt.plot(s, e29, c = 'red')
plt.plot(s, e30, c = 'brown')
plt.plot(s, e31, c = 'grey')

plt.xlabel("s = t/tA")
plt.ylabel("Energy (GHz)")
plt.show()