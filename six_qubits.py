import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pauli matrices and identity

s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])
I = np.matrix([[1, 0],[0,1]])

# tensor product

s1_I_I_I_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(s1, I), I), I), I), I)
I_s1_I_I_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, s1), I), I), I), I)
I_I_s1_I_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), s1), I), I), I)
I_I_I_s1_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), s1), I), I)
I_I_I_I_s1_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), s1), I)
I_I_I_I_I_s1 = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), I), s1)

s3_I_I_I_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(s3, I), I), I), I), I)
I_s3_I_I_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, s3), I), I), I), I)
I_I_s3_I_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), s3), I), I), I)
I_I_I_s3_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), s3), I), I)
I_I_I_I_s3_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), s3), I)
I_I_I_I_I_s3 = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), I), s3)

s3_s3_I_I_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(s3, s3), I), I), I), I)
s3_I_I_s3_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(s3, I), I), s3), I), I)
s3_I_I_I_s3_I = np.kron(np.kron(np.kron(np.kron(np.kron(s3, I), I), I), s3), I)
s3_I_I_I_I_s3 = np.kron(np.kron(np.kron(np.kron(np.kron(s3, I), I), I), I), s3)
I_s3_I_s3_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, s3), I), s3), I), I)
I_s3_I_I_s3_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, s3), I), I), s3), I)
I_s3_I_I_I_s3 = np.kron(np.kron(np.kron(np.kron(np.kron(I, s3), I), I), I), s3)
I_I_s3_s3_I_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), s3), s3), I), I)
I_I_s3_I_s3_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), s3), I), s3), I)
I_I_s3_I_I_s3 = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), s3), I), I), s3)
I_I_I_s3_s3_I = np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), s3), s3), I)

# objective function parameters

biases = np.array([7, 13/2, 3, 11/2, 15/2, 3])
coupling_strengths = np.matrix([[0, 5/2, 0, 5/2, 5/2, 5/2],
                                [0, 0, 0, 5/2, 5/2, 5/2], 
                                [0, 0, 0, 5/2, 5/2, -7.07], 
                                [0, 0, 0, 0, 5/2, 0], 
                                [0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0]])

# QPU anneal parameters

data = pd.read_excel(r'.\Advantage_system6_2_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

h = 6.62607015e-25

A = A * h
B = B * h

# hamiltonians

h0 = (s1_I_I_I_I_I + I_s1_I_I_I_I + I_I_s1_I_I_I + I_I_I_s1_I_I + I_I_I_I_s1_I + I_I_I_I_I_s1)
hf = (biases.item(0) * s3_I_I_I_I_I + biases.item(1) * I_s3_I_I_I_I + biases.item(2) * I_I_s3_I_I_I + biases.item(3) * I_I_I_s3_I_I + biases.item(4) * I_I_I_I_s3_I + biases.item(5) * I_I_I_I_I_s3) + (coupling_strengths.item(0, 1) * s3_s3_I_I_I_I + coupling_strengths.item(0, 3) * s3_I_I_s3_I_I + coupling_strengths.item(0, 4) * s3_I_I_I_s3_I + coupling_strengths.item(0,5) * s3_I_I_I_I_s3 + coupling_strengths.item(1, 3) * I_s3_I_s3_I_I + coupling_strengths.item(1, 4) * I_s3_I_I_s3_I + coupling_strengths.item(1, 5) * I_s3_I_I_I_s3 + coupling_strengths.item(2, 3) * I_I_s3_s3_I_I + coupling_strengths.item(2, 4) * I_I_s3_I_s3_I + coupling_strengths.item(2, 5) * I_I_s3_I_I_s3 + coupling_strengths.item(3, 4) * I_I_I_s3_s3_I)

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
    
df = pd.DataFrame(EigVectors, columns=['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'e21', 'e22', 'e23', 'e24', 'e25', 'e26', 'e27', 'e28', 'e29', 'e30', 'e31', 'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'e21', 'e22', 'e23', 'e24', 'e25', 'e26', 'e27', 'e28', 'e29', 'e30', 'e31'])
df.to_excel('final_eigenvec_six.xlsx', index=False)

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
e32 = []
e33 = []
e34 = []
e35 = []
e36 = []
e37 = []
e38 = []
e39 = []
e40 = []
e41 = []
e42 = []
e43 = []
e44 = []
e45 = []
e46 = []
e47 = []
e48 = []
e49 = []
e50 = []
e51 = []
e52 = []
e53 = []
e54 = []
e55 = []
e56 = []
e57 = []
e58 = []
e59 = []
e60 = []
e61 = []
e62 = []
e63 = []

for i in range(0, len(e) - 63, 64):
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
    e32 = np.append(e32, e.item(i + 32))
    e33 = np.append(e33, e.item(i + 33))
    e34 = np.append(e34, e.item(i + 34))
    e35 = np.append(e35, e.item(i + 35))
    e36 = np.append(e36, e.item(i + 36))
    e37 = np.append(e37, e.item(i + 37))
    e38 = np.append(e38, e.item(i + 38))
    e39 = np.append(e39, e.item(i + 39))
    e40 = np.append(e40, e.item(i + 40))
    e41 = np.append(e41, e.item(i + 41))
    e42 = np.append(e42, e.item(i + 42))
    e43 = np.append(e43, e.item(i + 43))
    e44 = np.append(e44, e.item(i + 44))
    e45 = np.append(e45, e.item(i + 45))
    e46 = np.append(e46, e.item(i + 46))
    e47 = np.append(e47, e.item(i + 47))
    e48 = np.append(e48, e.item(i + 48))
    e49 = np.append(e49, e.item(i + 49))
    e50 = np.append(e50, e.item(i + 50))
    e51 = np.append(e51, e.item(i + 51))
    e52 = np.append(e52, e.item(i + 52))
    e53 = np.append(e53, e.item(i + 53))
    e54 = np.append(e54, e.item(i + 54))
    e55 = np.append(e55, e.item(i + 55))
    e56 = np.append(e56, e.item(i + 56))
    e57 = np.append(e57, e.item(i + 57))
    e58 = np.append(e58, e.item(i + 58))
    e59 = np.append(e59, e.item(i + 59))
    e60 = np.append(e60, e.item(i + 60))
    e61 = np.append(e61, e.item(i + 61))
    e62 = np.append(e62, e.item(i + 62))
    e63 = np.append(e63, e.item(i + 63))
    
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
    e32[i] = e32[i] - e0[i]
    e33[i] = e33[i] - e0[i]
    e34[i] = e34[i] - e0[i]
    e35[i] = e35[i] - e0[i]
    e36[i] = e36[i] - e0[i]
    e37[i] = e37[i] - e0[i]
    e38[i] = e38[i] - e0[i]
    e39[i] = e39[i] - e0[i]
    e40[i] = e40[i] - e0[i]
    e41[i] = e41[i] - e0[i]
    e42[i] = e42[i] - e0[i]
    e43[i] = e43[i] - e0[i]
    e44[i] = e44[i] - e0[i]
    e45[i] = e45[i] - e0[i]
    e46[i] = e46[i] - e0[i]
    e47[i] = e47[i] - e0[i]
    e48[i] = e48[i] - e0[i]
    e49[i] = e49[i] - e0[i]
    e50[i] = e50[i] - e0[i]
    e51[i] = e51[i] - e0[i]
    e52[i] = e52[i] - e0[i]
    e53[i] = e53[i] - e0[i]
    e54[i] = e54[i] - e0[i]
    e55[i] = e55[i] - e0[i]
    e56[i] = e56[i] - e0[i]
    e57[i] = e57[i] - e0[i]
    e58[i] = e58[i] - e0[i]
    e59[i] = e59[i] - e0[i]
    e60[i] = e60[i] - e0[i]
    e61[i] = e61[i] - e0[i]
    e62[i] = e62[i] - e0[i]
    e63[i] = e63[i] - e0[i]
    e0[i] = e0[i] - e0[i]

print('The final eigenvectors are saved in the Excel file named "final_eigenvec_six.xlsx": the eigenvector e0 is the one corresponding to the ground state of the Hamiltonian.')

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
plt.plot(s, e32, c = 'black')
plt.plot(s, e33, c = 'yellow')
plt.plot(s, e34, c = 'purple')
plt.plot(s, e35, c = 'green')
plt.plot(s, e36, c = 'blue')
plt.plot(s, e37, c = 'red')
plt.plot(s, e38, c = 'brown')
plt.plot(s, e39, c = 'grey')
plt.plot(s, e40, c = 'black')
plt.plot(s, e41, c = 'yellow')
plt.plot(s, e42, c = 'purple')
plt.plot(s, e43, c = 'green')
plt.plot(s, e44, c = 'blue')
plt.plot(s, e45, c = 'red')
plt.plot(s, e46, c = 'brown')
plt.plot(s, e47, c = 'grey')
plt.plot(s, e48, c = 'black')
plt.plot(s, e49, c = 'yellow')
plt.plot(s, e50, c = 'purple')
plt.plot(s, e51, c = 'green')
plt.plot(s, e52, c = 'blue')
plt.plot(s, e53, c = 'red')
plt.plot(s, e54, c = 'brown')
plt.plot(s, e55, c = 'grey')
plt.plot(s, e56, c = 'black')
plt.plot(s, e57, c = 'yellow')
plt.plot(s, e58, c = 'purple')
plt.plot(s, e59, c = 'green')
plt.plot(s, e60, c = 'blue')
plt.plot(s, e61, c = 'red')
plt.plot(s, e62, c = 'brown')
plt.plot(s, e63, c = 'grey')

plt.xlabel("s = t/tA")
plt.ylabel("Energy (GHz)")
plt.show()