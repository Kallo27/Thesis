import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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
I_s3_s3_I = np.kron(np.kron(np.kron(I, s3), s3), I)
I_I_s3_s3 = np.kron(np.kron(np.kron(I, I), s3), s3)
s3_I_I_s3 = np.kron(np.kron(np.kron(s3, I), I), s3)

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

# objective function parameters

biases4 = np.array([0.5, 0, 1, 0.5])
coupling_strengths4 = np.matrix([[0, 1.5, 0, 0], 
                                [0, 0, 1.5, 0],
                                [0, 0, 0, 1.5], 
                                [0, 0, 0, 0]])
# QPU anneal parameters

data = pd.read_excel(r'.\DWAVE_2000Q_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

# hamiltonians

h04 = (s1_I_I_I + I_s1_I_I + I_I_s1_I + I_I_I_s1)

def hf4(J14):
    return (biases4.item(0) * s3_I_I_I + biases4.item(1) * I_s3_I_I + biases4.item(2) * I_I_s3_I + biases4.item(3) * I_I_I_s3) + (coupling_strengths4.item(0, 1) * s3_s3_I_I+ coupling_strengths4.item(1, 2) * I_s3_s3_I + coupling_strengths4.item(2, 3) * I_I_s3_s3 + J14 * s3_I_I_s3)

def H4(t, J14): 
    return - A.item(t) / 2 * h04 + B.item(t) / 2 * hf4(J14)

# eigenvalues and eigenvectors

E = np.empty([16, 16])
min_en4 = np.empty([16, 1])
t_min4 = np.empty([16, 1])
e_4 = []
e04 = []
e14 = []
pippo = []
pluto = []

for J14 in np.arange(-10, 1.1, 0.1):
    for i in range (0, len(s)):
        EigValues4, EigVectors4 = np.linalg.eig(H4(i, J14))

        permute = EigValues4.argsort()
        EigValues4 = EigValues4[permute]
        EigVectors4 = EigVectors4[:,permute]
        e_4 = np.append(e_4, EigValues4)

        EigVectors4 = np.real(EigVectors4)
                
        for pluto in range (0,16):
            round_half_up(EigVectors4.item(pluto), 2)
                
    E = np.append(E, EigVectors4, axis = 0)
    
    for i in range(0, len(e_4) - 15, 16):
        e04 = np.append(e04, e_4.item(i))
        e14 = np.append(e14, e_4.item(i + 1))

    for i in range (0, len(e04)):
        e14[i] = e14[i] - e04[i]
        e04[i] = e04[i] - e04[i]
    
    pluto = np.matrix([[round_half_up(min(e14).real, 5)], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-']])
    
    pippo = np.matrix([[round_half_up(s.item(e14.argmin()), 5)], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-']])
    
    min_en4 = np.append(min_en4, pluto, axis=0)
    t_min4 = np.append(t_min4, pippo, axis=0)
    
    e_4 = []    
    e04 = []
    e14 = []

for i in range (0, 16): 
    E = np.delete(E, 0, 0)
    min_en4 = np.delete(min_en4, 0, 0)
    t_min4 = np.delete(t_min4, 0, 0)

print('The final eigenvectors and band gaps are saved in the Excel files named "eigenvec_bandgap.xlsx"; the value of J14 goes from -10 to 1.')

# writing on Excel

data = pd.DataFrame(E[:,0], columns=['e0'])

data['E min'] = min_en4
data['t min'] = t_min4

data.to_excel('eigenvec_bandgap.xlsx', index=False)

# plt.grid()
# plt.plot(s, e04, c = 'black')
# plt.plot(s, e14, c = 'red')
# 
# plt.xlabel("s = t/tA")
# plt.ylabel("Energy (GHz)")
# plt.show()
# 
# print(len(E))