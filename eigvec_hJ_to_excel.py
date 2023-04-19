import numpy as np
import pandas as pd
import math

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

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

# objective function parameters

def coupling_strengths(t):
    return t

def biases(t):
    return t

# QPU anneal parameters

data = pd.read_excel(r'.\DWAVE_2000Q_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

# hamiltonians

h0 = (s1_I + I_s1)

def hf(h, J):
    return (biases(h) * (s3_I + I_s3)) + (coupling_strengths(J) * s3_s3)

def H(t, h, J):
    return - A.item(t) / 2 * h0 + B.item(t) / 2 * hf(h, J)
    
# eigenvalues and eigenvectors

E = np.empty([4, 4])
h_values = np.empty([4, 1])
j_values = np.empty([4, 1])
pippo = []
pluto = []

for h in np.arange (-1, 1.1, 0.1):
    for J in np.arange (-1, 1.1, 0.1): 
        for i in range (0, len(s)):
            EigValues, EigVectors = np.linalg.eig(H(i, h, J))
            permute = EigValues.argsort()
            EigValues = EigValues[permute]
            EigVectors = EigVectors[:,permute]
            
            EigVectors = np.real(EigVectors)
                
            for pluto in range (0,4):
                round_half_up(EigVectors.item(pluto), 2)
                
        E = np.append(E, EigVectors, axis = 0)
        pippo = np.matrix([[round_half_up(h, 1)], ['-'], ['-'], ['-']])
        pluto = np.matrix([[round_half_up(J, 1)], ['-'], ['-'], ['-']])
        
        h_values = np.append(h_values, pippo, axis=0)
        j_values = np.append(j_values, pluto, axis=0)

for i in range (0, 4): 
    E = np.delete(E, 0, 0)
    h_values = np.delete(h_values, 0, 0)
    j_values = np.delete(j_values, 0, 0)

# writing on the excel

data = pd.DataFrame(E, columns=['e0', 'e1', 'e2', 'e3'])

data['h'] = h_values
data['J'] = j_values

data.to_excel('eigenvectors_hJ.xlsx', index=False)