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

biases5 = np.array([7, 13/2, 6, 11/2, 15/2])
coupling_strengths5 = np.matrix([[0, 5/2, 5/2, 5/2, 5/2],
                                [0, 0, 5/2, 5/2, 5/2], 
                                [0, 0, 0, 5/2, 5/2], 
                                [0, 0, 0, 0, 5/2], 
                                [0, 0, 0, 0, 0]])

biases6 = np.array([7, 13/2, 3, 11/2, 15/2, 3])
def coupling_strengths6(J):
    return np.matrix([[0, 5/2, 0, 5/2, 5/2, 5/2],
                    [0, 0, 0, 5/2, 5/2, 5/2], 
                    [0, 0, 0, 5/2, 5/2, J], 
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

h05 = (s1_I_I_I_I + I_s1_I_I_I + I_I_s1_I_I + I_I_I_s1_I + I_I_I_I_s1)
hf5 = (biases5.item(0) * s3_I_I_I_I + biases5.item(1) * I_s3_I_I_I + biases5.item(2) * I_I_s3_I_I + biases5.item(3) * I_I_I_s3_I + biases5.item(4) * I_I_I_I_s3) + (coupling_strengths5.item(0, 1) * s3_s3_I_I_I + coupling_strengths5.item(0, 2) * s3_I_s3_I_I + coupling_strengths5.item(0, 3) * s3_I_I_s3_I + coupling_strengths5.item(0, 4) * s3_I_I_I_s3 + coupling_strengths5.item(1, 2) * I_s3_s3_I_I + coupling_strengths5.item(1, 3) * I_s3_I_s3_I + coupling_strengths5.item(1, 4) * I_s3_I_I_s3 + coupling_strengths5.item(2, 3) * I_I_s3_s3_I + coupling_strengths5.item(2, 4) * I_I_s3_I_s3 + coupling_strengths5.item(3, 4) * I_I_I_s3_s3)

def H5(t): 
    return - A.item(t) / 2 * h05 + B.item(t) / 2 * hf5


h06 = (s1_I_I_I_I_I + I_s1_I_I_I_I + I_I_s1_I_I_I + I_I_I_s1_I_I + I_I_I_I_s1_I + I_I_I_I_I_s1)
def hf6(J):
    return (biases6.item(0) * s3_I_I_I_I_I + biases6.item(1) * I_s3_I_I_I_I + biases6.item(2) * I_I_s3_I_I_I + biases6.item(3) * I_I_I_s3_I_I + biases6.item(4) * I_I_I_I_s3_I + biases6.item(5) * I_I_I_I_I_s3) + (coupling_strengths6(J).item(0, 1) * s3_s3_I_I_I_I + coupling_strengths6(J).item(0, 3) * s3_I_I_s3_I_I + coupling_strengths6(J).item(0, 4) * s3_I_I_I_s3_I + coupling_strengths6(J).item(0,5) * s3_I_I_I_I_s3 + coupling_strengths6(J).item(1, 3) * I_s3_I_s3_I_I + coupling_strengths6(J).item(1, 4) * I_s3_I_I_s3_I + coupling_strengths6(J).item(1, 5) * I_s3_I_I_I_s3 + coupling_strengths6(J).item(2, 3) * I_I_s3_s3_I_I + coupling_strengths6(J).item(2, 4) * I_I_s3_I_s3_I + coupling_strengths6(J).item(2, 5) * I_I_s3_I_I_s3 + coupling_strengths6(J).item(3, 4) * I_I_I_s3_s3_I)

def H6(J, t): 
    return - A.item(t) / 2 * h06 + B.item(t) / 2 * hf6(J)
# eigenvalues and eigenvectors

e_5 = []

for i in range (0, len(s)):
    EigValues5, EigVectors5 = np.linalg.eig(H5(i))
    
    permute = EigValues5.argsort()
    EigValues5 = EigValues5[permute]
    EigVectors5 = EigVectors5[:,permute]

    e_5 = np.append(e_5, EigValues5)

    
e05 = []
e15 = []

for i in range(0, len(e_5) - 31, 32):
    e05 = np.append(e05, e_5.item(i))
    e15 = np.append(e15, e_5.item(i + 1))

for i in range (0, len(e05)):
    e15[i] = e15[i] - e05[i]
    e05[i] = e05[i] - e05[i]

# minimum gap
minimum_gap5 = min(e15)
t_min5 = s.item(e15.argmin())

print('The band gap for the 5 qubits (blue line) is', minimum_gap5.real, 'Joule and occurs when s =', t_min5)

def ground_first(J):
    e_6 = []

    for i in range (0, len(s)):
        EigValues6, EigVectors6 = np.linalg.eig(H6(J, i))

        permute = EigValues6.argsort()
        EigValues6 = EigValues6[permute]
        EigVectors6 = EigVectors6[:,permute]

        e_6 = np.append(e_6, EigValues6)

    e06 = []
    e16 = []

    for i in range(0, len(e_6) - 63, 64):
        e06 = np.append(e06, e_6.item(i))
        e16 = np.append(e16, e_6.item(i + 1))

    for i in range (0, len(e06)):
        e16[i] = e16[i] - e06[i]
        e06[i] = e06[i] - e06[i]

    # minimum gap

    minimum_gap6 = min(e16)
    t_min6 = s.item(e16.argmin())
    
    print('The band gap for the 6 qubits (red line) is', minimum_gap6.real, 'Joule and occurs when s =', t_min6)
    
    return [e06, e16]

# drawing

plt.grid()
plt.plot(s, e05, c = 'black')
five_qubits, = plt.plot(s, e15, c = 'blue', label='5 qubits')

for J in [-10, -7.07, -4, -2]:
    x = ground_first(J)
    plt.plot(s, x[0], c = 'black')
    six_qubits, = plt.plot(s, x[1], c = 'red', label='6 qubits')

plt.xlabel("s = t/tA")
plt.ylabel("Energy (J)")
plt.legend(handles=[five_qubits, six_qubits])
plt.show()