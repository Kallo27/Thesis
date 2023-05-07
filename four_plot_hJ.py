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

zeta = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16]])

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

h0 = (s1_I_I_I + I_s1_I_I + I_I_s1_I + I_I_I_s1)

def hf(h, J):
    return (biases(h) * s3_I_I_I + biases(h) * I_s3_I_I + biases(h) * I_I_s3_I + biases(h) * I_I_I_s3) + (coupling_strengths(J) * s3_s3_I_I+ coupling_strengths(J) * I_s3_s3_I + coupling_strengths(J) * I_I_s3_s3 + coupling_strengths(J) * s3_I_I_s3)

def H(t, h, J):
    return - A.item(t) / 2 * h0 + B.item(t) / 2 * hf(h, J)
    
# eigenvalues and eigenvectors

def ground_state(h, J):
    for i in range (0, len(s)):
        EigValues, EigVectors = np.linalg.eig(H(i, h, J))
        permute = EigValues.argsort()
        EigValues = EigValues[permute]
        EigVectors = EigVectors[:,permute]
        
    EigVectors = np.real(EigVectors)
    pippo = round_half_up(np.linalg.norm(zeta * EigVectors[:,0]), 1)
    
    if pippo == 1.0:
        return 1
    elif pippo == 2.0:
        return 2
    elif pippo == 3.0:
        return 3
    elif pippo == 4.0:
        return 4
    elif pippo == 5.0:
        return 5
    elif pippo == 6.0:
        return 6
    elif pippo == 7.0:
        return 7
    elif pippo == 8.0:
        return 8
    elif pippo == 9.0:
        return 9
    elif pippo == 10.0:
        return 10
    elif pippo == 11.0:
        return 11
    elif pippo == 12.0:
        return 12
    elif pippo == 13.0:
        return 13
    elif pippo == 14.0:
        return 14
    elif pippo == 15.0:
        return 15
    elif pippo == 16.0:
        return 16
    else:
        return 20

z = []

for h in np.arange(-1, 1, 0.1):
    z.append([])
    for J in np.arange(-1, 1, 0.1):
         z[-1].append(ground_state(h,J))

xlist = np.linspace(-1, 0.9, 20)
ylist = np.linspace(-1, 0.9, 20)
X, Y = np.meshgrid(xlist, ylist)
 
fig, ax = plt.subplots(1,1)
cp = ax.contourf(X, Y, z)

fig.colorbar(cp) # Add a colorbar to a plot

ax.set_title('Condition: h1 = h2 = h3 = h4 = h and J12 = J23 = J34 = J41 = J')

ax.set_xlabel('J')
ax.set_ylabel('h')

plt.savefig('four_plot_hJ.pdf')
plt.savefig('four_plot_hJ.jpg')
plt.show()