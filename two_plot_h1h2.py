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

s1_I = np.kron(s1, I)
I_s1 = np.kron(I, s1)
s3_I = np.kron(s3, I)
I_s3 = np.kron(I, s3)
s3_s3 = np.kron(s3, s3)
zeta = np.matrix([[1, 0, 0, 0], [0, 10, 0, 0], [0, 0, 100, 0], [0, 0, 0, 1000]])

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

# objective function parameters

coupling_strength = float(input('Please enter the value of the parameter J:\n'))

biases = np.array([1, 1])

# QPU anneal parameters

data = pd.read_excel(r'.\Advantage_system6_2_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

# hamiltonians

h0 = (s1_I + I_s1)

def hf(h1, h2):
    biases[0] = h1
    biases[1] = h2
    return (biases[0] * s3_I + biases[1] * I_s3) + (coupling_strength * s3_s3)

def H(t, h1, h2):
    return - A.item(t) / 2 * h0 + B.item(t) / 2 * hf(h1, h2)
    
# eigenvalues and eigenvectors

def ground_state(h1, h2):
    for i in range (0, len(s)):
        EigValues, EigVectors = np.linalg.eig(H(i, h1, h2))
        permute = EigValues.argsort()
        EigValues = EigValues[permute]
        EigVectors = EigVectors[:,permute]
        
    for i in range(0, 4):
        EigVectors[:,0][i] = round_half_up(EigVectors[:,0][i], 1)
        
    pippo = round_half_up(np.linalg.norm(zeta * EigVectors[:,0]))
    
    print(h1, h2, EigVectors[:,0], pippo)
    if pippo == 1.0:
        return 1
    elif pippo == 10.0:
        return 2
    elif pippo == 100.0:
        return 3
    elif pippo == 1000.0:
        return 4
    else:
        return 0

z = []

for h1 in np.arange(-2, 2.1, 0.1):
    z.append([])
    for h2 in np.arange(-2, 2.1, 0.1):
         z[-1].append(ground_state(h1,h2))

xlist = np.linspace(-2, 2, 41)
ylist = np.linspace(-2, 2, 41)
X, Y = np.meshgrid(xlist, ylist)

fig, ax = plt.subplots(1,1)
cp = ax.contourf(X, Y, z)

fig.colorbar(cp)
ax.set_title('Condition: J fixed')

ax.set_xlabel('h2')
ax.set_ylabel('h1')

plt.savefig('two_plot_h1h2.pdf')
plt.savefig('two_plot_h1h2.jpg')
plt.show()