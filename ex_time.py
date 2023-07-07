import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.read_excel(r'.\energies_kp.xlsx', sheet_name = "time_kp")

n_qubits = np.array(data['n_qubits'])
time_qa = np.array(data['mean_qa'])
time_c = np.array(data['mean_c'])

fig = plt.figure(1, dpi=120)
ax = fig.add_subplot(1,1,1)

tqa, = plt.plot(n_qubits, time_qa, label="QPU access time")
tc, = plt.plot(n_qubits, time_c, label="CPU time")

ax.set_yscale('log')
plt.xlabel("Number of objects")
plt.ylabel("Execution time (s)")

plt.savefig('compilation_time_qa.pdf')
plt.savefig('compilation_time_qa.png')

plt.legend(handles = [tqa, tc], loc = 'upper left')

plt.show()