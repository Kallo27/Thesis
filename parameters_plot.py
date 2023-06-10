import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r'.\Advantage_system6_2_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

h = 6.62607015e-25

A = A * h
B = B * h
# drawing

plt.grid()
A_s, = plt.plot(s, A, c = 'blue', label = 'A(s)')
B_s, = plt.plot(s, B, c = 'red', label = 'B(s)')
plt.xlabel("s = t/tA")
plt.ylabel("Energy (J)")
plt.legend(handles=[A_s, B_s])
plt.show()
print(h)