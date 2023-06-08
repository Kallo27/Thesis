import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r'.\DWAVE_2000Q_annealing_schedule.xlsx')
A = np.array(data['A(s) (GHz)'])
B = np.array(data['B(s) (GHz)'])
s = np.array(data['s'])

# drawing

plt.grid()
A_s, = plt.plot(s, A, c = 'blue', label = 'A(s)')
B_s, = plt.plot(s, B, c = 'red', label = 'B(s)')
plt.xlabel("s = t/tA")
plt.ylabel("Energy (GHz)")
plt.legend(handles=[A_s, B_s])
plt.show()
