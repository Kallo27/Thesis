import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r'.\DWAVE_2000Q_annealing_schedule.xlsx') 

x = list(data['s'])
y = list(data['A(s) (GHz)'])
z = list(data['B(s) (GHz)'])

plt.figure(figsize=(10,10))
plt.style.use('ggplot')
plt.scatter(x,  y,marker = ".", s = 100, edgecolors = "red",c ="red")
plt.scatter(x, z, marker = ".", s = 100, edgecolors = "blue",c = "blue")
plt.title("A(s) e B(s)")
plt.show()