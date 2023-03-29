import matplotlib.pyplot as plt
import numpy as np

def retta(x):
   return 4*x+1

x=np.arange(100)

y=retta(x)

plt.plot(x,y)