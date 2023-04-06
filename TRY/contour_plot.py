import numpy as np
import matplotlib.pyplot as plt

xlist = np.linspace(-10.0, 10.0, 1000)
ylist = np.linspace(-10.0, 10.0, 1000)
X, Y = np.meshgrid(xlist, ylist)

Z = np.sqrt(X**2 + Y**2)

fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)

fig.colorbar(cp) # Add a colorbar to a plot

ax.set_title('Filled Contours Plot')

ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')

plt.show()