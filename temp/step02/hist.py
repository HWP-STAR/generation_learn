import numpy as np
import matplotlib.pyplot as plt

xs=np.loadtxt('height.txt')
print(xs.shape)

plt.hist(xs,bin='auto',density=True)
plt.show()
