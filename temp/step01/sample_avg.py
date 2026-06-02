import numpy as np
import matplotlib.pyplot as plt

x_means=[]
N=10000
for _ in range(10000):
    xs=[]
    for i in range(N):
        x=np.random.rand()
        xs.append(x)
    mean=np.mean(xs)
    x_means.append(mean)

plt.hist(x_means,density=True,bins='auto')
plt.title(f"N={N}")
plt.xlabel('x')
plt.ylabel('probabilety')
plt.show()
