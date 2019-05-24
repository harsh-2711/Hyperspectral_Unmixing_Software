import scipy.io
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

mat = scipy.io.loadmat('file.mat')
mat = mat['Y']
myData = np.asarray(mat)
myData = myData.reshape(10000, 224)
print(mat.shape) 

results = PCA(myData) 

print(results.fracs)

print(f'result.y - {results.fracs.shape}')

x = []
y = []
z = []
for item in results.Y:
 x.append(item[0])
 y.append(item[1])
 z.append(item[2])

plt.close('all')
fig1 = plt.figure()
ax = Axes3D(fig1)
pltData = [x,y,z] 
ax.scatter(pltData[0], pltData[1], pltData[2], 'bo')
 
xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
 
ax.set_xlabel("x-axis") 
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
ax.set_title("PCA plot")
plt.show()