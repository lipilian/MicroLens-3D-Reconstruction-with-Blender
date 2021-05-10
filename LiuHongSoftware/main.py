# %% 
from prerun import prerun
import matlab.engine
import os
import io
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# %% prerun to get datainformation from user
OpticInfo = prerun()
# %%
try:
    eng
    print('the engine already running')
except:
    names = matlab.engine.find_matlab()
    assert(len(names) == 1)
    print('The matlab shared session: {} is running'.format(names[0]))
    eng = matlab.engine.connect_matlab(name = names[0])
    
    # ! connect the matlab IO to current terminal
    out = io.StringIO()
    err = io.StringIO()
    
    eng.workspace['OpticInfo'] = OpticInfo
    matlabPath = os.path.abspath('LFIT-master')
    eng.cd(matlabPath)
# %% try calibration image 
# * OpticInfo directionary to matlab structure
# ! index start from 0 as tradinational python indexing format (first image)
# ! calibration True or False
start = time.time()
X,Y,Z,RayCounts = eng.LiuHongSingleImageProcess(OpticInfo, 0, False,  nargout = 4)
npX = (np.array(X._data, dtype = float) / OpticInfo['magnification']) * 1000
npY = (np.array(Y._data, dtype = float) / OpticInfo['magnification']) * 1000
npZ = (np.array(Z._data, dtype = float) / OpticInfo['magnification'] ** 2) * 1000
RayCounts = np.array(RayCounts._data, dtype = float)
end = time.time()
# print the calibration error from matlab to python terminal
print('--------Excuation time---------')
print('{} second needed to execute processing of single image'.format(end - start))
print('-------standard output---------')
print(out.getvalue())
print('-------------error-------------')
print(err.getvalue())

# %% filter the cloud points
npX_unique = np.unique(npX); deltaX = npX_unique[1] - npX_unique[0]
npY_unique = np.unique(npY); deltaY = npY_unique[1] - npY_unique[0]
npZ_unique = np.unique(npZ); deltaZ = npZ_unique[1] - npZ_unique[0]

del npX_unique, npY_unique, npZ_unique
maxRayCount = OpticInfo['maxRayCount']
npX = npX[RayCounts >= maxRayCount]
npY = npY[RayCounts >= maxRayCount]
npZ = npZ[RayCounts >= maxRayCount]
RayCounts = RayCounts[RayCounts >= maxRayCount]
print('{} Points left'.format(len(npX)))

print('x resolution is {} um'.format(deltaX))
print('y resolution is {} um'.format(deltaY))
print('z resolution is {} um'.format(deltaZ))


# %% KDTree filter the points 
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
print('There are {} 3d points'.format(len(RayCounts)))
searchRange = 2 * deltaX
print('KDTree local maxima searching Range is {} um'.format(searchRange))
points = np.concatenate((npX.reshape((-1,1)), npY.reshape((-1,1)), npZ.reshape((-1,1))), axis = 1)
start = time.time()
tree = KDTree(points)
neighbors = tree.query_radius(points, r = searchRange)
i_am_max = [RayCounts[i] == np.max(RayCounts[n]) for i,n in enumerate(neighbors)]
maxIndex = np.nonzero(i_am_max)
maxPoints = points[maxIndex]; maxPointsRay = RayCounts[maxIndex]
clustering = DBSCAN(eps = searchRange, min_samples = 1).fit(maxPoints)
clusterLabel = clustering.labels_
numCenters = np.max(clusterLabel) + 1
pointsCenter = np.zeros((numCenters, 3))
pointsRayCount = np.zeros((numCenters, 1))
for label in range(numCenters):
    pointsIndex = np.where(clusterLabel == label)
    validPoints = maxPoints[pointsIndex]
    validRays = maxPointsRay[pointsIndex]
    center = np.mean(validPoints, axis = 0)
    centerRay = np.mean(validRays, axis = 0)
    pointsCenter[label, :] = center
    pointsRayCount[label] = centerRay
end = time.time()
print('--------Excuation time---------')
print('{} second needed to execute kdTree find local maxima'.format(end - start))

 


# %% visualize 
# interactive mode on
%matplotlib 
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
#p = ax.scatter(npX, npY, npZ, c = RayCounts, cmap = 'jet', vmin = 100, vmax = 300, s= 2)
#L = ax.scatter(maxPoints[:,0], maxPoints[:,1], maxPoints[:,2], c = 'r', s = 30)
Lcenter = ax.scatter(pointsCenter[:,0], pointsCenter[:,1],pointsCenter[:,2],c = 'k', s = 20)
ax.set_xlabel('X(um)')
ax.set_ylabel('Y(um)')
ax.set_zlabel('Z(um)')
fig.colorbar(p)






# %%
eng.quit()  
del eng
# %%
