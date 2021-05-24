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

# %% histogram the RayCounts 
#plt.figure()
#plt.hist(RayCounts, 100, (0,600))
#plt.yscale('log')

# %% filter the cloud points
npX_unique = np.unique(npX); deltaX = npX_unique[1] - npX_unique[0]
npY_unique = np.unique(npY); deltaY = npY_unique[1] - npY_unique[0]
npZ_unique = np.unique(npZ); deltaZ = npZ_unique[1] - npZ_unique[0]





del npX_unique, npY_unique, npZ_unique
maxRayCount = OpticInfo['maxRayCount']

#? optional for adjust only
maxRayCount = 20

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
searchRange = 5 * deltaX
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


# %% DBSCAN for simulation 
%matplotlib
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
searchRange = 5 * deltaX
points = np.concatenate((npX.reshape((-1,1)), npY.reshape((-1,1)), npZ.reshape((-1,1))), axis = 1)
clustering1 = DBSCAN(eps = searchRange, min_samples = 1).fit(points)
clusterLabel1 = clustering1.labels_
print(np.max(clusterLabel1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
CenterPredict = []
for i in range(np.max(clusterLabel1) + 1):
    plot_points = points[clusterLabel1 == i]
    Rays = RayCounts[clusterLabel1 == i]
    CenterP = plot_points[Rays >= np.max(Rays)* 0.9]
    CenterP = np.mean(CenterP, axis = 0)
    CenterPredict.append(CenterP)
    #cp = ax.scatter(plot_points[:,0]/1000, plot_points[:,1]/1000, plot_points[:,2]/1000,c=Rays, cmap = 'jet',vmin = maxRayCount, vmax = 200, s= 2)
    #ax.scatter(plot_points[:,0]/1000, plot_points[:,1]/1000, plot_points[:,2]/1000)
CenterPredict = np.array(CenterPredict)
pp = ax.scatter(CenterPredict[:,0]/1000, CenterPredict[:,1]/1000, CenterPredict[:,2]/1000, c='r', s=20)
rp = ax.scatter(refPoints[:,0], refPoints[:,1], refPoints[:,2], c='k', s=20)
ax.set_xlabel('X(mm)')
ax.set_ylabel('Y(mm)')
ax.set_zlabel('Z(mm)')
ax.set_xlim([OpticInfo['xmin_mm'], OpticInfo['xmax_mm']])
ax.set_ylim([OpticInfo['ymin_mm'], OpticInfo['ymax_mm']])
ax.set_zlim([OpticInfo['dmin_mm'], OpticInfo['dmax_mm']])
fig.colorbar(cp)
# %% Simulation Matching only 
Points = np.load('../BlenderAlgorithmSimulationCase/Points.npy')
Points = Points *1000
Points[:, [0, 1 ,2]] = Points[:, [1, 2, 0]] #! switch to x,y,z format in mm
# calculate the imaging location based on the objects' locations
Points[:, 2] = Points[:, 2] - 12.7 #! switch to x,y,s format in mm 
S_prime = 1/(1/100.73 - 1/-Points[:,2]); S_prime = S_prime.reshape((-1,1))
Points = np.append(Points, S_prime, 1) #! switch to x,y,s,s_prime format
Points[:,0] = np.multiply(Points[:,0], np.divide(Points[:,3], Points[:,2]))
Points[:,1] = np.multiply(Points[:,1], np.divide(Points[:,3], Points[:,2]))
#! switch to Ix,Iy,s,s_prime format in mm 
Points[:,3] = Points[:,3] - 100.73 * 2; Points = Points[:,[0,1,3]] #! Ix, Iy, d

%matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(Points[:,0], Points[:,1], Points[:,2], c = 'k', s = 20)

refPoints = Points
refPoints[:,0] = Points[:,0] + 35.9/2
refPoints[:,1] = -Points[:,1] + 23.9/2
# %% visualize 
# interactive mode on
%matplotlib 
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(npX/1000, npY/1000, npZ/1000, c = RayCounts, cmap = 'jet', vmin = maxRayCount, vmax = 800, s= 2)
#L = ax.scatter(maxPoints[:,0]/1000, maxPoints[:,1]/1000, maxPoints[:,2]/1000, c = 'k', s = 20)
Lcenter = ax.scatter(pointsCenter[:,0]/1000, pointsCenter[:,1]/1000,pointsCenter[:,2]/1000,c = 'r', s = 20)
rp = ax.scatter(refPoints[:,0], refPoints[:,1], refPoints[:,2], c='k', s=20)
ax.set_xlabel('X(mm)')
ax.set_ylabel('Y(mm)')
ax.set_zlabel('Z(mm)')
fig.colorbar(p)

# %%
eng.quit()  
del eng
# %%
