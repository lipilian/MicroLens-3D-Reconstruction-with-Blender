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
import pandas as pd
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
maxRayCount = 140

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

# %% Simulation Matching only 
Points = np.load('../BlenderAlgorithmSimulationCase/z0.02Points.npy')
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

# %% DBSCAN for simulation 
%matplotlib
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
searchRange = 5 * deltaX
points = np.concatenate((npX.reshape((-1,1)), npY.reshape((-1,1)), npZ.reshape((-1,1))), axis = 1)
clustering1 = DBSCAN(eps = searchRange, min_samples = 5).fit(points)
clusterLabel1 = clustering1.labels_
print(np.max(clusterLabel1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


#! pick largest number cluster
Allone = np.ones(clusterLabel1.shape)
NumRayForLabel = []
for i in range(np.max(clusterLabel1) + 1):
    NumRayForLabel.append(np.sum(Allone[clusterLabel1 == i]))
NumRayForLabel = np.array(NumRayForLabel)
ValidLabel = NumRayForLabel.argsort()[-5:]

CenterPredict = []
for i in ValidLabel:
    plot_points = points[clusterLabel1 == i]
    
    centerNew = np.mean(plot_points, axis = 0)
    
    Rays = RayCounts[clusterLabel1 == i]
    CenterP = plot_points[Rays >= np.max(Rays)* 0.9]
    CenterP = np.mean(CenterP, axis = 0)
    CenterPredict.append(CenterP)
    #CenterPredict.append(centerNew)
    cp = ax.scatter(plot_points[:,0]/1000, plot_points[:,1]/1000, plot_points[:,2]/1000,c=Rays, cmap = 'jet',vmin = maxRayCount, vmax = 200, s= 2)
    #ax.scatter(plot_points[:,0]/1000, plot_points[:,1]/1000, plot_points[:,2]/1000, label = str(i))
CenterPredict = np.array(CenterPredict)
pp = ax.scatter(CenterPredict[:,0]/1000, CenterPredict[:,1]/1000, CenterPredict[:,2]/1000, c='r', s=20)
rp = ax.scatter(refPoints[:,0], refPoints[:,1], refPoints[:,2], c='k', s=20)
ax.set_xlabel('X(mm)')
ax.set_ylabel('Y(mm)')
ax.set_zlabel('Z(mm)')
ax.set_xlim([5, 30])
ax.set_ylim([5, 20])
ax.set_zlim([-20, 20])
ax.legend()
fig.colorbar(cp)


#%% Save the data for future use
import pickle
saveDict = {}
saveDict['RayCounts'] = RayCounts
saveDict['points'] = points
saveDict['Validlabel'] = ValidLabel
saveDict['label'] = clusterLabel1
saveDict['refPoints'] = refPoints[1:]
saveDict['Center'] = CenterPredict
with open('../BlenderAlgorithmSimulationCase/temp/z002.pkl','wb') as handle:
    pickle.dump(saveDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# %%
MaxPredictDistance = 6 #! maximum distance to filter out outliers
for pointA in refPoints:
    
    distance_2 = np.sum((CenterPredict/1000 - pointA) ** 2, axis =1 )
    pointB = CenterPredict[np.argmin(distance_2)]/1000
    if np.min(distance_2) < MaxPredictDistance ** 2:
        line3D = [pointA, pointB]; line3D = np.array(line3D)
        ax.plot3D(line3D[:,0], line3D[:,1], line3D[:,2], c = 'b')
        

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
Points[:,3] = Points[:,3] - 100.73 * 2 - 1.84 ; Points = Points[:,[0,1,3]] #! Ix, Iy, d

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
p = ax.scatter(npX/1000, npY/1000, npZ/1000, c = RayCounts, cmap = 'jet', vmin = maxRayCount, vmax = 300, s= 2)
#L = ax.scatter(maxPoints[:,0]/1000, maxPoints[:,1]/1000, maxPoints[:,2]/1000, c = 'k', s = 20)
#Lcenter = ax.scatter(pointsCenter[:,0]/1000, pointsCenter[:,1]/1000,pointsCenter[:,2]/1000,c = 'r', s = 20)
#rp = ax.scatter(refPoints[:,0], refPoints[:,1], refPoints[:,2], c='k', s=20)
ax.set_xlabel('X(mm)')
ax.set_ylabel('Y(mm)')
ax.set_zlabel('Z(mm)')
fig.colorbar(p)

# %%
eng.quit()  
del eng
# %% test
%matplotlib
testPoints = points[clusterLabel1 == 4]
testRays = RayCounts[clusterLabel1 == 4]
data = np.concatenate((testPoints, testRays.reshape((-1,1))), axis = 1)

#zArray = testPoints[:,2]; zArray = 
    


actualPoint = refPoints[5]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(testPoints[:,0]/1000, testPoints[:,1]/1000, testPoints[:,2]/1000,c=testRays, cmap = 'jet',vmin = maxRayCount, vmax = 200, s= 2)
rp = ax.scatter(actualPoint[0], actualPoint[1],actualPoint[2], c='k', s=20)
ax.set_xlabel('X(mm)')
ax.set_ylabel('Y(mm)')
ax.set_zlabel('Z(mm)')
'''
ax.set_xlim([5, 30])
ax.set_ylim([5, 20])
ax.set_zlim([-20, 20])
'''
# %% visualize for paper Demonstration
%matplotlib 
fig = plt.figure(figsize=(14,10)) 
ax = fig.add_subplot(111,projection='3d')
# ! Try to normalize each cluster



from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
points_plot = np.concatenate((npX.reshape((-1,1)), npY.reshape((-1,1)), npZ.reshape((-1,1))), axis = 1)
clustering = DBSCAN(eps = 5 * deltaX, min_samples = 1).fit(points_plot)
clusterLabel = clustering.labels_
numCenters = np.max(clusterLabel) + 1

pointsCenter = np.zeros((numCenters,3))
pointsRays = np.zeros((numCenters,))

for i in range(numCenters):
    #RayCounts[clusterLabel == i] /= np.max(RayCounts[clusterLabel == i])
    localmax = np.max(RayCounts[clusterLabel == i])
    localRayCounts = RayCounts[clusterLabel == i]
    localpoints = points_plot[clusterLabel == i]
    center = localpoints[localRayCounts == np.max(localRayCounts)]
    pointsCenter[i,:] = np.mean(localpoints, axis = 0)
    pointsRays[i] = np.max(localRayCounts)
    
xypoints = pointsCenter[:,0:2]
clustering = DBSCAN(eps = 2 * deltaX, min_samples = 1).fit(xypoints)    
xyLabel = clustering.labels_

newPoints_Center = []
for i in range(np.max(xyLabel) + 1):
    newPoints_Center.append(np.average(pointsCenter[xyLabel == i], axis = 0, weights = pointsRays[xyLabel == i]))
newPoints_Center = np.array(newPoints_Center)

RayCountsNormalized = RayCounts
for i in range(numCenters):
    RayCountsNormalized[clusterLabel == i] /= np.max(RayCounts[clusterLabel == i])

clustering = DBSCAN(eps = 15 * deltaX, min_samples = 1).fit(pointsCenter)
newLabel = clustering.labels_
numCenters = np.max(newLabel) + 1
print(numCenters)


p = ax.scatter(npX/1000, npY/1000, npZ/1000, c = RayCountsNormalized, cmap = 'jet', vmin = 0.4, vmax = 1, s= 5)

#L = ax.scatter(maxPoints[:,0]/1000, maxPoints[:,1]/1000, maxPoints[:,2]/1000, c = 'r', s = 40)

#p = ax.scatter(npX/1000, npY/1000, npZ/1000, c = clusterLabel, s= 5)

#Lcenter = ax.scatter(newPoints_Center[:,0]/1000, newPoints_Center[:,1]/1000,newPoints_Center[:,2]/1000,c = 'k', s = 40, alpha = 1)


    



ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])



# %%
