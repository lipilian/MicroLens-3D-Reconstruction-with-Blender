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
NumF = 35 # give the number of frames for processing
fps = 7 # camera sampling frequency
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
# ! LiuHongSingleImageProcess arguement: OpticInfo, frame number, calibration, batch
X,Y,Z,RayCounts = eng.LiuHongSingleImageProcess(OpticInfo, NumF, False, True, nargout = 4)
npX = np.array(X._data, dtype = float) 
npY = np.array(Y._data, dtype = float) 
npZ = np.array(Z._data, dtype = float) 
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
plt.figure()
plt.hist(RayCounts[0:len(npX)], 100, (0,600))
plt.yscale('log')

# %% filter the cloud points
npX_unique = np.unique(npX); deltaX = npX_unique[1] - npX_unique[0]
npY_unique = np.unique(npY); deltaY = npY_unique[1] - npY_unique[0]
npZ_unique = np.unique(npZ); deltaZ = npZ_unique[1] - npZ_unique[0]
del npX_unique, npY_unique, npZ_unique
maxRayCount = OpticInfo['maxRayCount']
#? optional for adjust only
maxRayCount = 100

npX_valid = []# npX_valid array for npX
npY_valid = []# npY_valid array for npY
npZ_valid = []# npZ_valid array for npZ
RayCounts_Valid = [] #Valid RayCounts corresponding to npX,npY,npZ
singleImgCounts = len(npX)
for i in range(NumF):
    RayCountByFrame = RayCounts[i * singleImgCounts: (i+1)*singleImgCounts]
    npX_valid.append(npX[RayCountByFrame >= maxRayCount])
    npY_valid.append(npY[RayCountByFrame >= maxRayCount])
    npZ_valid.append(npZ[RayCountByFrame >= maxRayCount])
    RayCounts_Valid.append(RayCountByFrame[RayCountByFrame >= maxRayCount])
    print('{} Points left for Frame {}'.format(len(npX_valid[i]), i))

print('x resolution is {} um'.format(deltaX))
print('y resolution is {} um'.format(deltaY))
print('z resolution is {} um'.format(deltaZ))


# %% visualize the cloud points
for i in range(35):
    FrameID = i
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(npX_valid[FrameID], npY_valid[FrameID], npZ_valid[FrameID], c = RayCounts_Valid[FrameID], cmap = 'jet', vmin = maxRayCount, vmax = 300, s= 1)
    fps = 7.0
    ax.view_init(15.3284, -2.6613)
    ax.set_title('FrameID = {}, t = {}s'.format(FrameID, FrameID/fps))
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1,14)
    ax.set_zlim(-40,40)
    fig.savefig(OpticInfo['output_path'] + '/' + str(FrameID).zfill(5) + '.jpg', dpi = 300)


# %% KDTREE + DBSCAN
#from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
searchRange = 4 * deltaX
AllFrameMaxPoints = []
AllFrameCenterPoints = []
start = time.time()
for FrameID in range(NumF):
    #for i in range(len(npX_valid)):
    #print('There are {} 3d points'.format(len(npX_valid[testFrameID])))
    #print('KDTree local maxima searching Range is {} mm'.format(searchRange))
    points = np.concatenate((npX_valid[FrameID].reshape((-1,1)), npY_valid[FrameID].reshape((-1,1)), npZ_valid[FrameID].reshape((-1,1))), axis = 1)
    weight = np.array([1,1,0.2])
    weight = weight/np.linalg.norm(weight)
    tree = BallTree(points, metric = 'wminkowski', p = 2, w = weight)# use Ball tree 
    neighbors = tree.query_radius(points, r = searchRange) # search neighbor
    i_am_max = [RayCounts_Valid[FrameID][k] == np.max(RayCounts_Valid[FrameID][n]) for k,n in enumerate(neighbors)]
    maxIndex = np.nonzero(i_am_max)
    maxPoints = points[maxIndex]
    maxPointsRay = RayCounts_Valid[FrameID][maxIndex]
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
    AllFrameMaxPoints.append(np.concatenate((maxPoints, np.expand_dims(maxPointsRay, axis = 1)), axis = 1))
    AllFrameCenterPoints.append(np.concatenate((pointsCenter, pointsRayCount), axis = 1))
                             
end = time.time()
print('--------Excuation time---------')
print('{} second needed to execute kdTree find local maxima'.format(end - start))




# %%
for FrameID in range(NumF):
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    pt = ax.scatter(AllFrameMaxPoints[FrameID][:,0], AllFrameMaxPoints[FrameID][:,1], AllFrameMaxPoints[FrameID][:,2], c = AllFrameMaxPoints[FrameID][:,3], cmap = 'jet', vmin = maxRayCount, vmax = 300, s= 1)
    ptCenter = ax.scatter(AllFrameCenterPoints[FrameID][:,0], AllFrameCenterPoints[FrameID][:,1], AllFrameCenterPoints[FrameID][:,2], c = 'k', s= 5)
    ax.view_init(15.3284, -2.6613)
    ax.set_title('FrameID = {}, t = {}s'.format(FrameID, FrameID/fps))
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1,14)
    ax.set_zlim(-40,40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.savefig(OpticInfo['output_path'] + '/Center/' + str(FrameID).zfill(5) + '.jpg', dpi = 300)


#%% trackpy tracking these particles 
import trackpy as tp
column_names = ['x', 'y', 'z', 'mass', 'frame']
df = pd.DataFrame(columns = column_names)

for FrameID in range(NumF):
    ptCenter = AllFrameMaxPoints[FrameID]
    ptCenter = np.append(ptCenter, np.ones([len(ptCenter), 1]) * FrameID, axis = 1)
    ptCenter = pd.DataFrame(ptCenter, columns = column_names)
    Alldf.append(ptCenter)
    df = pd.concat([df, ptCenter])
df.reset_index()
df['xum'] = df['x']/20 * 1000
df['yum'] = df['y']/20 * 1000
df['zum'] = df['z']/400 * 1000
pred = tp.predict.NearestVelocityPredict()
t = pred.link_df(df, search_range = (0.4,6,0.4),pos_columns = ['x','y', 'z'],memory = 0)
#t = tp.link(df,search_range = (0.4,6,0.4),pos_columns = ['x','y', 'z'], memory = 1, predictor = pred)

# %%
traj = 0
NumParticle = t['particle'].max()
velocityProfile = [] # x, y, z, vx, vy, yz
for i in range(NumParticle):
    tNew = t[t['particle'] == i]
    if len(tNew) > 1:
        for i in range(len(tNew) - 1):
            p1 = tNew.iloc[i][['xum','yum','zum']].to_numpy()
            p2 = tNew.iloc[i + 1][['xum','yum','zum']].to_numpy()
            v_particle = p2 - p1
            velocityProfile.append(np.concatenate((p1, v_particle)))
velocityProfile = np.array(velocityProfile)
velocityProfile = velocityProfile[velocityProfile[:,4] < 0]
#velocityMag = np.linalg.norm(velocityProfile[:,4], axis = 1)
velocityMag = np.abs(velocityProfile[:,4])

points = velocityProfile[:,[0,2]] # only use x, z data 
tree = BallTree(points, leaf_size = 2)
neighbors = tree.query_radius(points, r = 10) # search neighbor
factor = 0.9

filteredVelocityMag = velocityMag
for k, n in enumerate(neighbors):
    filteredVelocityMag[k] = factor * np.median(velocityMag[n]) + (1-factor)*filteredVelocityMag[k]
velocityMag = np.abs(velocityProfile[:,4])    
ax = plt.figure().add_subplot(projection='3d')
#ax.scatter(Map2D[:,0], Map2D[:,1], np.abs(Map2D[:,2]))
# calculate the velocity um per second
#points = points[velocityMag<125,:]
#velocityMag = velocityMag[velocityMag < 125]
#velocityMag[points[:,1] > 60] = 0
#velocityMag[points[:,0] > 158] = 0

points = points[filteredVelocityMag<125,:]
filteredVelocityMag = filteredVelocityMag[filteredVelocityMag < 125]
filteredVelocityMag[points[:,1] > 70] = 0
filteredVelocityMag[points[:,0] > 158] = 0

newPoints = points[points[:,1]>= 30,:]
newfilteredVelocityMag = filteredVelocityMag[points[:,1]>=30]
newPoints[:,1] = -60 - newPoints[:,1]

points = np.concatenate((points,newPoints), axis = 0)
filteredVelocityMag = np.concatenate((filteredVelocityMag,newfilteredVelocityMag), axis = 0)

newPoints = points[points[:,0]>=120,:]
newfilteredVelocityMag = filteredVelocityMag[points[:,0]>=120]
newPoints[:,0] = 120 - newPoints[:,0]

points = np.concatenate((points,newPoints), axis = 0)
filteredVelocityMag = np.concatenate((filteredVelocityMag,newfilteredVelocityMag), axis = 0)


tree = BallTree(points, leaf_size = 2)
neighbors = tree.query_radius(points, r = 10) # search neighbor
factor = 0.9

newfilteredVelocityMag = filteredVelocityMag
for k, n in enumerate(neighbors):
    filteredVelocityMag[k] = factor * np.median(filteredVelocityMag[n]) + (1-factor)*newfilteredVelocityMag[k]




ax.scatter(points[:,0], points[:,1], newfilteredVelocityMag)
#ax.scatter(newPoints[:,0], newPoints[:,1], newfilteredVelocityMag, c = 'r')
filteredVelocityMag = newfilteredVelocityMag


ax.set_zlim([0,200])
ax.set_ylabel('y(um)')
ax.set_xlabel('x(um)')
ax.set_zlabel('z(um/s)')
savemat('../PaperDataResult/velocity_scatter.mat', {'x':points[:,0], 'y':points[:,1], 'v':filteredVelocityMag})

# %%
colorLow = 0
colorHigh = 150
c = velocityMag
c = (c.ravel() - colorLow) / (colorHigh - colorLow)
c = np.concatenate((c, np.repeat(c, 2)))
c = plt.cm.jet(c)

ax = plt.figure().add_subplot(projection='3d')

ax.quiver(velocityProfile[:,0], velocityProfile[:,1], velocityProfile[:,2], 0* velocityProfile[:,3], -filteredVelocityMag, 0*velocityProfile[:,5],colors = c, normalize = True, length = 40)


# %%
from scipy.interpolate import griddata
Map2D = velocityProfile[:,[0,2,4]]
xi = np.arange(-40,165,5)
yi = np.arange(-130,75,5)
xi,yi = np.meshgrid(xi,yi)
zi = griddata(points,filteredVelocityMag,(xi,yi), method = 'cubic')

ax = plt.figure().add_subplot(projection='3d')
#ax.scatter(Map2D[:,0], Map2D[:,1], np.abs(Map2D[:,2]))
zi = zi * fps # calculate the velocity um per second
ax.scatter(xi, yi, zi)

ax.set_zlim([0,1000])
ax.set_ylabel('y(um)')
ax.set_xlabel('x(um)')
ax.set_zlabel('z(um/s)')
# %% refill entire domain  
# ! get largest x with z value
(m,n) = xi.shape
max_x_index = 0
for i in range(m):
    for j in range(n):
        if not np.isnan(zi[i,j]):
            if j > max_x_index:
                max_x_index = j
print('max x index is {} in total {}'.format(max_x_index, n))   
xmin = xi[0,max_x_index] - 200
xmax = xi[0,max_x_index]
ymax = 60
ymin = -140
current_valid_x = [0,200] # actual z valid until to 160
current_valid_y = [-60,60]
new_xi = np.arange(-40,165,5)
new_yi = np.arange(-140,65,5)
new_xi, new_yi = np.meshgrid(new_xi, new_yi)
# ! cut zi until xi = 160
new_zi = np.zeros([41,41])
(m,n) = new_xi.shape
for i in range(m):
    for j in range(n):
        x = new_xi[i,j]
        y = new_yi[i,j]
        col_index = np.where(xi[0,:] == x)
        row_index = np.where(yi[:,0] == y)

        if col_index[0].size > 0 and row_index[0].size > 0:
            new_zi[i,j] = zi[row_index[0][0], col_index[0][0]]

for i in range(int(20)):
    new_zi[:,i] = new_zi[:,-i-1]          

for j in range(int(20)):
    new_zi[j,:] = new_zi[-j-1,:]
    
new_xi = new_xi - 60
new_yi = new_yi + 40

for i in range(m):
    for j in range(n):
        if np.abs(new_xi[i,j]) == 100 or np.abs(new_yi[i,j]) == 100:
            new_zi[i,j] = 0      
            
ax = plt.figure().add_subplot(projection='3d')

ax.scatter(new_xi, new_yi, new_zi)

ax.set_zlim([0,1000])
ax.set_ylabel('y(um)')
ax.set_xlabel('x(um)')
ax.set_zlabel('z(um/s)')

from scipy.io import savemat
savemat('../PaperDataResult/velocity.mat', {'x':new_xi, 'y':new_yi, 'v':new_zi})

# %% Load back from matlab 
from scipy.io import loadmat
data = loadmat('../PaperDataResult/data.mat')

y = np.arange(-100,105,5)
z = np.arange(-100,105,5)
yi,zi = np.meshgrid(y,z)
vi = data['NewV']
[m,n] = yi.shape
xi = np.ones((m,n)); xi *= 200
u = vi; v = np.zeros((m,n)); w = np.zeros((m,n))
colorLow = 0
colorHigh = np.max(vi)
c = u.flatten()
c = (c - colorLow) / c.ptp()
#c = np.concatenate((c, np.repeat(c, 2)))
#c = plt.cm.jet(c)
%matplotlib
ax = plt.figure().add_subplot(projection='3d')
ax.quiver(xi.flatten(),yi.flatten(),zi.flatten(),u.flatten(),v.flatten(),w.flatten(), c , cmap = plt.cm.jet, length = 0.02)
ax.set_ylabel('y(um)')
ax.set_xlabel('x(um)')
ax.set_zlabel('z(um)')
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
