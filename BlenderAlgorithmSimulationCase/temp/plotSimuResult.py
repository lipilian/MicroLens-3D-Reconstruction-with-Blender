# %%
import matlab.engine
import os
import io
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import pickle
# %%
with open('z0.pkl', 'rb') as handle:
    data0 = pickle.load(handle)
with open('z-002.pkl', 'rb') as handle:
    dataNeg = pickle.load(handle)
with open('z002.pkl', 'rb') as handle:
    dataPos = pickle.load(handle)
# %%
allDistance = 0
%matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

points = data0['points']
ValidLabel = data0['Validlabel']
clusterLabel1 = data0['label']
refPoints = data0['refPoints']
CenterPredict = data0['Center']

pp = ax.scatter(CenterPredict[:,0]/1000, CenterPredict[:,1]/1000, CenterPredict[:,2]/1000, c='r', s=20)
rp = ax.scatter(refPoints[:,0], refPoints[:,1], refPoints[:,2], c='k', s=20)

MaxPredictDistance = 6 
for pointA in refPoints:
    
    distance_2 = np.sum((CenterPredict/1000 - pointA) ** 2, axis =1 )
    pointB = CenterPredict[np.argmin(distance_2)]/1000
    if np.min(distance_2) < MaxPredictDistance ** 2:
        allDistance = allDistance + np.min(distance_2)
        line3D = [pointA, pointB]; line3D = np.array(line3D)
        ax.plot3D(line3D[:,0], line3D[:,1], line3D[:,2], c = 'b')



points = dataNeg['points']
ValidLabel = dataNeg['Validlabel']
clusterLabel1 = dataNeg['label']
refPoints = dataNeg['refPoints']
CenterPredict = dataNeg['Center']



pp = ax.scatter(CenterPredict[:,0]/1000, CenterPredict[:,1]/1000, CenterPredict[:,2]/1000, c='r', s=20)
rp = ax.scatter(refPoints[:,0], refPoints[:,1], refPoints[:,2], c='k', s=20)

MaxPredictDistance = 6 
for pointA in refPoints:
    
    distance_2 = np.sum((CenterPredict/1000 - pointA) ** 2, axis =1 )
    pointB = CenterPredict[np.argmin(distance_2)]/1000
    if np.min(distance_2) < MaxPredictDistance ** 2:
        allDistance = allDistance + np.min(distance_2)
        line3D = [pointA, pointB]; line3D = np.array(line3D)
        ax.plot3D(line3D[:,0], line3D[:,1], line3D[:,2], c = 'b')


points = dataPos['points']
ValidLabel = dataPos['Validlabel']
clusterLabel1 = dataPos['label']
refPoints = dataPos['refPoints']
CenterPredict = dataPos['Center']

pp = ax.scatter(CenterPredict[:,0]/1000, CenterPredict[:,1]/1000, CenterPredict[:,2]/1000, c='r', s=20)
rp = ax.scatter(refPoints[:,0], refPoints[:,1], refPoints[:,2], c='k', s=20)

MaxPredictDistance = 6 
for pointA in refPoints:
    
    distance_2 = np.sum((CenterPredict/1000 - pointA) ** 2, axis =1 )
    pointB = CenterPredict[np.argmin(distance_2)]/1000
    if np.min(distance_2) < MaxPredictDistance ** 2:
        allDistance = allDistance + np.min(distance_2)
        line3D = [pointA, pointB]; line3D = np.array(line3D)
        ax.plot3D(line3D[:,0], line3D[:,1], line3D[:,2], c = 'b')



ax.set_xlabel('X(mm)')
ax.set_ylabel('Y(mm)')
ax.set_zlabel('Z(mm)')
ax.set_aspect('auto')
#ax.set_xlim([5, 30])
#ax.set_ylim([5, 20])
#ax.set_zlim([-20, 20])
#ax.legend()

# %%
error = np.sqrt(allDistance/15)
# %%
