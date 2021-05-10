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
npX = np.array(X._data, dtype = float) / OpticInfo['magnification'] * 1000
npY = np.array(Y._data, dtype = float) / OpticInfo['magnification'] * 1000
npZ = np.array(Z._data, dtype = float) / OpticInfo['magnification'] ** 2 * 1000
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
npX_unique = np.unique(npX)
npY_unique = np.unique(npY)
npZ_unique = np.unique(npZ)
print('x resolution is {} um'.format(deltaX))
print('y resolution is {} um'.format(deltaY))
print('z resolution is {} um'.format(deltaZ))
# %% visualize 
# interactive mode on
%matplotlib 
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(npX, npY, npZ, c = RayCounts, cmap = 'jet', vmin = 100, vmax = 300, s= 2)
ax.set_xlabel('X(um)')
ax.set_ylabel('Y(um)')
ax.set_zlabel('Z(um)')
fig.colorbar(p)

# %% KDTree filter the points 
from sklearn.neighbors import KDTree
print('There are {} 3d points'.format(len(RayCounts)))
searchRange = 2 * deltaX
print('KDTree local maxima searching Range is {} um'.format(searchRange))
points = np.concatenate((npX.reshape((-1,1)), npY.reshape((-1,1)), npZ.reshape((-1,1))), axis = 1)
start = time.time()



end = time.time()


 


# %% valid points calculation cost based on KDTree or clusters







# %%
eng.quit()  
del eng
# %%
