# %% 
from prerun import prerun
import matlab.engine
import os
import io
import numpy as np
import time
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
npX = np.array(X._data, dtype = float).reshape(X.size, order = 'F')
npY = np.array(Y._data, dtype = float).reshape(Y.size, order = 'F')
npZ = np.array(Z._data, dtype = float).reshape(Z.size, order = 'F')
RayCounts = np.array(RayCounts._data, dtype = float).reshape(RayCounts.size, order = 'F')

end = time.time()
# print the calibration error from matlab to python terminal
print('--------Excuation time---------')
print('{} second needed to execute processing of single image'.format(end - start))
print('-------standard output---------')
print(out.getvalue())
print('-------------error-------------')
print(err.getvalue())

# %% process images from first one 


# %%
eng.quit()  
del eng
# %%
