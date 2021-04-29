# %% 
from prerun import prerun
import matlab.engine
import os
# %% prerun to get datainformation from user
OpticInfo = prerun()
# %%
names = matlab.engine.find_matlab()
assert(len(names) == 1)
print('The matlab shared session: {} is running'.format(names[0]))
eng = matlab.engine.connect_matlab(name = names[0])
eng.workspace['OpticInfo'] = OpticInfo
matlabPath = os.path.abspath('LFIT-master')
eng.cd(matlabPath)



eng.quit()  
# %%
