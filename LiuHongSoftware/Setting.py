# %% 
import pickle
# %%
class lfpivSetting:
    def __init__(self, Setting_dict = {}):
        if not Setting_dict:
            Setting_dict = {\
                'target_path' : '', \
                'calibration_path' : '', \
                'output_path' : '', \
                'sensor_type' : 'rect', \
                'pixel_mm' : 0.00435,\
                'MLA_F_mm' : 3.75,\
                'MLA_size_mm' : 0.125,\
                'sensorH_resolution' : 2400,\
                'num_Micro_X' : 130,\
                'num_Micro_Y' : 80,\
                'dmin_mm' : -20,\
                'dmax_mm' : 20,\
                'dnum' : 100,\
                'xmin_mm' : -2,\
                'xmax_mm' : 20,\
                'ymin_mm' : -2,\
                'ymax_mm' : 12,\
                'scaleFactor (used for supersample x, y refocused resolution' : 1\        
            }
        self.Setting_dict = Setting_dict  
        
    def get_setting(self):
        return self.Setting_dict     
    
    def update_setting(self, data):
        self.Setting_dict = data
      
    def saveDict(self, savePath = './cache/Setting.pkl'):
        with open(savePath,'wb') as handle:
            pickle.dump(self.Setting_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    def loadDict(self, loadPath = './cache/Setting.pkl'):
        try:
            infile = open(loadPath, 'rb')
            Setting_dict = pickle.load(infile)
            if Setting_dict:
                self.Setting_dict = Setting_dict
            infile.close()
        except FileNotFoundError:
            print('The Setting cache path is not not exist, read failed')
        