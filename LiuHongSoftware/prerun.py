# %%
import PySimpleGUI as sg
from Setting import lfpivSetting

# %% define main function
def prerun():
    sg.theme('BluePurple')
    # load the parameter from the setting class
    Settings = lfpivSetting()
    Settings.loadDict()
    parameters = Settings.get_setting()
    
    layout = [[sg.Text('Enter your optic information here:', font=('Helvetica', 16))]]
    for k, v in parameters.items():
        v = str(v) 
        if 'path' in k:
            layout.append([sg.Text(k), sg.In(default_text = v, size=(25,1), enable_events=True ,key=k), sg.FolderBrowse()])
        else:
            layout.append([sg.Text(k, size =(30,1)), sg.In(default_text = v, size = (10,1), key = k)])
    layout.append([sg.Button('Calibrate'), sg.Button('Exit')])
    layout.append([sg.Text('Contact Liu Hong: liuhong2@illinois.edu if you have any problem',font=('Helvetica', 16))])

    window = sg.Window('Please input your optic setup', layout)

    while True:  # Event Loop
        event, values = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Calibrate':
            for k, v in parameters.items():
                if 'path' in k or 'type' in k:
                    parameters[k] = values[k]
                else:
                    parameters[k] = float(values[k])
            Settings.update_setting(parameters)
            Settings.saveDict()
            sg.popup_ok('Open your matlab, and keep it running')
            break
    window.close()
    return parameters
