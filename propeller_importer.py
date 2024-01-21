# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 02:18:45 2023

@author: Michel Gordillo
"""

import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt


def is_float(text):
    return text.replace(' ','').replace('.','').replace('-','').isnumeric()

propdir = 'E:\Documentos\Aero Design\ILSP\Propellers\PERFILES_WEB\PERFILES2\PER3_11x6.dat'


with open(propdir, 'r') as f:
    lines = f.readlines()

#Store every index of new tables delimited by the RPM header
sep = [i for i,line in enumerate(lines) if 'RPM' in line]

# RPM values
RPM_list = [float(lines[element].rstrip().split()[-1]) for element in sep]

sep.append(len(lines)-1)

table_headers = lines[sep[0]+2].split()

#Store the beggining line and end of each table in a list to slice
intervals = np.array([np.array(sep[:-1])+1,sep[1:]]).transpose()



# or, alternatively, there's the `ignore_index` option in the `pd.concat()` function:


df_list = []

for interval, RPM in zip(intervals,RPM_list):
    #Slice file
    table_lines = lines[interval[0]:interval[1]]
    
    #headers
    headers = table_lines[1].split()
    data_lines = table_lines[3:]
    data = pd.read_csv(io.StringIO(''.join(data_lines)), names = headers, delim_whitespace=True)
    data.dropna(axis=0,inplace = True)
    data['RPM'] = RPM
    data['J'] = data['J'].astype(float)
    
    data.dropna(axis=0,inplace = True)
    df_list.append(data)
    
    
df_propeller = pd.concat(df_list, ignore_index=True)    


#%% Units
df_propeller['V'] = 0.44704*df_propeller['V']
df_propeller['J'] = df_propeller['J']
df_propeller['Pe'] = df_propeller['Pe']
df_propeller['Ct'] = df_propeller['Ct']
df_propeller['Cp'] = df_propeller['Cp']
df_propeller['PWR'] = 745.7*df_propeller['PWR']
df_propeller['Torque'] = 0.112985*df_propeller['Torque']
df_propeller['Thrust'] = 4.44822*df_propeller['Thrust']
df_propeller['omega'] = (2*np.pi/60)*df_propeller['RPM']
     
#%% Clean-up
df_propeller[df_propeller < 0] = None
df_propeller.dropna(axis=0,inplace = True)


df_propeller['EnginePower'] = df_propeller["Torque"]*df_propeller["omega"]
df_propeller['PropellerPower'] = df_propeller["Thrust"]*df_propeller["V"]
df_propeller['Efficiency'] = df_propeller['PropellerPower']/df_propeller['EnginePower']

plt.close('all')

prop_avanzada = df_propeller.loc[(df_propeller['EnginePower']<1000) & (df_propeller['V']<23)]

prop_avanzada.plot(x='EnginePower', y='Thrust', c='RPM', cmap="viridis" , kind='scatter')	

prop_avanzada.plot(x='EnginePower', y='Thrust', c='V', cmap="magma" , kind='scatter')	

prop_avanzada.plot(x='V', y='Thrust', c='EnginePower', cmap="magma" , kind='scatter')	

prop_avanzada.plot(x='Thrust', y='EnginePower', c='RPM', cmap="viridis" , kind='scatter')	

prop_avanzada.plot(x='V', y='Thrust', c='RPM', cmap="tab20" , kind='scatter')	

ax = prop_avanzada.plot.scatter(x='EnginePower', y='Thrust', c='V', cmap="viridis")

for rpm in prop_avanzada['RPM'].unique():
    df = prop_avanzada[prop_avanzada['RPM'] == rpm] 
    df.plot.scatter(x='EnginePower', y='Thrust', label="RPM "+str(rpm), s = 1 , ax=ax)

from matplotlib import cm

# Plot using `.trisurf()`:

fig = plt.figure(clear=True,figsize=plt.figaspect(0.5))
#fig, (axs,axs1) = plt.subplots(1, 2, num=plot_num, clear=True)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

ax1.set_proj_type(proj_type='ortho')

surf = ax1.plot_trisurf(df_propeller.V, df_propeller.RPM, df_propeller.Efficiency, cmap=cm.jet, linewidth=0.2)

fig.colorbar(surf)

ax1.set(xlabel='V', 
       ylabel='RPM', 
       zlabel='Efficiency')

plt.title('Propeller Eficiency Chart')

plt.show()
