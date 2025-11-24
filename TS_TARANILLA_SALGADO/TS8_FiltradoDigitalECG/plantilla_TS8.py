# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 20:44:37 2025

@author: Mora
"""

import numpy as np
import matplotlib.pyplot as plt

fs = 1000 #Hz

nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB

wp = np.array([1, 35]) #Hz
ws = np.array([0.1, 40]) #Hz

frecs = np.array([0, ws[0], wp[0], wp[1], ws[1], nyq_frec])

def plantilla(ws,wp,fs,ripp,att):
    frec_1 = np.array([0,ws[0]])
    y_1_abajo = np.array([-att, -att])
    y_1_arriba = np.array([100,100])

    y_2_arriba = np.array([-ripple,-ripple])
    y_2_abajo = np.array([-100,-100])

    frec_3 = np.array([ws[1],fs/2])
    y_3_abajo = np.array([-att,-att])
    y_3_arriba = np.array([100,100])

    #Gráfico
    plt.plot(frec_1, y_1_abajo, color='black')
    plt.plot([ws[0],ws[0]], [-atenuacion,100], color='black')
    plt.fill_between(frec_1, y_1_arriba ,y_1_abajo, color='lightblue', alpha=0.5)

    plt.plot(wp, y_2_arriba, color='black')
    plt.plot([wp[0],wp[0]],[-100,-ripple], color='black')
    plt.plot([wp[1],wp[1]],[-100,-ripple], color='black')
    plt.fill_between(wp,y_2_arriba, y_2_abajo, color='lightblue', alpha=0.5)

    plt.plot(frec_3, y_1_abajo, color='black')
    plt.plot([ws[1],ws[1]], [-atenuacion,100], color='black')
    plt.fill_between(frec_3, y_3_arriba ,y_3_abajo, color='lightblue', alpha=0.5)

    plt.grid(True)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud [dB]')
    plt.show()
    return 

#%%
# frec_1 = np.array([0,ws[0]])
# y_1_abajo = np.array([-atenuacion, -atenuacion])
# y_1_arriba = np.array([100,100])

# y_2_arriba = np.array([-ripple,-ripple])
# y_2_abajo = np.array([-100,-100])

# frec_3 = np.array([ws[1],nyq_frec])
# y_3_abajo = np.array([-atenuacion,-atenuacion])
# y_3_arriba = np.array([100,100])

# #Gráfico
# plt.figure()
# plt.title('Plantilla de Diseño')

# plt.plot(frec_1, y_1_abajo, color='black')
# plt.plot([ws[0],ws[0]], [-atenuacion,100], color='black')
# plt.fill_between(frec_1, y_1_arriba ,y_1_abajo, color='lightblue', alpha=0.5)

# plt.plot(wp, y_2_arriba, color='black')
# plt.plot([wp[0],wp[0]],[-100,-ripple], color='black')
# plt.plot([wp[1],wp[1]],[-100,-ripple], color='black')
# plt.fill_between(wp,y_2_arriba, y_2_abajo, color='lightblue', alpha=0.5)

# plt.plot(frec_3, y_1_abajo, color='black')
# plt.plot([ws[1],ws[1]], [-atenuacion,100], color='black')
# plt.fill_between(frec_3, y_3_arriba ,y_3_abajo, color='lightblue', alpha=0.5)

# plt.grid(True)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Magnitud [dB]')
# plt.show()