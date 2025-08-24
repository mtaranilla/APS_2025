# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:08:08 2025

@author: Mora
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Funcion Sinusoidal

fs = 200 # Hz es la frecuencia de muestreo, la cantidad de muestras en un segundo
N = fs # se igualan para asi obtener la muestra en un segundo

def mi_funcion_sen( ax = 1, dc = 0, fx = 1, ph = 0, nn = N, fs = fs):
    Ts = 1/fs #tiempo de muestreo
    T_simulacion = N * Ts # segundos
    
    tt = np.arange(start=0, stop=T_simulacion, step = Ts) #grilla temporal
    xx = ax * np.sin( 2 * np.pi * fx * tt + ph ) + dc
    
    return tt, xx

tt, xx = mi_funcion_sen( ax = 1, dc = 0, fx = 1, ph=0, nn = N, fs = fs)

#%% A distintas frecuencias y desfasajes
t2, x2 = mi_funcion_sen( ax = 1, dc = 0, fx = 1, ph= np.pi/2, nn = N, fs = fs) #desfasaje en pi/2
t3, x3 = mi_funcion_sen( ax = 1, dc = 0, fx = 10, ph= 0, nn = N, fs = fs) 

plt.figure(1)
plt.plot( tt, xx, color = 'blue', label='Fase 0')
plt.plot( tt, x2, color = 'red', label='Fase pi/2')
plt.plot( tt, x3, color = 'magenta', label='Frecuencia 10')
plt.legend()
plt.grid(True)
plt.title('Mi Sinusoidal')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.show()
