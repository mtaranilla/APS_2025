# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 22:03:49 2025

@author: Mora
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io as sio
from plantilla_TS8 import plantilla

#%% Plantilla de Diseño
fs = 1000 #Hz

nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB

wp = [1, 35] #Hz
ws = [0.1, 40] #Hz

plt.figure(1)
plt.title('Plantilla de Diseño')
plantilla(ws = ws, wp = wp, fs = fs ,ripp = ripple, att = atenuacion)
plt.xlim([0,70])
plt.ylim([-70,5])
plt.show()

#%% DISEÑO DE FIR
wp_FIR = [1, 35] #comienzo y fin banda de paso
ws_FIR = [0.1, 35.9] #comienzo y fin de banda de stop
# Elijo que sean iguales, para facilitar el armado del filtro
frecuencias = np.sort(np.concatenate((wp_FIR,ws_FIR,[0,fs/2]))) #Ordeno las frecuencias
deseado = [0,0,1,1,0,0] #Respuesta deseada para cada frecuencia.

########## METODO VENTANA ##########
# Parámtros
cant_coef_win = 11001 #Elijo impar para que el retardo sea entero
retardo_win = (cant_coef_win - 1)//2
window = 'boxcar' # Rectangular para permitir una transición rapida (lóbulo principal angosto)

fir_win= signal.firwin2(numtaps = cant_coef_win, freq = frecuencias, gain = deseado, nfreqs=int((np.ceil(np.sqrt(cant_coef_win))*8)**2)-1, fs = fs, window = window)
#Devuelve los coeficientes b del filtro.

# Cálculo de las frecuencias y rta en frecuencia del filtro.
w_win, h_win = signal.freqz(b = fir_win, worN=np.logspace(-2, 2, 1000), fs=fs) #por defecto a=1, no recursivo
phase_win = np.unwrap(np.angle(h_win))

########### METODO CUADRADOS MINIMOS ##########
cant_coef_ls = 3251
retardo_ls = (cant_coef_ls - 1)//2

fir_ls = signal.firls(numtaps=cant_coef_ls,bands=frecuencias, desired=deseado, fs=fs)
#Devuelve los coeficientes b del filtro.

#Cálculo de las frecuencias y rta en frecuencia del filtro.
w_ls, h_ls = signal.freqz(b = fir_ls, worN=np.logspace(-2, 1.9, 1000), fs=fs) #por defecto a=1, no recursivo
phase_ls = np.unwrap(np.angle(h_ls))

########### METODO PARKS McCLELLAN - REMEZ ##########
cant_coef_remez = 3251
retardo_remez = (cant_coef_remez - 1)//2
deseado_remez = [0,1,0]
peso_remez = [1,1.5,1]

fir_remez = signal.remez(numtaps=cant_coef_remez, bands=frecuencias, desired=deseado_remez, weight=peso_remez, fs=fs)

w_remez, h_remez = signal.freqz(b = fir_remez, worN=np.logspace(-2, 1.9, 1000), fs=fs) #por defecto a=1, no recursivo
phase_remez = np.unwrap(np.angle(h_remez))

#%% GRÁFICOS PARA VER SI CUMPLEN CON LA PLANTILLA
plt.figure()
# Magnitud
plt.subplot(2,1,1)
plantilla(ws = ws, wp = wp, fs = fs ,ripp = ripple, att = atenuacion)
plt.plot(w_win, 20*np.log10(np.abs(h_win)), label = 'FIR Window: Rectangular')
plt.plot(w_ls, 20*np.log10(np.abs(h_ls)), label = 'FIR Least Squears')
plt.plot(w_remez, 20*np.log10(np.abs(h_remez)), label = 'FIR Remez')
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.ylim(-80,1)
plt.xlim(-1,50)
plt.grid(True, which='both', ls=':')
plt.legend()
# Fase
plt.subplot(2,1,2)
plt.plot(w_win, phase_win, label = 'FIR Window: Rectangular')
plt.plot(w_ls, phase_ls, label = 'FIR Least Squears')
plt.plot(w_remez, phase_remez, label = 'FIR Remez')
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.tight_layout()
plt.show()

#%% FILTRADO DE ECG
fs_ecg = 1000 # Hz

sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].squeeze()
N = len(ecg_one_lead)

# FILTRADO

ecg_filt_win = signal.lfilter(b = fir_win, a = 1, x = ecg_one_lead)
ecg_filt_ls = signal.lfilter(b = fir_ls, a = 1, x = ecg_one_lead)
ecg_filt_remez = signal.lfilter(b = fir_remez, a = 1, x = ecg_one_lead)

# VISUALIZACIÓN GRAL
plt.figure(figsize=(12,10))

# Señal original
plt.subplot(4,1,1)
plt.plot(ecg_one_lead[:200000], color='k', alpha=0.5)
plt.title('ECG original')
plt.grid(True)

# Window
plt.subplot(4,1,2)
plt.plot(ecg_filt_win[retardo_win:200000+retardo_win], label='Window')
plt.plot(ecg_one_lead[:200000], color='k', alpha=0.5)
plt.title('ECG filtrado – Window')
plt.legend()
plt.grid(True)

# Least Squares
plt.subplot(4,1,3)
plt.plot(ecg_filt_ls[retardo_ls:200000+retardo_ls], label='Least Squares')
plt.plot(ecg_one_lead[:200000], color='k', alpha=0.5)
plt.title('ECG filtrado – Least Squares')
plt.legend()
plt.grid(True)

# Remez
plt.subplot(4,1,4)
plt.plot(ecg_filt_remez[retardo_remez:200000+retardo_ls], label='Remez')
plt.plot(ecg_one_lead[:200000], color='k', alpha=0.5)
plt.title('ECG filtrado – Remez')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%% REGIONES DE INTERES

###### SIN RUIDO #######
plt.figure(figsize=(12,10))

# Señal original
plt.subplot(4,1,1)
plt.plot(ecg_one_lead[4000:5500], color='k', alpha=0.5)
plt.title('ECG original')
plt.grid(True)

# Window
plt.subplot(4,1,2)
plt.plot(ecg_filt_win[4000+retardo_win:5500+retardo_win], label='Window')
plt.plot(ecg_one_lead[4000:5500], color='k', alpha=0.5)
plt.title('ECG filtrado – Window')
plt.legend()
plt.grid(True)

# LS
plt.subplot(4,1,3)
plt.plot(ecg_filt_ls[4000+retardo_ls:5500+retardo_ls], label='Least Squares')
plt.plot(ecg_one_lead[4000:5500], color='k', alpha=0.5)
plt.title('ECG filtrado – Least Squares')
plt.legend()
plt.grid(True)

# Remez
plt.subplot(4,1,4)
plt.plot(ecg_filt_remez[4000+retardo_remez:5500+retardo_remez], label='Remez')
plt.plot(ecg_one_lead[4000:5500], color='k', alpha=0.5)
plt.title('ECG filtrado – Remez')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,10))

# Señal original
plt.subplot(4,1,1)
plt.plot(ecg_one_lead[10000 : 11000], color='k', alpha=0.5)
plt.title('ECG original')
plt.grid(True)

# Window
plt.subplot(4,1,2)
plt.plot(ecg_filt_win[10000+retardo_win:11000+retardo_win], label='Window')
plt.plot(ecg_one_lead[10000:11000], color='k', alpha=0.5)
plt.title('ECG filtrado – Window')
plt.legend()
plt.grid(True)

# LS
plt.subplot(4,1,3)
plt.plot(ecg_filt_ls[10000+retardo_ls: 11000+retardo_ls], label='Least Squares')
plt.plot(ecg_one_lead[10000 : 11000], color='k', alpha=0.5)
plt.title('ECG filtrado – Least Squares')
plt.legend()
plt.grid(True)

# Remez
plt.subplot(4,1,4)
plt.plot(ecg_filt_remez[10000+retardo_remez : 11000+retardo_remez], label='Remez')
plt.plot(ecg_one_lead[10000 : 11000], color='k', alpha=0.5)
plt.title('ECG filtrado – Remez')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

##### CON RUIDO ######
regs_interes = (
    np.array([5, 5.2])  * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs,
)

regs_interes = [(int(a), int(b)) for a, b in regs_interes]

filtros = [
    (ecg_filt_win, "Window", retardo_win),
    (ecg_filt_ls,  "Least Squares", retardo_ls),
    (ecg_filt_remez, "Remez", retardo_remez),
]

for (i0, i1) in regs_interes:
    
    plt.figure(figsize=(18,4))

    for j, (filt, nombre, retardo) in enumerate(filtros, start=1):

        plt.subplot(1, 3, j)
        plt.plot(ecg_one_lead[i0:i1], color='k', alpha=0.4, label="Original")
        plt.plot(filt[i0+retardo:i1+retardo], label=nombre)
        plt.title(f"{nombre}\n(min {i0/(fs*60):.1f} – {i1/(fs*60):.1f})")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()