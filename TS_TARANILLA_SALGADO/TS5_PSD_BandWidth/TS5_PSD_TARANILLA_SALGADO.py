#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 12:45:05 2025

@author: victoria24
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import write

#%% ECG
fs_ecg = 1000 # Hz

# Cargo la señal
ecg_one_lead = np.load('ecg_sin_ruido.npy')
#.shape devuelve tuplas
N = ecg_one_lead.shape[0] #solo el elemento 0


plt.figure(figsize=(8, 6))
plt.subplot(2,1,1)
plt.title("ECG sin ruido")
plt.grid(True)
plt.plot(ecg_one_lead)

#PARAMETROS WELCH

cant_promedios = 30 #cambia mucho la forma, cuanto mas chico mas varianza
nperseg = N // cant_promedios
nfft = 2 * nperseg
win = "hamming"

# Welch: devuelve f (vector de frecuencias) y Pxx
#si le especifico frecuencia de muestreo  me devuelve f
#nfft mejora resolucion espectral

f, Pxx = sig.welch(ecg_one_lead, fs=fs_ecg, window = win, nperseg=nperseg, nfft=nfft)

#Gráfico de la PSD - normal
plt.subplot(2,1,2)
plt.plot(f, Pxx)
plt.title("Densidad Espectral de Potencia (Welch)")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.xlim([0, 50]) #como es pasabajos, limito
plt.tight_layout()
plt.show()


#%% PPG

fs_ppg = 400 # Hz
ppg = np.load('ppg_sin_ruido.npy')

N_ppg = ppg.shape[0] 

plt.figure(figsize=(8, 6))
plt.subplot(2,1,1)
plt.title("PPG")
plt.grid(True)
plt.plot(ppg)

#PARAMETROS WELCH

cant_promedios_ppg = 20 #cambia mucho la forma, cuanto mas chico mas varianza
nperseg_ppg = N // cant_promedios_ppg
nfft_ppg = 2 * nperseg_ppg
win_ppg = "hamming"

f_ppg, Pxx_ppg = sig.welch(ppg, fs=fs_ppg, window = win_ppg, nperseg=nperseg_ppg, nfft=nfft_ppg)

#Gráfico de la PSD - PPG
plt.subplot(2,1,2)
plt.plot(f_ppg, Pxx_ppg)
plt.title("Densidad Espectral de Potencia (Welch)")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.tight_layout()
plt.xlim([0, 30]) #como es pasabajos, limito
plt.show()

#%%
# Cargar el archivo CSV como un array de NumPy

#fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
#fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
fs_audio, wav_data = sio.wavfile.read('silbido.wav')

N_audio = wav_data.shape[0] 

plt.figure(figsize=(8, 6))
plt.subplot(2,1,1)
plt.title("Audio")
plt.grid(True)
plt.plot(wav_data)

# import sounddevice as sd
# sd.play(wav_data, fs_audio)

#PARAMETROS WELCH

cant_promedios_audio = 30 #cambia mucho la forma, cuanto mas chico mas varianza
nperseg_audio = N_audio // cant_promedios_audio
nfft_audio = 1 * nperseg_audio
win_audio = "hamming"

f_audio, Pxx_audio = sig.welch(wav_data, fs=fs_audio, window = win_audio, nperseg=nperseg_audio, nfft=nfft_audio)

#Gráfico de la PSD - Audio
plt.subplot(2,1,2)
plt.plot(f_audio, Pxx_audio)
plt.title("Densidad Espectral de Potencia (Welch)")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% BLACKMAN TUKEY

# M = 2000 # M<N/5 - M tamaño de la ventana - dejar afuera lags autocorr

# # Autocorrelación completa (sec_corr_ecg)
# sec_corr_ecg = correlate(ecg_one_lead, ecg_one_lead, mode = 'full')

# # El índice de inicio y fin:
# start_idx = N - M - 1 # N-1 (centro) - M
# end_idx = N + M       # N-1 (centro) + M + 1 (regla de Python)

# r_x_hat_M = sec_corr_ecg[start_idx : end_idx] 

# %%

#ANCHO DE BANDA ECG
#area total cumsum, ultimo elemento
#PLantear proporcion de esa area para calcular ancho de banda

porcentaje = 0.99

pot_acum = np.cumsum(Pxx)
pot_acum_norm = pot_acum / pot_acum[-1]  # normaizo, estoy accediendo a la ultima pos del vector cumsum que es el total

# Encuentro el índice donde la potencia supera el 99%
index_bw = np.where(pot_acum_norm >= porcentaje)[0][0]

#np.where devuelve tupla de arrays, uno opr cada dimension
#Primer [0] → extrae el array de índices de la tupla que devuelve np.where.
#Segundo [0] → toma el primer elemento de ese array.

# Frecuencia correspondiente
freq_bw = f[index_bw]
print(freq_bw)

#%%
#ANCHO DE BANDA PPG
porcentaje_ppg = 0.99

pot_acum_ppg = np.cumsum(Pxx_ppg)
pot_acum_norm_ppg = pot_acum_ppg / pot_acum_ppg[-1]  # normaizo, estoy accediendo a la ultima pos del vector cumsum que es el total

# Encuentro el índice donde la potencia supera el 99%
index_bw_ppg = np.where(pot_acum_norm_ppg >= porcentaje_ppg)[0][0]

#np.where devuelve tupla de arrays, uno opr cada dimension
#Primer [0] → extrae el array de índices de la tupla que devuelve np.where.
#Segundo [0] → toma el primer elemento de ese array.

# Frecuencia correspondiente
freq_bw_ppg = f_ppg[index_bw_ppg]
print(freq_bw_ppg)


#%%

pot_acum_audio = np.cumsum(Pxx_audio)
pot_acum_norm_audio = pot_acum_audio / pot_acum_audio[-1]  # normaizo, estoy accediendo a la ultima pos del vector cumsum que es el total

# Encuentro el índice donde la potencia supera el 99%
index_inf_bw_audio = np.where(pot_acum_norm_audio >= 0.01)[0][0]
index_bw_audio = np.where(pot_acum_norm_audio >= porcentaje)[0][0]

#np.where devuelve tupla de arrays, uno opr cada dimension
#Primer [0] → extrae el array de índices de la tupla que devuelve np.where.
#Segundo [0] → toma el primer elemento de ese array.

# Frecuencia correspondiente
freq_inf_bw_audio = f_audio[index_inf_bw_audio]
freq_bw_audio = f_audio[index_bw_audio]
print("frecuencia cota inferior ancho banda audio", freq_inf_bw_audio)
print("frecuencia cota superior ancho banda audio", freq_bw_audio)


# %% Audio 2: La cucaracha
fs_cucaracha, wav_data_cucaracha = sio.wavfile.read('la cucaracha.wav')

N_cucaracha = wav_data_cucaracha.shape[0] 

# PSD Welch
cant_promedios_cucaracha = 9 
nperseg_cucaracha = N_cucaracha // cant_promedios_cucaracha
nfft_cucaracha = 5 * nperseg_cucaracha
win_cucaracha = "hamming"

f_cucaracha, Pxx_cucaracha = sig.welch(wav_data_cucaracha, fs=fs_cucaracha, window = win_cucaracha, nperseg=nperseg_cucaracha, nfft=nfft_cucaracha)

# Grafico
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(wav_data_cucaracha)
plt.title('La Cucaracha')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(f_cucaracha, Pxx_cucaracha)
plt.title('Densidad espectral de Potencia, Welch')
plt.xlabel('Frecuencia [Hz]')
plt.xlim((750,2200))
plt.grid(True)

plt.tight_layout()
plt.show()

# Ancho de banda
# Defino el porcentaje en base al ruido que observo
porcentaje_cucaracha = 0.99
porcentaje_inf = 0.005

pot_acum_cucaracha = np.cumsum(Pxx_cucaracha)
pot_acum_norm_cucaracha = pot_acum_cucaracha / pot_acum_cucaracha[-1] #Vector con valores de 0 a 1

# En el gráfico se observa que es pasa banda, por eso definimos limite inferior y superior
index_inf_cucaracha = np.where(pot_acum_norm_cucaracha >= porcentaje_inf)[0][0]
index_sup_cucaracha = np.where(pot_acum_norm_cucaracha >= porcentaje_cucaracha)[0][0]

freq_inf_bw_cucaracha = f_cucaracha[index_inf_cucaracha]
freq_sup_bw_cucaracha = f_cucaracha[index_sup_cucaracha]

print("frecuencia cota inferior ancho banda audio", freq_inf_bw_cucaracha)
print("frecuencia cota superior ancho banda audio", freq_sup_bw_cucaracha)

# %% Audio 3: Prueba
fs_audio2, wav_data2 = sio.wavfile.read('prueba psd.wav')
N_audio2 = wav_data2.shape[0] 

# Estimación de PSD
cant_promedios_audio2 = 25
nperseg_audio2 = N_audio2 // cant_promedios_audio2
nfft_audio2 = 3 * nperseg_audio2
win_audio2 = "hamming"

f_audio2, Pxx_audio2 = sig.welch(wav_data2, fs=fs_audio2, window = win_audio2, nperseg=nperseg_audio2, nfft=nfft_audio2)

# Grafico
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(wav_data2)
plt.title('Audio de Prueba')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(f_audio2, Pxx_audio2)
plt.title('Densidad espectral de Potencia, Welch')
plt.xlabel('Frecuencia [Hz]')
plt.xlim((-10,2750))
plt.grid(True)

plt.tight_layout()
plt.show()

# Ancho de banda
# Defino el porcentaje en base al ruido que observo
porcentaje_audio2 = 0.99

pot_acum_audio2 = np.cumsum(Pxx_audio2)
pot_acum_norm_audio2 = pot_acum_audio2 / pot_acum_audio2[-1] #Vector con valores de 0 a 1

# En el gráfico se observa que es pasa bajo, por eso definimos limite superior
index_sup_audio2 = np.where(pot_acum_norm_audio2 >= porcentaje_audio2)[0][0]

freq_sup_bw_audio2 = f_audio2[index_sup_audio2]

print("frecuencia cota superior ancho banda audio", freq_sup_bw_audio2)