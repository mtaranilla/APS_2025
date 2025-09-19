# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 11:29:55 2025

@author: Mora
"""

import numpy as np
import matplotlib.pyplot as plt

# %% PSD de senoidales
## Planteo mi senoidal
N = 1000
fs = 1000

def mi_funcion_sen(ff, nn, amp = 1, dc = 0, ph = 0, fs = 2): ##Redefino, fs como 2 para que nyquist sea 1.
    Ts = 1/fs #tiempo de muestreo
    T_simulacion = nn * Ts # segundos
    
    tt = np.arange(start=0, stop=T_simulacion, step = Ts) #grilla temporal
    xx = amp * np.sin( 2 * np.pi * ff * tt + ph ) + dc
    
    return tt, xx

k_0 = N/4
k_1 = N/4 + 0.25
k_2 = N/4 + 0.5
 
## Genero mis senoidales, todas con tt, ya que tienen la misma grilla temporal
tt, x_0 = mi_funcion_sen(ff = k_0 * fs/N, nn = N, amp = np.sqrt(2), fs = fs)
tt, x_1 = mi_funcion_sen(ff = k_1 * fs/N, nn = N, amp = np.sqrt(2), fs = fs)
tt, x_2 = mi_funcion_sen(ff = k_2 * fs/N, nn = N, amp = np.sqrt(2), fs = fs)

potx0 = np.var(x_0) #Para verificar la potencia unitaria
potx1 = np.var(x_1)
potx2 = np.var(x_2)

## Transformo Fourier
bins = np.arange(N) * fs/N

X_0 = np.fft.fft(x_0)
X_1 = np.fft.fft(x_1)
X_2 = np.fft.fft(x_2)

plt.figure(1)
plt.plot(bins, (np.abs(X_0)**2)/(N), 'o',label='k=N/4')
plt.plot(bins, (np.abs(X_1)**2)/(N), 'o',label='k=N/4+0.25')
plt.plot(bins, (np.abs(X_2)**2)/(N), 'o',label='k=N/4+0.5')
plt.title('Densidad Espectral de Potencia')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [V^2]')
plt.legend()
plt.grid(True)
plt.show()

# %% Verifico PSD por Identidad de Parseval
# Lado A
sumaA = 0

for k in range(N):
    sumaA += np.abs(x_0[k])**2

# Lado B
sumaB = 0

for k in range(N):
    sumaB += np.abs(X_0[k])**2

sumaB = sumaB/N

print(f'{sumaA}')
print(f'{sumaB}')

# Se verifica la Identidad de Parseval. 
# En este caso, si quiero verificar la potencia unitaria, puedo calcular el lado B y lo divido por N para tener potencia promedio.

def ladoA_parseval(xx):
    N = len(xx)
    suma = 0
    
    for k in range(N):
        suma += np.abs(xx[k])**2
        
    return suma

def ladoB_Parseval(xx):
    N = len(xx)
    XX = np.fft.fft(xx)
    suma = 0
    tot = 0
    
    for k in range(N):
        suma += np.abs(XX[k])**2
    
    tot = suma/N
    return tot

pot_unitaria_x0 = ladoB_Parseval(x_0)/N
pot_unitaria_x1 = ladoB_Parseval(x_1)/N
pot_unitaria_x2 = ladoB_Parseval(x_2)/N

print(f'{pot_unitaria_x0:.4}')
print(f'{pot_unitaria_x1:.4}')
print(f'{pot_unitaria_x2:.4}')

# %% Zero Padding
def padding(xx, mult = 10):
    n = len(xx)
    xpad = np.zeros(mult*n)
    
    xpad[:n] = xx
    
    return xpad

eje_padd = np.arange(10*N) *1/fs

x_0_padd = padding(x_0)
x_1_padd = padding(x_1)
x_2_padd = padding(x_2)

plt.figure(2)
plt.title('x con zero padding')
plt.plot(eje_padd, x_0_padd, label='k=N/4')
plt.plot(eje_padd, x_1_padd, label='k=N/4+0.25')
plt.plot(eje_padd, x_2_padd, label='k=N/4+0.5')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.legend()
plt.show()

## Transformo con Zero Padding
X_0_PADD = np.fft.fft(x_0_padd)
X_1_PADD = np.fft.fft(x_1_padd)
X_2_PADD = np.fft.fft(x_2_padd)

bins_padd = np.arange(10*N) * fs/(10*N)

plt.figure(3)
plt.plot(bins_padd, np.abs(X_0_PADD), 'o',label='k=N/4')
plt.plot(bins_padd, np.abs(X_1_PADD), 'o',label='k=N/4+0.25')
plt.plot(bins_padd, np.abs(X_2_PADD), 'o',label='k=N/4+0.5')
plt.title('Transformada de Fourier de distintas senoidales, con Zero Padding')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X(k)|')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(4)
plt.plot(bins_padd, (np.abs(X_0_PADD)**2)/(10*N), 'o',label='k=N/4')
plt.plot(bins_padd, (np.abs(X_1_PADD)**2)/(10*N), 'o',label='k=N/4+0.25')
plt.plot(bins_padd, (np.abs(X_2_PADD)**2)/(10*N), 'o',label='k=N/4+0.5')
plt.title('Densidad Espectral de Potencia, con Zero Padding')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [V^2]')
plt.legend()
plt.grid(True)
plt.show()

# %% Bonus
def ec_diferencias (xx, aa, bb):
    nn = len(bb)
    mm = len(aa)
    Y = np.zeros(len(xx))
    
    for n in range (len(xx)):
        y_x = 0 #Inicializo las sumas para cada n
        y_y = 0
        for k in range(nn):
            if n-k >= 0: 
                y_x += bb[k] * xx[n-k]
        for k in range(mm):
            if n-k-1 >= 0:
                y_y += aa[k] * Y[n-1-k]
        
        Y[n] = y_x + y_y
        
    return Y

def delta(nn, k = 0):
    d = np.zeros(nn)
    d[k] = 1
    
    return d

imp = delta(N)

# y[n]= 3⋅10−2⋅x[n] + 5⋅10−2⋅x[n−1] + 3⋅10−2⋅x[n−2] + 1.5⋅y[n−1] − 0.5⋅y[n−2]
aa = [1.5, -0.5]
bb = [3*(10**-2), 5*(10**-2), 3*(10**-2)]

h1 = ec_diferencias(imp,aa,bb) #Respuesta al impulso

# y[n]=x[n]+3⋅x[n−10]
a2i = [0]
b2i = [1,0,0,0,0,0,0,0,0,0,3]

h2i = ec_diferencias(imp,a2i,b2i) #Rta al Impulso

# y[n]=x[n]+3⋅y[n−10]
a2ii = [0,0,0,0,0,0,0,0,0,3]
b2ii = [1]

h2ii = ec_diferencias(imp,a2ii,b2ii) #Rta al Impulso

## Para calcular la respuesta en frecuencia, transformo Fourier las respuestas al impulso
H1 = np.fft.fft(h1)
H2 = np.fft.fft(h2i)
H3 = np.fft.fft(h2ii)

plt.figure(5, figsize=(15,10))

plt.subplot(2,1,1)
plt.plot(bins, np.abs(H1), label='Módulo H[k]')
plt.title('y[n]= 3⋅10−2⋅x[n] + 5⋅10−2⋅x[n−1] + 3⋅10−2⋅x[n−2] + 1.5⋅y[n−1] − 0.5⋅y[n−2]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(bins, np.angle(H1), label='Fase H[k]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase')
plt.grid(True)
plt.legend()

plt.show()

plt.figure(6, figsize=(15,10))

plt.subplot(2,1,1)
plt.plot(bins, np.abs(H2), label='Módulo H[k]')
plt.title('y[n]=x[n]+3⋅x[n−10]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(bins, np.angle(H2), label='Fase H[k]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase')
plt.grid(True)
plt.legend()

plt.show()

plt.figure(7, figsize=(15,10))

plt.subplot(2,1,1)
plt.plot(bins, np.abs(H3), label='Módulo H[k]')
plt.title('y[n]=x[n]+3⋅y[n−10]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(bins, np.angle(H3), label='Fase H[k]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase')
plt.grid(True)
plt.legend()

plt.show()


