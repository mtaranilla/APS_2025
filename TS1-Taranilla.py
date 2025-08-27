# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 13:28:44 2025

@author: Mora
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sign

#%% Punto 1

def mi_funcion_sen( ax = 1, dc = 0, fx = 1, ph = 0, nn = 1000, fs = 1000):
    Ts = 1/fs #tiempo de muestreo
    T_simulacion = nn * Ts # segundos
    
    tt = np.arange(start=0, stop=T_simulacion, step = Ts) #grilla temporal
    xx = ax * np.sin( 2 * np.pi * fx * tt + ph ) + dc
    
    return tt, xx

def mi_funcion_cuadrada ( ax = 1, dc = 0, fx = 1, ph = 0, nn = 1000, fs = 1000):
    Ts = 1/fs #tiempo de muestreo
    T_simulacion = nn * Ts # segundos
    
    tc = np.arange(start=0, stop=T_simulacion, step = Ts) #grilla temporal
    xc = ax * np.sign(np.sin( 2 * np.pi * fx * tt + ph ) + dc) #utilizo la funcion sign, que me devuelve -1 para valores negativos, 1 para positivos y 0 para 0.
    
    return tc, xc

def energia (xx):
    E=0
    long = len(xx)
    for k in range(long):
        E = E + (xx[k])**2
    return E

#Para cada Item utilicé misma cantidad de muestras y tiempo de muestreo
fs = 40000
Ts = 1/fs
N = 1000
T_simulacion = N/fs

# Item 1: Una señal sinusoidal de 2KHz
tt, x1 = mi_funcion_sen(fx = 2000, nn = N, fs = fs)
potx1 = np.mean(x1**2)

print(f'La potencia para la sinusoidal de 2KHz es de {potx1} W')

# Item 2: Misma señal amplificada y desfazada en π/2
tt, x2 = mi_funcion_sen(ax=2, fx = 2000, ph = np.pi/2, nn = N, fs = fs)
potx2 = np.mean(x2**2)

print(f'La potencia para la sinusoidal de 2KHz, Amplitud 2 y fase π/2 es de {potx2} W')

# Item 3: Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.
tt, xaux = mi_funcion_sen(fx = 1000, nn = N, fs = fs)
x3 = x1 * xaux
potx3 = np.mean(x3**2)

print(f'La potencia para la señal modulada es de {potx3} W')

# Item 4: Señal anterior recortada al 75% de su potencia.
a4 = 1*0.75 #reduzco la amplitud en un 75%
x4 = np.clip(x1, -a4, a4)
potx4 = np.mean(x4**2)

print(f'La potencia para la señal clippeada es de {potx4} W')

# Item 5: Una señal cuadrada de 4KHz.
tc, x5 = mi_funcion_cuadrada (fx = 4000, nn = N, fs = fs)
potx5 = np.mean(x5**2)

print(f'La potencia para la señal cuadrada de 4KHz es de {potx5} W')

# Item 6: Un pulso rectangular de 10ms.
tp=np.arange(0,T_simulacion,Ts)

x6=np.zeros(1000)
x6[400:800]=1

enx6 = energia(x6)

print(f'La energia para el pulso rectangular es de {enx6} J')

#En cada caso indique tiempo entre muestras, número de muestras y potencia o energía según corresponda.

# Graficos de Punto 1
plt.figure(1)
plt.plot(tt, x1, color='red')
plt.title('Sinusoidal de 2kHz')
plt.grid(True)
plt.legend()
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')

plt.figure(2)
plt.plot(tt, x2, color='magenta')
plt.title('Sinusoidal de 2kHz, Amplitud 2 y desfasada en π/2')
plt.grid(True)
plt.legend()
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')

plt.figure(3)
plt.plot(tt, x3, color='green')
#plt.plot(tt, x1)
#plt.plot(tt, xaux)
plt.title('Sinusoidal de 2kHz modulada con una igual, pero de frecuencia 1kHz')
plt.grid(True)
plt.legend()
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')

plt.figure(4)
plt.plot(tt, x4, color='blue')
plt.title('Sinusoidal de 2kHz clippeada al 75% de la Amplitud')
plt.grid(True)
plt.legend()
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')

plt.figure(5)
plt.plot(tt, x5, color='grey')
plt.title('Cuadrada de 4kHz')
plt.grid(True)
plt.legend()
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')

plt.figure(6)
plt.plot(tt, x6, color='red')
plt.title('Pulso de 10ms')
plt.grid(True)
plt.legend()
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')

plt.show()

#%% Punto 2: Verificar ortogonalidad entre la primera señal y las demás.
#Para verificar la ortogonalidad, debo hacer el producto interno entre los arrays, y estos deben dar 0
prodx1x2 = np.dot(x1,x2)
prodx1x3 = np.dot(x1,x3)
prodx1x4 = np.dot(x1,x4)
prodx1x5 = np.dot(x1,x5)
prodx1x6 = np.dot(x1,x6)

print(f'{prodx1x2},{prodx1x3},{prodx1x4},{prodx1x5},{prodx1x6}')

#No son todas ortogonales

#%% Punto 3: Graficar la autocorrelación de la primera señal y la correlación entre ésta y las demás
#Para calcular la correlación utilizo la funcion de scipy
def correlacion (v1, v2):
    rv1v2 = sign.correlate(v1,v2)
    eje_rv1v2 = np.arange(-(N), N-1)
    return rv1v2, eje_rv1v2

rx1x1, eje_correlx1 = correlacion(x1,x1)
rx1x2, eje_correlx2 = correlacion(x1,x2)
rx1x3, eje_correlx3 = correlacion(x1,x3)
rx1x4, eje_correlx4 = correlacion(x1,x4)
rx1x5, eje_correlx5 = correlacion(x1,x5)
rx1x6, eje_correlx6 = correlacion(x1,x6)

#Grafico las correlaciones
plt.figure(7)
plt.plot(eje_correlx1, rx1x1, color='red')
plt.title('Autocorrelación de Sinusoidal de 2KHz')
plt.grid(True)
plt.xlabel('Retardo k')
plt.ylabel('Valor de Correlación')

plt.figure(8)
plt.plot(eje_correlx2, rx1x2, color='magenta')
plt.title('Correlación de Sinusoidal de 2KHz con otra de 2KHz, doble de amplitud y desfasada en pi/2')
plt.grid(True)
plt.xlabel('Retardo k')
plt.ylabel('Valor de Correlación')

plt.figure(9)
plt.plot(eje_correlx3, rx1x3, color='green')
plt.title('Correlación de Sinusoidal de 2KHz con la señal modulada')
plt.grid(True)
plt.xlabel('Retardo k')
plt.ylabel('Valor de Correlación')

plt.figure(10)
plt.plot(eje_correlx4, rx1x4, color='blue')
plt.title('Correlación de Sinusoidal de 2KHz con la señal clippeada')
plt.grid(True)
plt.xlabel('Retardo k')
plt.ylabel('Valor de Correlación')

plt.figure(11)
plt.plot(eje_correlx5, rx1x5, color='grey')
plt.title('Correlación de Sinusoidal de 2KHz con la señal cuadrada')
plt.grid(True)
plt.xlabel('Retardo k')
plt.ylabel('Valor de Correlación')

plt.figure(12)
plt.plot(eje_correlx6, rx1x6, color='red')
plt.title('Correlación de Sinusoidal de 2KHz con el pulso')
plt.grid(True)
plt.xlabel('Retardo k')
plt.ylabel('Valor de Correlación')

plt.show()

#%% Punto 4: Mostrar que la igualdad se cumple con señales sinosoidales, considerando α=ω⋅t, el doble de β (Use la frecuencia que desee)
#Voy a utilizar las sinusoidales realizadas anteriormente: x1 y xaux, que son iguales pero una tiene frecuencia 2KHz y la otra 1KHz
xx4i = 2*xaux*x1

#Para el otro lado del igual, utilizo una sinusoidal desfasada en pi/2 para realizar los cosenos. 
tt, xx4aux1 = mi_funcion_sen(fx = 1000,ph = np.pi/2, nn = 1000, fs = 40000)
tt, xx4aux2 = mi_funcion_sen(fx = 3000,ph = np.pi/2, nn = 1000, fs = 40000)

xx4ii=np.zeros(1000) #Inicializo el array

for k in range(1000):
    xx4ii[k] = xx4aux1[k] - xx4aux2[k]
    
#Para demostrar igualdad puedo realizar una resta entre ambas y graficarla, y deberia dar 0
xx4 = np.zeros(1000)

for k in range(1000):
    xx4[k] = xx4i[k] - xx4ii[k]

plt.figure(13)

plt.subplot(2,1,1) #Grafico un lado del igual
plt.plot(tt, xx4i)
plt.title('2sin(α)sin(β)')
plt.legend()
plt.grid(True)
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')

plt.subplot(2,1,2) #Grafico el otro lado del igual
plt.plot(tt, xx4ii, color='red')
plt.title('cos(α-β) - cos(α+β)')
plt.legend()
plt.grid(True)
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')

plt.tight_layout() #Para que se visualice bien ambos graficos y que no se superponga el texto

plt.figure(14)
plt.plot(tt, xx4, color='red')
plt.title('2sin(α)sin(β)-[cos(α-β)-cos(α+β)]')
plt.legend()
plt.grid(True)
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [Volts]')

plt.show()

#%% Bonus
