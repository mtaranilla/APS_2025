# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 18:46:23 2025

@author: Mora
"""

## TS2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sign

def mi_funcion_sen(ff, nn, amp = 1, dc = 0, ph = 0, fs = 2): ##Redefino, fs como 2 para que nyquist sea 1.
    Ts = 1/fs #tiempo de muestreo
    T_simulacion = N * Ts # segundos
    
    tt = np.arange(start=0, stop=T_simulacion, step = Ts) #grilla temporal
    xx = amp * np.sin( 2 * np.pi * ff * tt + ph ) + dc
    
    return tt, xx

def mi_funcion_cuadrada (ff, nn, amp = 1, dc = 0, ph = 0, fs = 2):
    Ts = 1/fs #tiempo de muestreo
    T_simulacion = nn * Ts # segundos
    
    tt = np.arange(start=0, stop=T_simulacion, step = Ts) #grilla temporal
    xc = amp * np.sign(np.sin( 2 * np.pi * ff * tt + ph ) + dc) #utilizo la funcion sign, que me devuelve -1 para valores negativos, 1 para positivos y 0 para 0.
    
    return tt, xc

#%% Primer punto
fs = 400000
Ts = 1/fs
N = 1000
T_simulacion = N/fs

# Señales de TS1
tt, x1 = mi_funcion_sen(ff = 2000, nn = N, fs = fs)

tt, x2 = mi_funcion_sen(ff = 2000, nn = N, amp = 2, ph = np.pi/2, fs = fs)

tt, xaux = mi_funcion_sen(ff = 1000, nn = N, fs = fs)
x3 = x1 * xaux

a4 = 1*0.75 #reduzco la amplitud en un 75%
x4 = np.clip(x1, -a4, a4)

tt, x5 = mi_funcion_cuadrada (ff = 4000, nn = N, fs = fs)

x6=np.zeros(N)
x6[0:400]=1

# Genero mi propia definicion de una ecuacion en diferencias
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

#y[n]= 3⋅10−2⋅x[n] + 5⋅10−2⋅x[n−1] + 3⋅10−2⋅x[n−2] + 1.5⋅y[n−1] − 0.5⋅y[n−2]
aa = [1.5, -0.5]
bb = [3*(10**-2), 5*(10**-2), 3*(10**-2)]

Y1 = ec_diferencias (x1,aa,bb)
Y2 = ec_diferencias (x2,aa,bb)
Y3 = ec_diferencias (x3,aa,bb)
Y4 = ec_diferencias (x4,aa,bb)
Y5 = ec_diferencias (x5,aa,bb)
Y6 = ec_diferencias (x6,aa,bb)

plt.figure(1)
plt.plot(tt, x1, label='Señal de Entrada')
plt.plot(tt, Y1, label='Señal de Salida')
plt.title('Sinusoidal de 2KHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(tt, x2, label='Señal de Entrada')
plt.plot(tt, Y2, label='Señal de Salida')
plt.title('Cosenoidal de 2KHz y Amplitud 2')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(3)
plt.plot(tt, x3, label='Señal de Entrada')
plt.plot(tt, Y3, label='Señal de Salida')
plt.title('Sinusoidal de 2KHz, modulada con una de 1KHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(4)
plt.plot(tt, x4, label='Señal de Entrada')
plt.plot(tt, Y4, label='Señal de Salida')
plt.title('Sinusoidal de 2KHz clippeada el 75%')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(5)
plt.plot(tt, x5, label='Señal de Entrada')
plt.plot(tt, Y5, label='Señal de Salida')
plt.title('Cuadrada de 4KHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(6)
plt.plot(tt, x6, label='Señal de Entrada')
plt.plot(tt, Y6, label='Señal de Salida')
plt.title('Pulso de 10ms')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()

## Inciso B
# Hallar la respuesta al impulso, por lo tanto debo ingresar delta.
def delta(nn, k = 0):
    d = np.zeros(nn)
    d[k] = 1
    
    return d

imp = delta(N)

h1 = ec_diferencias(imp,aa,bb)

plt.figure(7)
plt.plot(tt, h1, label = 'h(t), Salida')
plt.plot(tt, imp, label = 'Entrada')
plt.title('Respuesta al Impulso')
plt.grid(True)
plt.legend()
plt.show()

# Hago convolucion de las senales usadas anteriormente y verifico que den lo mismo.
def convolucion(xx,hh):
    Y = np.zeros(len(xx))
    for n in range(len(xx)):
        suma = 0
        for k in range(len(xx)):
            if n-k>=0:
                suma += xx[k]*hh[n-k]
        Y[n] = suma
        
    return Y

#Y1vol2 = convolucion(x1, h1)
Y1vol2 = sign.convolve(x1,h1)
Y2vol2 = sign.convolve(x2,h1) 
Y3vol2 = sign.convolve(x3,h1) 
Y5vol2 = sign.convolve(x5,h1) 

Y1vol2_cort = Y1vol2[:len(x1)]
Y2vol2_cort = Y2vol2[:len(x2)] 
Y3vol2_cort = Y3vol2[:len(x3)]
Y5vol2_cort = Y5vol2[:len(x5)]

plt.figure(8)
plt.plot(Y1vol2_cort)
plt.title('X1 en Ec Diferencias')
plt.grid(True)
#plt.legend()
plt.show()

plt.figure(9)
plt.plot(Y2vol2_cort)
plt.title('X2 en Ec Diferencias')
plt.grid(True)
#plt.legend()
plt.show()

plt.figure(10)
plt.plot(Y3vol2_cort)
plt.title('X3 en Ec Diferencias')
plt.grid(True)
#plt.legend()
plt.show()

plt.figure(11)
plt.plot(Y5vol2_cort)
plt.title('X5 en Ec Diferencias')
plt.grid(True)
#plt.legend()
plt.show()

## Potencia y energía.
def potencia(xx, periodo):
    pot = 0
    for k in range(periodo):
        pot += (xx[k])**2
    
    pot = pot/periodo
    
    return pot

def energia (xx):
    E=0
    long = len(xx)
    for k in range(long):
        E = E + (xx[k])**2
    return E


# Y1 tiene periodo 0.0005s (200 muestras)
potY1 = potencia(Y1, 200)
potY2 = potencia(Y2, 200)
potY3 = potencia(Y3, 400)
potY4 = potencia(Y4, 200)
enerY5 = energia(Y5)
enerY6 = energia(Y6)

print(f'La potencia de la señal de salida para la sinusoidal de 2 kHz es: {potY1} W')
print(f'La potencia de la señal de salida para la cosenoidal de 2 kHz y Amplitud 2 es: {potY2} W')
print(f'La potencia de la señal de salida para la sinusoidal de 2 kHz modulada con otra de 1 kHz es: {potY3} W')
print(f'La potencia de la señal de salida para la sinusoidal de 2 kHz clippeada es: {potY4} W')
print(f'La energía de la señal de salida para la señal cuadrada de 2 kHz es: {enerY5} J')
print(f'La energía de la señal de salida para el pulso de 10 ms es: {enerY6} J')

# %% Punto 2
# Rta al Implulso y Salida de una sinusoidal (uso x1)

#y[n]=x[n]+3⋅x[n−10]

a2i = [0]
b2i = [1,0,0,0,0,0,0,0,0,0,3]

h2i = ec_diferencias(imp,a2i,b2i) #Rta al Impulso
Y2i = ec_diferencias(x1,a2i,b2i) #Sinusoidal

#y[n]=x[n]+3⋅y[n−10]

a2ii = [0,0,0,0,0,0,0,0,0,3]
b2ii = [1]

h2ii = ec_diferencias(imp,a2ii,b2ii) #Rta al Impulso
Y2ii = ec_diferencias(x1,a2ii,b2ii) #Sinusoidal

#Graficos
plt.figure(12, figsize=(15,10))

plt.subplot(2,1,1)
plt.plot(tt, imp, 'o',label='Entrada')
plt.plot(tt, h2i, 'x',label='Salida', color='red')
plt.title('Respuesta al Impulso, ecuación 2')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(tt, x1, label='Entrada')
plt.plot(tt, Y2i, label='Salida', color='red')
plt.title('Sinusoidal de 2KHz, ecuación 2')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid(True)

plt.tight_layout()
plt.show()

# Grafico 2
plt.figure(13, figsize=(15,10))

plt.subplot(2,1,1)
plt.plot(tt, imp, 'o', label='Entrada')
plt.plot(tt, h2ii, 'x', label='Salida', color='red')
plt.title('Respuesta al Impulso, ecuación 3')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(tt, x1, label='Entrada')
plt.plot(tt, Y2ii, label='Salida', color='red')
plt.title('Sinusoidal de 2KHz, ecuación 3')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid(True)

plt.tight_layout()
plt.show()


# %% Prueba
tt, xprueba = mi_funcion_sen(ff = 2000, nn = N, amp = 7, dc = 7, ph = 3*np.pi/2, fs = fs)

plt.figure(14)
plt.plot(tt, xprueba, label='Señal de Entrada')
plt.title('Sinusoidal de 2KHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()

#pot_prueba = potencia(xprueba, 200)

#print(f'{pot_prueba}')

# %% Bonus

DeltaT = 0.01
Nb = 1000
t_sim = DeltaT * Nb
t_bonus = np.arange(start = 0, stop = t_sim, step = DeltaT)

Q = np.zeros(N)

#Uso la función de ecuación de diferencias.
#P[n] = 3/200 * Q[n-1] + 197/200 * P[n-1]

a_bonus = [197/200]
b_bonus = [0, 3/200]

## Si Q es cte en 80
Q[0:N-1] = 80

## Si Q es una función sinusoidal
t_2, Q2 = mi_funcion_sen(ff = 1.25, nn = N, amp = 25, dc = 80, fs = 1/DeltaT)

Pc = ec_diferencias(Q, a_bonus, b_bonus)
Ps = ec_diferencias(Q2, a_bonus, b_bonus)

plt.figure(15)
plt.plot(t_bonus, Pc, label='Presión')
plt.title('Presión en función del tiempo con Q constante')
plt.xlabel('Tiempo [s]')
plt.ylabel('Presión [mmHg]')
plt.grid(True)
plt.show()

plt.figure(16)
plt.plot(t_2, Ps, label='Presión')
plt.title('Presión en función del tiempo con Q sinusoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Presión [mmHg]')
plt.grid(True)
plt.show()
