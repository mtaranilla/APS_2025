# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:42:27 2025

@author: Mora
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

#%% Polos y Ceros de Redes (punto 4)
## a
sos_a = [[1,0,9,1,np.sqrt(2),1]]  #(b2,b1,b0,a2,a1,a0)

z_a, p_a, k_a = sig.sos2zpk(sos_a)

# Parámetros de la circunferencia
w0_z_a = 3
w0_p_a = 1
x0, y0 = 0, 0  # centro 
theta = np.linspace(0, 2*np.pi, 500)

plt.figure(figsize=(4,4))
plt.plot(np.real(p_a), np.imag(p_a), 'x', markersize=10, label='Polos')
if len(z_a) > 0:
    plt.plot(np.real(z_a), np.imag(z_a), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.plot(w0_z_a * np.cos(theta), w0_z_a * np.sin(theta), label='Circunferencia de radio w0 ceros', linestyle='--')
plt.plot(w0_p_a * np.cos(theta), w0_p_a * np.sin(theta), label='Circunferencia de radio w0 polos', linestyle='--')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros, T1(s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

# b
sos_b = [[1,0,1/9,1,1/5,1]]

z_b, p_b, k_b = sig.sos2zpk(sos_b)

# Parámetros de la circunferencia
w0_z_b = 1/3
w0_p_b = 1
x0, y0 = 0, 0  # centro 

plt.figure(figsize=(4,4))
plt.plot(np.real(p_b), np.imag(p_b), 'x', markersize=10, label='Polos')
if len(z_b) > 0:
    plt.plot(np.real(z_b), np.imag(z_b), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.plot(w0_z_b * np.cos(theta), w0_z_b * np.sin(theta), label='Circunferencia de radio w0 ceros', linestyle='--')
plt.plot(w0_p_b * np.cos(theta), w0_p_b * np.sin(theta), label='Circunferencia de radio w0 polos', linestyle='--')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros, T2(s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

# c
sos_c = [[1,1/5,1,1,np.sqrt(2),1]]

z_c, p_c, k_c = sig.sos2zpk(sos_c)

# Parámetros de la circunferencia
w0_z_c = 1
w0_p_c = 1
x0, y0 = 0, 0  # centro 

plt.figure(figsize=(4,4))
plt.plot(np.real(p_c), np.imag(p_c), 'x', markersize=10, label='Polos')
if len(z_c) > 0:
    plt.plot(np.real(z_c), np.imag(z_c), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.plot(w0_z_c * np.cos(theta), w0_z_c * np.sin(theta), label='Circunferencia de radio w0 ceros', linestyle='--')
plt.plot(w0_p_c * np.cos(theta), w0_p_c * np.sin(theta), label='Circunferencia de radio w0 polos', linestyle='--')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros, T3(s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)