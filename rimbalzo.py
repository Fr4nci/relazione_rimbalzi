import wave
import numpy as np
from matplotlib.pyplot import pyplot as plt
from scipy.optimize import curve_fit

stream = wave.open("./audio.wave")
signal = np.frombuffer(stream.readframes(stream.getnframes(), dtype=np.int16))
t = np.arange(len(signal)) / stream.getframerate()

plt.figure("Rimbalzi pallina")
plt.plot(t, signal)
plt.xlabel("Tempo [s]")
plt.savefig("Grafico_rimbalzo.pdf")
t = []

def expo(n, h0, gamma):
    return h0 * gamma ** n

plt.figure("Altezze dei rimbalzi")
plt.errorbar(n, h, sigma_h, fmt='o')
popt, pconv = curve_fit(expo, n, h, sigma=sigma_h)
h0_hat, gamma_hat = popt, pconv
sigma_h0, sigma_gamma = np.sqrt(pconv.diagonal())
print(f"h0: {h0} \pm {sigma_h0}")
print(f"gamma: {gamma} \pm {sigma_gamma}")
x = np.linspace(0.0, n_rimbalzi, 100) # n_rimbalzi da sostituire con il numero effettivo
plt.plot(x, expo(x, h0_hat, gamma_hat))
plt.yscale("log")
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('Numero rimbalzi')
plt.ylabel('Altezze rimbalzi [m]')

def chi_quadro(arr, sigma_arr, h0, gamma):
    chi_quadro = 0
    for x in range(len(arr)):
        chi_quadro = chi_quadro + np.power(arr[x] - expo(x + 1, h0, gamma), 2)/sigma_arr[x]
    return(chi_quadro) 

plt.figure("Grafico dei residui")

chi_quadro = chi_quadro(h, sigma_h, h0_hat, gamma_hat)
print(f"Il chi_quadro vale {chi_quadro}")
