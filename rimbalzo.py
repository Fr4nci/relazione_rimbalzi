import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

stream = wave.open("./Audio.wav")
signal = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
t = np.arange(len(signal)) / stream.getframerate()

plt.figure("Rimbalzi pallina")
plt.plot(t, signal)
plt.xlabel("Tempo [s]")
plt.savefig("Grafico_rimbalzo.pdf")
plt.show()
g = 9.81
t = [0.25374149, 0.258494, 1.3378907, 1.3424733, 2.1928567, 2.197514, 2.879543, 2.888841, 3.43581, 3.44512] # array dei tempi non filtrato
t_star = []
sigma_t_star = []
counter = 0
while counter < len(t):
    t_star.append((t[counter] + t[counter+1])/2)
    sigma_t_star.append((t[counter+1] - t[counter])/2)
    counter = counter + 2
sigma_t_star.append((t[len(t_star)] - t[len(t_star)-1])/2)
n = []
for el in range(len(t_star)):
    n.append(el+1)
def expo(n, h0, gamma):
    return h0 * gamma ** n
h = [2.50]
sigma_h = []
print(t_star)
print(len(t_star))
print(len(sigma_t_star))
for el in range(len(t_star)-1):
    value = (1./8.) * g * (t_star[el + 1] - t_star[el])**2
    h.append(value)
    sigma_h.append(2 * np.sqrt((1. / (t_star[el + 1] - t_star[el])) * (sigma_t_star[el+1]** 2 + sigma_t_star[el]**2)))
sigma_h.append(2 * np.sqrt((1. / (t_star[len(t_star)-1] - t_star[len(t_star)-2])) * (sigma_t_star[len(t_star)-1]** 2 + sigma_t_star[len(t_star)-2]**2)))
plt.figure("Altezze dei rimbalzi")
plt.errorbar(n, h, sigma_h, fmt='o')
popt, pconv = curve_fit(expo, n, h, sigma=sigma_h)
h0_hat, gamma_hat = popt
sigma_h0, sigma_gamma = np.sqrt(pconv.diagonal())
print(f"h0: {h0_hat} \pm {sigma_h0}")
print(f"gamma: {gamma_hat} \pm {sigma_gamma}")
x = np.linspace(0.0, n[len(n)-1], 100) # n_rimbalzi da sostituire con il numero effettivo
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
plt.errorbar(n, h - expo(n, h0_hat, gamma_hat), yerr=sigma_h, fmt='o')
plt.show()
chi_quadro = chi_quadro(h, sigma_h, h0_hat, gamma_hat)
print(f"Il chi_quadro vale {chi_quadro}")
