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
t_1 = [0.25374149, 1.3378907, 2.1928567, 2.879543, 3.43581, 3.888680, 4.25915, 4.56335, 4.81170, 5.01293]               # per rendere più semplice la trattazione abbiamo deciso di riportare i dati in due array
t_2 = [0.254240358, 1.34247168, 2.197514,  2.888841,  3.44512, 3.898321, 4.259859, 4.5642629, 4.820339, 5.02143]        # differenti per poi iterare (siccome hanno la stessa dimensione con zip). Sicuramente non sarà
n = []                                                                                                                  # il metodo più efficiente ma il più funzionale per non impazzire sulle dimensioni degli array
frequency = 1/(44.1 * 10**3)
t_star = []
h = []
sigma_t = []
sigma_h = []
for el in zip(t_1, t_2):
    t_star.append((el[1]+el[0])/2)
    sigma_t.append((el[1]-el[0])/2)
print(t_star)
print(sigma_t)
for x in range(len(t_star)-1):    
    h.append((1./8.) * g * (t_star[x] - t_star[x+1])**2)
    sigma_h.append((1./4.) * g * np.sqrt(sigma_t[x]**2 + sigma_t[x+1]**2) * (t_star[x] - t_star[x+1]))
    n.append(x+1)
print(h)
print(sigma_h)
def expo(n, h0, gamma):
    return h0 * gamma ** n
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
plt.xlabel(r'Numero dei rimbalzi')
plt.ylabel(r'Altezze dei rimbalzi ($m$)')
plt.title(r"Grafico $n-h(t)$")
plt.savefig("Grafico_n-h(t).pdf")
plt.figure("Grafico dei residui")
plt.grid()
plt.errorbar(n, h - expo(n, h0_hat, gamma_hat), yerr=sigma_h, fmt='o')
plt.axhline(0, color="black", linestyle="--")
plt.savefig("Grafico_residui.pdf")
plt.show()
chi_quadro = 0
for x in range(len(h)):
    chi_quadro = chi_quadro + ((h[x] - expo(x+1, h0_hat, gamma_hat))/sigma_h[x])**2         # funzione usata per calcolare il chi_quadro
print(f"Il chi_quadro normalizzato vale {chi_quadro/len(t_star)}")
