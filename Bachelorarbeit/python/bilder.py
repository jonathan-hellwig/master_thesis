import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(-np.pi, np.pi, 1000)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.plot(x, x * np.sin(x))
plt.title('Drei Funktionen')
plt.xlabel('x-Achse')
plt.ylabel('y-Achse')
plt.legend(['Sinus', 'Kosinus', 'x·sin(x)'])
plt.savefig('Bild.pdf')
plt.savefig('Bild.eps')
plt.savefig('Bild.jpg')
plt.savefig('Bild.png')
plt.show()
