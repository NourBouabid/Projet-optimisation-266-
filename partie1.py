import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares,minimize

# MÉTHODE 1 — Ajustement linéaire

def methode1_lineaire(x,y):

    # Construction du système Aθ = b
    A = np.column_stack((x, y, np.ones(len(x))))
    b = -(x**2 + y**2)#explique mieux le chois de b et a     # Résolution par moindres carrés
    a, b_coef, c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calcul du centre
    x0 = -a / 2 #? prq ce choit 
    y0 = -b_coef / 2

    # Calcul du rayon
    r = np.sqrt(x0**2 + y0**2 - c)

    return x0, y0, r

# TEST SIMPLE

theta = np.linspace(0, 2*np.pi, 50)
x = 2 + 5*np.cos(theta)
y = -1 + 5*np.sin(theta)


# Appel de la fonction
x0, y0, r = methode1_lineaire(x,y)

# Affichage propre
print(f"Centre : ({x0:.4f}, {y0:.4f})")
print(f"Rayon  : {r:.4f}")


r=3  
n=100
theta = np.random.uniform(0, 2*np.pi, 100)
x = r * np.cos(theta)
y = r * np.sin(theta)

sigma=0.5
bruit = np.random.normal(loc=0, scale=np.sqrt(sigma), size=(n, 2))
x_bruite = x + bruit[:,0]
y_bruite = y + bruit[:,1]
xc, yc, r = methode1_lineaire(x_bruite, y_bruite)




theta = np.linspace(0, 2*np.pi, 300)
x_cercle = xc + r*np.cos(theta)
y_cercle = yc + r*np.sin(theta)
plt.scatter(x_cercle, y_cercle, label="cercle approché")
plt.scatter(x_bruite, y_bruite, alpha=0.6, label="points bruités")

plt.axis("equal")
plt.title("ajustement cercle methode 3")
plt.legend()
plt.show()