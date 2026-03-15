import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import least_squares

#fonction à minimiser
def F(X,Y,xc,yc,r):
    return sum([(((x-xc)**2 + (y-yc)**2)**0.5 - r)**2 for i in range(len(X))])

#minimisation sans contrainte
def methode2(x, y, method='lm',max_nfev=1000):
    '''
    minimise F sans contrainte
    Datas (x,y)
    par défaut methode Levenberg-Marquardt avec régularisation Ridge
    '''
    #definition de la fonction des residus
    def residus(theta):
        xc, yc, r = theta
        return np.sqrt((x-xc)**2 + (y-yc)**2) - r
    
    # initialisation
    xc0 = np.mean(x)
    yc0 = np.mean(y)
    r0  = np.mean(np.sqrt((x-xc0)**2 + (y-yc0)**2))

    result = least_squares(residus, x0=(xc0, yc0, r0), method=method, max_nfev=max_nfev)  #max nombre d'itération1000
    #voir vitesse de convergence et critère d'arret
    
    return result.x
#LM est une version améliorée de la methode de Gausse-Newton, étudier la convergence de la suite des paramêtres

#minimisation avec contrainte
def methode3(x,y, method='lm',max_nfev=1000):
    '''
    minimise F avec contrainte
    Datas (x,y)
    par défaut methode Levenberg-Marquardt avec régularisation Ridge
    '''

    #definition avec contrainte
    def residus(theta):
        xc, yc = theta #2 paramêtres car r dépend de xc et yc
        r = np.mean(np.sqrt((x[-1]-xc)**2 + (y[-1]-yc)**2))
        return np.sqrt((x-xc)**2 + (y-yc)**2) - r
    
    # initialisation
    xc0 = np.mean(x)
    yc0 = np.mean(y)
    r=np.sqrt((x_bruite[-1]-xc0)**2 + (y_bruite[-1]-yc0)**2)

    result = least_squares(residus, x0=(xc0, yc0,r), method=method, max_nfev=max_nfev)  #max nombre d'itération1000

    return result.x

random.seed()

#création d'un jeu de données de test

r=3
n=100
theta = np.random.uniform(0, 2*np.pi, 100)
x = r * np.cos(theta)
y = r * np.sin(theta)
sigma=0.5
bruit = np.random.normal(loc=0, scale=np.sqrt(sigma), size=(n, 2))
x_bruite = x + bruit[:,0]
y_bruite = y + bruit[:,1]

#visualisation methode 2

xc, yc, r =methode2(x_bruite,y_bruite)
theta = np.linspace(0, 2*np.pi, 300)
x_cercle = xc + r*np.cos(theta)
y_cercle = yc + r*np.sin(theta)
plt.scatter(x_cercle, y_cercle, label="cercle approché")
plt.scatter(x_bruite, y_bruite, alpha=0.6, label="points bruités")

plt.axis("equal")
plt.title("ajustement cercle methode 2")
plt.legend()
plt.show()

#visulaisation methode 3
xc, yc =methode3(x_bruite,y_bruite)
r=np.sqrt((x_bruite[-1]-xc)**2 + (y_bruite[-1]-yc)**2)
theta = np.linspace(0, 2*np.pi, 300)
x_cercle = xc + r*np.cos(theta)
y_cercle = yc + r*np.sin(theta)
plt.scatter(x_cercle, y_cercle, label="cercle approché")
plt.scatter(x_bruite, y_bruite, alpha=0.6, label="points bruités")

plt.axis("equal")
plt.title("ajustement cercle methode 3")
plt.legend()
plt.show()