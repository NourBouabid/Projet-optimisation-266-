import numpy as np
from scipy.optimize import least_squares

#fonction à minimiser
def F(X,Y,xc,yc,r):
    return sum([(((X[i]-xc)**2 + (Y[i]-yc)**2)**0.5 - r)**2 for i in range(len(X))])

def residus(x,y,xc, yc, r):
        return np.sqrt((x-xc)**2 + (y-yc)**2) - r

#minimisation sans contrainte
def methode2(x,y,method='lm',max_nfev=1000):
    '''
    minimise F sans contrainte
    Datas (x,y)
    par défaut methode Levenberg-Marquardt avec régularisation Ridge
    '''
    # initialisation
    xc0 = np.mean(x)
    yc0 = np.mean(y)
    r0  = np.mean(residus(x,y,xc0, yc0, r0))

    result = least_squares(residus, [xc0, yc0, r0], method, max_nfev)  #max nombre d'itération1000

    return result.x
#LM est une version améliorée de la methode de Gausse-Newton, étudier la convergence de la suite des paramêtres

#minimisation avec contrainte
def methode3(X,Y):
    return ...