import numpy as np
from scipy.optimize import least_squares

#fonction à minimiser
def F(X,Y,xc,yc,r):
    return sum([(((X[i]-xc)**2 + (Y[i]-yc)**2)**0.5 - r)**2 for i in range(len(X))])

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

    result = least_squares(residus, x0=(xc0, yc0), method=method, max_nfev=max_nfev)  #max nombre d'itération1000

    return result.x