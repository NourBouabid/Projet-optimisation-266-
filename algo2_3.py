from scipy.optimize import least_squares

#fonction à minimiser
def F(X,Y,xc,yc,r):
    return sum([(((X[i]-xc)**2 (Y[i]-yc)**2)**0.5 - r)**2 for i in range(len(X))])

#minimisation sans contrainte
def methode2(X,Y):
    return ...

#minimisation avec contrainte
def methode3(X,Y):
    return ...