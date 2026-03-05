import numpy as np
from scipy.optimize import least_squares,minimize

# MÉTHODE 1 — Ajustement linéaire

def methode1_lineaire(points):
    # Séparation des coordonnées
    x = points[:, 0]
    y = points[:, 1]

    # Construction du système Aθ = b
    A = np.column_stack((x, y, np.ones(len(x))))
    b = -(x**2 + y**2)

    # Résolution par moindres carrés
    a, b_coef, c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calcul du centre
    x0 = -a / 2
    y0 = -b_coef / 2

    # Calcul du rayon
    r = np.sqrt(x0**2 + y0**2 - c)

    return x0, y0, r

# TEST SIMPLE

theta = np.linspace(0, 2*np.pi, 50)
x = 2 + 5*np.cos(theta)
y = -1 + 5*np.sin(theta)

points = np.column_stack((x, y))

# Appel de la fonction
x0, y0, r = methode1_lineaire(points)

# Affichage propre
print(f"Centre : ({x0:.4f}, {y0:.4f})")
print(f"Rayon  : {r:.4f}")

# Méthode 2 — Ajustement géométrique (non-linéaire)
def methode2_geometrique(points):
    x = points[:, 0]
    y = points[:, 1]

    # Résidus géométriques : distance au centre - r
    def residus(p):
        xc, yc, r = p
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    # Initialisation avec la méthode 1
    x0, y0, r0 = methode1_lineaire(points)
    p0 = np.array([x0, y0, r0])

    # Levenberg–Marquardt (least_squares)
    res = least_squares(residus, p0, method="lm")

    xc, yc, r = res.x
    return xc, yc, r
# Petit test + comparaison
if __name__ == "__main__":
    # Exemple 
    theta = np.linspace(0, 2*np.pi, 50)
    x = 2 + 5*np.cos(theta) + 0.1*np.random.randn(50)
    y = -1 + 5*np.sin(theta) + 0.1*np.random.randn(50)
    points = np.column_stack((x, y))

    x1, y1, r1 = methode1_lineaire(points)
    x2, y2, r2 = methode2_geometrique(points)

    print("Methode 1 (algebrique)")
    print(f"Centre : ({x1:.4f}, {y1:.4f})  Rayon : {r1:.4f}")

    print("\nMethode 2 (geometrique)")
    print(f"Centre : ({x2:.4f}, {y2:.4f})  Rayon : {r2:.4f}")

def methode3_contrainte(points):
    
   # Ajustement géométrique avec contrainte :le dernier point (x_n, y_n) appartient exactement au cercle.
    
    x = points[:, 0]
    y = points[:, 1]
    xn, yn = points[-1]  # dernier point

    # Fonction objectif : somme des (distance au centre - r)^2
    def objectif(p):
        xc, yc, r = p
        d = np.sqrt((x - xc)**2 + (y - yc)**2) - r
        return np.sum(d**2)

    # Contrainte : (xn-xc)^2 + (yn-yc)^2 = r^2
    def contrainte(p):
        xc, yc, r = p
        return (xn - xc)**2 + (yn - yc)**2 - r**2  # doit être 0

    # Initialisation simple (centre = moyenne, rayon = distance moyenne)
    xc0, yc0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x - xc0)**2 + (y - yc0)**2))
    p0 = np.array([xc0, yc0, r0])

    # Bornes : r >= 0
    bounds = [(None, None), (None, None), (0, None)]
    cons = {"type": "eq", "fun": contrainte}

    res = minimize(objectif, p0, method="SLSQP", bounds=bounds, constraints=[cons])

    if not res.success:
        raise RuntimeError("SLSQP n'a pas convergé : " + res.message)

    return tuple(res.x)
# ------------------ petit test ------------------
if __name__ == "__main__":
    theta = np.linspace(0, 2*np.pi, 50)
    x = 2 + 5*np.cos(theta) + 0.1*np.random.randn(50)
    y = -1 + 5*np.sin(theta) + 0.1*np.random.randn(50)
    points = np.column_stack((x, y))

    xc, yc, r = methode3_contrainte(points)
    print(f"Centre : ({xc:.4f}, {yc:.4f})")
    print(f"Rayon  : {r:.4f}")

    # Vérification de la contrainte (doit être proche de 0)
    xn, yn = points[-1]
    check = (xn - xc)**2 + (yn - yc)**2 - r**2
    print(f"Contrainte proche de 0 : {check:.3e}")