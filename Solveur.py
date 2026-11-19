import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Assignation des paramètres par défaut (Tableau 2 de l'énoncé)
class parametres():
    Lx = 1.0        # [-] Longueur du domaine en x
    Ly = 1.0        # [-] Longueur du domaine en y
    k0 = 1e-2       # [-] Conductivité du matériau de base (chi = 0)
    k1 = 1.0        # [-] Conductivité du matériau conducteur (chi = 1)
    Q = 1.0         # [-] Terme source volumique uniforme
    phi_max = 0.4   # [-] Fraction maximale de matériau conducteur

PRM_DEFAUT = parametres()

#Convertit une valeur de χ en propriétés physiques, avec les valeurs par défaut de l’énoncé.
def chi_vers_proprietes(chi, prm) : 

    k_0 = prm.k0
    k_1 = prm.k1

    k = -k_0*(chi-1) + k_1*chi

    return k

def assembler_systeme(chi, cl, prm, terme_source=None) :

    ny, nx = chi.shape
    h = prm.Lx / (nx - 1)

    A = lil_matrix((nx*ny, nx*ny))
    b = np.zeros(nx*ny)

    k = chi_vers_proprietes(chi, prm)

    for j in range(1, ny-1) : #parcourir les lignes mais on evite les extremites car calculer dans conditions limites
        for i in range(1, nx-1) : #parcourir les colonnes mais on evite les extremites car calculer dans conditions limites
            m = i + nx*j
            k_est = (2*k[j,i+1]*k[j,i]) / (k[j,i+1] + k[j,i])
            k_ouest = (2*k[j,i-1]*k[j,i]) / (k[j,i-1] + k[j,i])
            k_nord = (2*k[j+1,i]*k[j,i]) / (k[j+1,i] + k[j,i])
            k_sud = (2*k[j-1,i]*k[j,i]) / (k[j-1,i] + k[j,i])

            A[m, m] = (k_est + k_ouest + k_nord + k_sud) / h**2
            A[m, m+1] = -k_est / (h**2)
            A[m, m-1] = -k_ouest / (h**2)
            A[m, m+nx] = -k_nord / (h**2)
            A[m, m-nx] = -k_sud / (h**2)

            if terme_source is None :
                b[m] = prm.Q
            else :
                b[m] = terme_source[j, i]

    from conditions_limites import appliquer_cl :

    A, b = appliquer_cl(A, b, k, cl, prm)

    return A, b


def resoudre(A, b) : 
    




    return 

