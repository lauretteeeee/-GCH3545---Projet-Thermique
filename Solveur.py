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
    """Convertit une valeur de χ en propriétés physiques, avec les valeurs par défaut de l’énoncé.
    • Entrée :
        – chi : scalaires compris entre 0 et 1 inclusivement.
        – prm : paramètres physiques du problème.
    • Sortie :
        – propriété(s) physique(s) associée(s) à chi ; leur nombre dépend du problème (voir l’exigence A1 de votre énoncé).
    """

    k_0 = prm.k0
    k_1 = prm.k1

    k = -k_0*(chi-1) + k_1*chi

    return k

def assembler_systeme(chi, cl, prm, terme_source=None) :
    """Construit le système linéaire discret A u= b.
    • Entrées :
        – chi : tableau d’entiers de forme (ny, nx) ;
        – cl : conditions aux limites (structure précisée à la section 5.1) ;
        – prm : paramètres physiques du problème ;
        – terme_source : None par défaut (aucun terme source). Pour la vérification par la 
                         méthode des solutions manufacturées (section 4.1), on y passe une fonction Q(x, y) qui
                         renvoie la valeur scalaire du terme source au point (x, y).
    • Sorties :
        – A : matrice de taille N × N avec N= nxny ;
        – b : vecteur de taille N.
    """

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

    from conditions_limites import appliquer_cl
    A, b = appliquer_cl(A, b, k, cl, prm)

    return A, b


def resoudre(A, b) : 
    """Résout le système linéaire A u= b
    
    • Entrées :
        – A, b : système retourné par assembler_systeme.
    • Sortie :
        – la solution : vecteur de taille N , dont la composante m = i + nxj est la valeur au nœud
                        (i, j). Indice : sol.reshape(ny, nx) redonne le champ indexé [j, i]
    """
    
    A_csr = A.tocsr(copy = True)
    sol = spsolve(A_csr, b)

    return sol

