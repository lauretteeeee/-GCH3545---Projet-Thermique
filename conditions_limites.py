from solveur import PRM_DEFAUT
import numpy as np

def creer_cl_thermique(nx, ny, Ly=1.0):
    """Construit le dictionnaire cl par défaut pour le problème thermique."""
    cl = {
        "gauche": {"type": np.full(ny, "N"), "valeur": np.zeros(ny)},
        "droite": {"type": np.full(ny, "N"), "valeur": np.zeros(ny)},
        "bas":    {"type": np.full(nx, "N"), "valeur": np.zeros(nx)},
        "haut":   {"type": np.full(nx, "N"), "valeur": np.zeros(nx)},
    }
    # Puits de chaleur sur le bord gauche (x = 0, 0.45 <= y <= 0.55)
    y = np.linspace(0.0, Ly, ny)
    masque_puits = (y >= 0.45 - 1e-12) & (y <= 0.55 + 1e-12)
    cl["gauche"]["type"][masque_puits] = "D"
    cl["gauche"]["valeur"][masque_puits] = 0.0
    return cl


def appliquer_cl(A_lil, b, k, cl, prm) :
    def indice(i,j,nx): #traduit les coorodonnees (i,j) en indice m de la matrice 
      m = i +nx*j
      return m
    #k est la matrice contenant les conductivites a chaque noeud

    #dimensions du domaine
    prm.nx,prm.ny = k.shape

    #pas spatial hx et hy
    h = prm.Lx/(prm.nx-1)

    #stocker les conditions de chaque noeud 
    conditions = {}

    #parcourir les 4 bords du domaine
    for bord in ["gauche", "droite", "haut", "bas"] :
        if bord == "gauche" :
            i = 0
            for j in range(prm.ny) :
                m = indice(i,j,prm.nx)
                A_lil[m,:] = 0.0
                A_lil[m,m] = 1.0
                b[m] = cl.gauche(j*h)
                conditions[m] = cl.gauche
        elif bord == "droite" :
            i = prm.nx-1
            for j in range(prm.ny) :
                m = indice(i,j,prm.nx)
                A_lil[m,:] = 0.0
                A_lil[m,m] = 1.0
                b[m] = cl.droite(j*h)
                conditions[m] = cl.droite
        elif bord == "haut" :
            j = prm.ny-1
            for i in range(prm.nx) :
                m = indice(i,j,prm.nx)
                A_lil[m,:] = 0.0
                A_lil[m,m] = 1.0
                b[m] = cl.haut(i*h)
                conditions[m] = cl.haut
        elif bord == "bas" :
            j = 0
            for i in range(prm.nx) :
                m = indice(i,j,prm.nx)
                A_lil[m,:] = 0.0
                A_lil[m,m] = 1.0
                b[m] = cl.bas(i*h)
                conditions[m] = cl.bas

    return conditions


def mesurer_performance(sol, chi, prm) :
    


    return 


