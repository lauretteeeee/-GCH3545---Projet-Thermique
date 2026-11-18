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
    


    return


def mesurer_performance(sol, chi, prm) :
    


    return 


