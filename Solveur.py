import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from conditions_limites import creer_cl_thermique

# Paramètres physiques par défaut (Tableau 2 de l'énoncé)
PRM_DEFAUT = {
    "Lx": 1.0,       # Longueur du domaine en x
    "Ly": 1.0,       # Longueur du domaine en y
    "k0": 1e-2,      # Conductivité du matériau de base (chi = 0)
    "k1": 1.0,       # Conductivité du matériau conducteur (chi = 1)
    "Q": 1.0,        # Terme source volumique uniforme
    "phi_max": 0.4   # Fraction maximale de matériau conducteur
}

cl = creer_cl_thermique(nx =21, ny=21) # changer 21 ce sont des exemples

def chi_vers_proprietes(chi, prm) :
    



    return

def assembler_systeme(chi, cl, prm, terme_source=None) :
    


    return


def resoudre(A, b) : 
    




    return 

