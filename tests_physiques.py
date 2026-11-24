from solveur import PRM_DEFAUT, chi_vers_proprietes, assembler_systeme, resoudre
from conditions_limites import creer_cl_thermique, mesurer_performance
import numpy as np


def verifier_bilan_energie(sol, chi, prm):
    """Vérifie que toute la chaleur produite dans le domaine sort par le puits.

    Entrées:
        - sol : solution retournée par la fonction "resoudre" (vecteur de taille nx*ny)
        - chi : carte des matériaux, tableau de forme (ny, nx)
        - prm : paramètres (Lx, Ly, Q, k0, k1)

    Sorties:
        - q_produit : chaleur produite dans le domaine (intégrale de Q)
        - q_sortant : chaleur qui sort par le puits
        - erreur : écart relatif |q_sortant - q_produit| / q_produit
    """

    ny, nx = chi.shape
    h = prm.Lx/(nx-1)

    T = np.reshape(sol, (ny, nx))               # On remet la solution sous forme de tableau [j, i] (comme chi)
    cl = creer_cl_thermique(nx, ny)
    k = chi_vers_proprietes(chi, prm)

    # 1) Chaleur produite : Q est uniforme (toujours 1), donc Q * aire du domaine
    q_produit = prm.Q*prm.Lx*prm.Ly

    # 2) Chaleur évacuée : on trouve les noeuds du puits (type 'D' sur le bord gauche),
    #       puis on additionne le flux qui sort par chacun d'eux
    j_puits = []
    for j in range(ny):
        if cl["gauche"]["type"][j] == "D":
            j_puits.append(j)
    j_debut = j_puits[0]                   # premier noeud du puits
    j_fin = j_puits[-1]                    # dernier noeud du puits

    q_sortant = 0.0
    for j in range(j_debut, j_fin+1):
        dTdx = (-3*T[j, 0] + 4*T[j, 1] - T[j, 2])/(2*h)     # dT/dx au bord x = 0 : différence avant d'ordre 2 (3 points T0, T1, T2)
        flux = k[j, 0]*dTdx                                 # flux sortant par unité de longueur : q_n = k * dT/dx (k du noeud de bord)
        if j == j_debut or j == j_fin:                      # intégrale par la méthode des trapèzes : les 2 extrémités comptent pour 1/2
            q_sortant = q_sortant + 0.5*h*flux
        else:
            q_sortant = q_sortant + h*flux

    # 3) Écart relatif entre ce qui est produit et ce qui sort
    erreur = abs(q_sortant - q_produit)/q_produit

    return q_produit, q_sortant, erreur


def verifier_monotonie(prm, n=41):
    """Vérifie qu'ajouter du matériau conducteur ne fait jamais augmenter la
    température moyenne T_moy.

    Entrées:
        - prm : paramètres (Lx, Ly, Q, k0, k1)
        - n : nombre de noeuds en x et en y (grille n x n)

    Sortie:
        - True si T_moy ne remonte jamais, False sinon
    """

    cl = creer_cl_thermique(n, n)
    chi = np.zeros((n, n))                 # on part sans conducteur (chi = 0 partout)

    
    positions = [(5, 10), (20, 25), (30, 8), (12, 30), (25, 12), (8, 20)]       # Positions (j0, i0) des coins inférieurs gauches de blocs (taille x taille) de conducteur, ajoutés un à un
    taille = 8                             # chaque bloc fait 8 x 8 noeuds

    monotone = True
    T_moy_precedent = None

    for etape in range(len(positions)+1):
        # Résolution pour la carte actuelle
        A, b = assembler_systeme(chi, cl, prm)
        sol = resoudre(A, b)
        T_moy, T_max, phi = mesurer_performance(sol, chi, prm)
        print(f"  étape {etape} : {etape} bloc(s) de conducteur, T_moy = {T_moy:.4f}")

        # T_moy ne doit pas avoir augmenté (petite tolérance pour les arrondis)
        if T_moy_precedent is not None and T_moy > T_moy_precedent + 1e-12:
            monotone = False
        T_moy_precedent = T_moy

        # On ajoute le bloc suivant (chi passe à 1 sur un carré de noeuds)
        if etape < len(positions):
            j0, i0 = positions[etape]
            chi[j0:j0+taille, i0:i0+taille] = 1.0

    return monotone


def verifier_deux_couches(prm, n, orientation="x"):
    """Compare le solveur à la solution exacte d'un problème à deux couches
    (une couche de matériau de base, une couche de conducteur).

    Montage (le problème devient 1D) :
        - orientation "x" : couches côte à côte selon x (colonnes). T = 0 sur
          tout le bord gauche, flux nul sur les 3 autres bords.
        - orientation "y" : couches empilées selon y (lignes). T = 0 sur tout
          le bord du bas, flux nul sur les 3 autres bords.
    Solution exacte : la chaleur produite entre l'abscisse s et l'extrémité
    isolée (s = L) doit traverser la section, donc le flux vaut Q*(L - s) et
        T(s) = intégrale de 0 à s de Q*(L - u)/k(u) du

    Entrées:
        - prm : paramètres (Lx, Ly, Q, k0, k1)
        - n : nombre de noeuds en x et en y (grille n x n)
        - orientation : "x" ou "y"

    Sortie:
        - erreur relative maximale entre le solveur et la solution exacte
    """

    h = prm.Lx/(n-1)
    L = prm.Lx

    # Carte : chi = 0 (base) avant l'indice ia, chi = 1 (conducteur) à partir de ia
    ia = n//2
    chi = np.zeros((n, n))
    if orientation == "x":
        chi[:, ia:] = 1.0                  # colonnes de droite = conducteur
    else:
        chi[ia:, :] = 1.0                  # lignes du haut = conducteur

    # Conditions aux limites : on part du cas thermique (flux nul partout et puits à gauche), puis on remplace ce qui est nécessaire
    cl = creer_cl_thermique(n, n)
    cl["gauche"]["type"][:] = "N"          # on enlève le petit puits
    cl["gauche"]["valeur"][:] = 0.0
    if orientation == "x":
        cl["gauche"]["type"][:] = "D"      # T = 0 sur tout le bord gauche
    else:
        cl["bas"]["type"][:] = "D"         # T = 0 sur tout le bord du bas
        cl["bas"]["valeur"][:] = 0.0

    # Résolution avec le solveur
    A, b = assembler_systeme(chi, cl, prm)
    sol = resoudre(A, b)
    T = np.reshape(sol, (n, n))

    # Profil de température dans la direction des couches (ligne ou colonne du milieu)
    if orientation == "x":
        profil = T[n//2, :]
    else:
        profil = T[:, n//2]

    # Position de l'interface entre les deux matériaux : à mi-chemin entre le dernier noeud de base (ia - 1) et le premier noeud de conducteur (ia)
    a = (ia - 0.5)*h

    # Solution exacte aux noeuds : F(s) = Q*(L*s - s^2/2)
    erreur_max = 0.0
    T_exact_max = 0.0
    for i in range(n):
        s = i*h
        Fs = prm.Q*(L*s - s**2/2)
        Fa = prm.Q*(L*a - a**2/2)
        if s <= a:
            T_exact = Fs/prm.k0
        else:
            T_exact = Fa/prm.k0 + (Fs - Fa)/prm.k1
        if abs(profil[i] - T_exact) > erreur_max:
            erreur_max = abs(profil[i] - T_exact)
        if T_exact > T_exact_max:
            T_exact_max = T_exact

    return erreur_max/T_exact_max


if __name__ == "__main__":
    # --- Bilan d'énergie sur des grilles de plus en plus fines (chi = 0 partout)
    print("=== Bilan d'énergie (chi = 0 partout) ===")
    erreur_precedente = None
    for n in [41, 81, 161, 321]:
        chi = np.zeros((n, n))
        cl = creer_cl_thermique(n, n)
        A, b = assembler_systeme(chi, cl, PRM_DEFAUT)
        sol = resoudre(A, b)
        q_produit, q_sortant, erreur = verifier_bilan_energie(sol, chi, PRM_DEFAUT)
        if erreur_precedente is None:
            print(f"n = {n:3d} : q produit = {q_produit:.3f} | q sortant = {q_sortant:.4f} | erreur = {erreur:.3e}")
        else:
            ordre = np.log(erreur_precedente/erreur)/np.log(2)
            print(f"n = {n:3d} : q produit = {q_produit:.3f} | q sortant = {q_sortant:.4f} | erreur = {erreur:.3e} | ordre = {ordre:.2f}")
        erreur_precedente = erreur

    # --- Monotonie
    print("=== Monotonie (ajout de blocs de conducteur) ===")
    if verifier_monotonie(PRM_DEFAUT):
        print("Test réussi : T_moy ne remonte jamais.")
    else:
        print("ÉCHEC : T_moy a augmenté en ajoutant du conducteur.")

    # --- Deux couches (vérifie la moyenne harmonique), dans les deux directions
    print("=== Deux couches (solution exacte 1D) ===")
    for orientation in ["x", "y"]:
        erreur_precedente = None
        for n in [41, 81, 161]:
            erreur = verifier_deux_couches(PRM_DEFAUT, n, orientation)
            if erreur_precedente is None:
                print(f"couches selon {orientation}, n = {n:3d} : erreur relative = {erreur:.3e}")
            else:
                ordre = np.log(erreur_precedente/erreur)/np.log(2)
                print(f"couches selon {orientation}, n = {n:3d} : erreur relative = {erreur:.3e} | ordre = {ordre:.2f}")
            erreur_precedente = erreur