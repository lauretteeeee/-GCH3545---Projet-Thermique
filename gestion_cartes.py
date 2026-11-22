import numpy as np

def lire_cartes(chemin) :

    # ouvrir le fichier
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            lignes = f.read().splitlines()
    except FileNotFoundError:
        raise ValueError(f"Fichier introuvable : {chemin}")
     #  enlever les lignes vides
    lignes = [l.strip() for l in lignes if l.strip() != ""]
    if len(lignes) == 0:
        raise ValueError(f"Fichier vide : {chemin}")
    # lire chaque ligne et verifier les valeurs
    carte = []
    for num, ligne in enumerate(lignes, start=1):
        valeurs = ligne.split(",")
        rangee = []
        for v in valeurs:
            v = v.strip()
            if v not in ("0", "1"):
                raise ValueError(f"Ligne {num} : valeur '{v}' invalide ...")
            rangee.append(int(v))
        carte.append(rangee)
     # toutes les lignes meme longueur
    nx = len(carte[0])
    for num, rangee in enumerate(carte, start=1):
        if len(rangee) != nx:
            raise ValueError(f"Ligne {num} : {len(rangee)} valeurs au lieu de {nx}")
        
     # convert tableau 
        chi = np.array(carte, dtype=int)
        return chi

    


def ecrire_cartes(chemin, chi) :

    np.savetxt(chemin, chi, fmt="%d", delimiter=",")
    


def evaluer_cartes(chi) :
    ny, nx = chi.shape

    # critere 1 : nx - 1 et ny - 1 multiples de 20
    if (nx - 1) % 20 != 0 or (ny - 1) % 20 != 0:
        return False

     # critere 2 : meme matériau sur les 3 premiers noeuds de chaque bord
    for c in [1, 2]:
        if not np.array_equal(chi[:, 0], chi[:, c]):       # bord gauche
            return False
        if not np.array_equal(chi[:, -1], chi[:, -1 - c]): # bord droit
            return False
        if not np.array_equal(chi[0, :], chi[c, :]):       # bord bas
            return False
        if not np.array_equal(chi[-1, :], chi[-1 - c, :]): # bord haut
            return False

    # critere 3 : au plus 40 % de conducteur (methode des trapezes)
    somme = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            somme = somme + (chi[j, i] + chi[j, i+1] + chi[j+1, i] + chi[j+1, i+1]) * 0.25
    phi = somme / ((nx - 1) * (ny - 1))
   
    if phi > 0.4 :
        return False

    return True



