"""Teste les cartes CSV : lecture, admissibilité, puis résolution si admissible.
Usage :  python executer_cartes.py carte1.csv carte2.csv ...
"""
import sys
import numpy as np
import gestion_cartes as gc
from solveur import PRM_DEFAUT, assembler_systeme, resoudre
from conditions_limites import creer_cl_thermique, mesurer_performance

# Accepte noms au singulier (lire_carte) ou au pluriel (lire_cartes)
lire = getattr(gc, "lire_carte", None) or gc.lire_cartes
evaluer = getattr(gc, "evaluer_carte", None) or gc.evaluer_cartes

for chemin in sys.argv[1:]:
    print(f"\n=== {chemin}")
    try:
        chi = lire(chemin)
    except ValueError as e:
        print("ERREUR DE FORMAT :", e)
        continue
    ny, nx = chi.shape
    print(f"Lecture OK : {nx} x {ny} noeuds")
    if not evaluer(chi):
        print("Carte INADMISSIBLE")
        continue
    print("Carte ADMISSIBLE")
    cl = creer_cl_thermique(nx, ny)
    A, b = assembler_systeme(chi, cl, PRM_DEFAUT)
    T_moy, T_max, phi = mesurer_performance(resoudre(A, b), chi, PRM_DEFAUT)
    print(f"T moyenne = {T_moy:.3f}   T max = {T_max:.3f}   phi = {phi:.4f}")