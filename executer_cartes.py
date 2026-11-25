"""Teste les cartes CSV : lecture, admissibilité, puis résolution si admissible.
Usage :  python executer_cartes.py carte1.csv carte2.csv ...
Sans argument (bouton Run de VS Code) : teste les 3 cartes de test.
"""
import sys
import numpy as np
from gestion_cartes import lire_carte, evaluer_carte
from solveur import PRM_DEFAUT, assembler_systeme, resoudre
from conditions_limites import creer_cl_thermique, mesurer_performance

# Sans argument : on teste les 3 cartes de la présentation
if len(sys.argv) == 1:
    sys.argv += ["carte_test_valide.csv",
                 "carte_test_non_binaire.csv",
                 "carte_test_invalide.csv"]

for chemin in sys.argv[1:]:
    print(f"\n=== {chemin}")

    # 1) Lecture : une erreur de format arrête le traitement de cette carte
    try:
        chi = lire_carte(chemin)
    except ValueError as e:
        print("ERREUR DE FORMAT :", e)
        continue
    ny, nx = chi.shape
    print(f"Lecture OK : {nx} x {ny} noeuds")

    # 2) Admissibilité : si la carte ne respecte pas les critères, on ne résout pas
    if not evaluer_carte(chi):
        print("Carte INADMISSIBLE")
        continue
    print("Carte ADMISSIBLE")

    # 3) Résolution et performance
    cl = creer_cl_thermique(nx, ny)
    A, b = assembler_systeme(chi, cl, PRM_DEFAUT)
    T_moy, T_max, phi = mesurer_performance(resoudre(A, b), chi, PRM_DEFAUT)
    print(f"T moyenne = {T_moy:.3f}   T max = {T_max:.3f}   phi = {phi:.4f}")