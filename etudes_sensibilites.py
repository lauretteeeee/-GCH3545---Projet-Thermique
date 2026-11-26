"""
Etudes de sensibilite du projet thermique GCH3545.

Usage :
    python etudes_sensibilite.py conception_finale.csv
    python etudes_sensibilite.py carte_depart_vide.csv carte_depart_reference.csv

Les cartes adaptees sont revalidees. Si une carte adaptee ne respecte plus
les contraintes, le programme s'arrete au lieu de la corriger silencieusement.
"""

from copy import copy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from gestion_cartes import lire_carte, evaluer_carte
from solveur import PRM_DEFAUT, assembler_systeme, resoudre
from conditions_limites import creer_cl_thermique, mesurer_performance


MAILLAGES = [(21, 21), (41, 41), (81, 81)]
CONTRASTES = [10, 100, 1000]
DOSSIER_RESULTATS = Path("resultats_sensibilite")


def resoudre_et_mesurer(chi, prm):
    """Resout le probleme et retourne T (2D), T_moy, T_max et phi."""
    ny, nx = chi.shape
    cl = creer_cl_thermique(nx, ny, prm.Ly)
    A, b = assembler_systeme(chi, cl, prm)
    sol = resoudre(A, b)

    T_moy, T_max, phi = mesurer_performance(sol, chi, prm)
    T = np.asarray(sol).reshape((ny, nx))
    return T, T_moy, T_max, phi


def adapter_carte(chi, nx_cible, ny_cible):
    """
    Adapte une carte binaire a une grille cible par plus proche voisin.

    Les extremites du domaine sont conservees. Apres reechantillonnage,
    on impose la bande uniforme de 3 noeuds sur chaque bord, comme dans
    gestion_cartes.evaluer_carte(). La fraction phi doit ensuite etre
    reverifiee : le reechantillonnage peut legerement la modifier.
    """
    chi = np.asarray(chi)

    if chi.ndim != 2:
        raise ValueError("chi doit etre un tableau 2D.")
    if not np.all(np.isin(chi, [0, 1])):
        raise ValueError("La carte source doit contenir uniquement 0 et 1.")
    if nx_cible < 5 or ny_cible < 5:
        raise ValueError("La grille cible doit avoir au moins 5 noeuds.")

    ny_source, nx_source = chi.shape

    ix = np.rint(np.linspace(0, nx_source - 1, nx_cible)).astype(int)
    iy = np.rint(np.linspace(0, ny_source - 1, ny_cible)).astype(int)
    cible = chi[np.ix_(iy, ix)].copy()

    # Bandes de trois noeuds uniformes a gauche et a droite.
    cible[:, 1] = cible[:, 0]
    cible[:, 2] = cible[:, 0]
    cible[:, -2] = cible[:, -1]
    cible[:, -3] = cible[:, -1]

    # Bandes de trois noeuds uniformes en bas et en haut.
    # Cela impose egalement des blocs uniformes 3x3 aux quatre coins.
    cible[1, :] = cible[0, :]
    cible[2, :] = cible[0, :]
    cible[-2, :] = cible[-1, :]
    cible[-3, :] = cible[-1, :]

    return cible.astype(int)


def verifier_admissibilite(chi, description):
    """Arrete l'etude si la carte ne respecte pas les contraintes du projet."""
    if not evaluer_carte(chi):
        ny, nx = chi.shape
        raise ValueError(
            f"{description} inadmissible ({nx}x{ny}). "
            "Verifier phi <= 0.4, les bandes de 3 noeuds et la grille."
        )


def tracer_carte_temperature(chi, T, titre, chemin):
    """Enregistre la carte chi et le champ de temperature cote a cote."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(
        chi, origin="lower", extent=[0, 1, 0, 1],
        interpolation="nearest", aspect="equal", vmin=0, vmax=1
    )
    axes[0].set_title(r"Carte de materiau $\chi$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], ticks=[0, 1])

    im1 = axes[1].imshow(
        T, origin="lower", extent=[0, 1, 0, 1],
        interpolation="nearest", aspect="equal"
    )
    axes[1].set_title("Temperature $T$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(chemin, dpi=200, bbox_inches="tight")
    plt.close(fig)


def etude_maillage(chi_reference, nom_carte):
    """Evalue la meme conception physique sur 21x21, 41x41 et 81x81."""
    dossier = DOSSIER_RESULTATS / nom_carte / "maillage"
    dossier.mkdir(parents=True, exist_ok=True)
    resultats = []

    for nx, ny in MAILLAGES:
        prm = copy(PRM_DEFAUT)
        chi = adapter_carte(chi_reference, nx, ny)
        verifier_admissibilite(chi, f"Carte adaptee au maillage {nx}x{ny}")

        T, T_moy, T_max, phi = resoudre_et_mesurer(chi, prm)
        resultats.append({
            "nx": nx, "ny": ny,
            "T_moy": T_moy, "T_max": T_max, "phi": phi
        })

        tracer_carte_temperature(
            chi, T, f"{nom_carte} — maillage {nx}x{ny}",
            dossier / f"carte_temperature_{nx}x{ny}.png"
        )
        print(
            f"Maillage {nx}x{ny} : "
            f"T_moy={T_moy:.6g}, T_max={T_max:.6g}, phi={phi:.6g}"
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        [r["nx"] for r in resultats],
        [r["T_moy"] for r in resultats],
        marker="o"
    )
    ax.set_xlabel("Nombre de noeuds par direction")
    ax.set_ylabel("Temperature moyenne")
    ax.set_title(f"Sensibilite au maillage — {nom_carte}")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(dossier / "sensibilite_maillage.png", dpi=200)
    plt.close(fig)

    return resultats


def etude_contraste(chi_reference, nom_carte):
    """Evalue une meme carte pour les contrastes k1/k0 = 10, 100 et 1000."""
    dossier = DOSSIER_RESULTATS / nom_carte / "contraste"
    dossier.mkdir(parents=True, exist_ok=True)
    verifier_admissibilite(chi_reference, "Carte de reference")
    resultats = []

    for contraste in CONTRASTES:
        prm = copy(PRM_DEFAUT)
        prm.k1 = 1.0
        prm.k0 = prm.k1 / contraste

        T, T_moy, T_max, phi = resoudre_et_mesurer(chi_reference, prm)
        resultats.append({
            "contraste": contraste, "k0": prm.k0, "k1": prm.k1,
            "T_moy": T_moy, "T_max": T_max, "phi": phi
        })

        tracer_carte_temperature(
            chi_reference, T,
            f"{nom_carte} — contraste k1/k0={contraste}",
            dossier / f"carte_temperature_contraste_{contraste}.png"
        )
        print(
            f"Contraste {contraste} : k0={prm.k0:g}, k1={prm.k1:g}, "
            f"T_moy={T_moy:.6g}, T_max={T_max:.6g}, phi={phi:.6g}"
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        [r["contraste"] for r in resultats],
        [r["T_moy"] for r in resultats],
        marker="o", label="$T_{moy}$"
    )
    ax.plot(
        [r["contraste"] for r in resultats],
        [r["T_max"] for r in resultats],
        marker="s", label="$T_{max}$"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Contraste de conductivite $k_1/k_0$")
    ax.set_ylabel("Temperature")
    ax.set_title(f"Sensibilite au contraste — {nom_carte}")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dossier / "sensibilite_contraste.png", dpi=200)
    plt.close(fig)

    return resultats


def sauvegarder_csv(resultats, chemin, colonnes):
    """Ecrit les resultats dans un fichier CSV."""
    with open(chemin, "w", encoding="utf-8") as fichier:
        fichier.write(",".join(colonnes) + "\n")
        for resultat in resultats:
            fichier.write(
                ",".join(str(resultat[c]) for c in colonnes) + "\n"
            )


def lancer_etudes(chemin_csv):
    """Lance les deux etudes pour une carte optimale."""
    chemin_csv = Path(chemin_csv)
    nom_carte = chemin_csv.stem

    chi = lire_carte(str(chemin_csv))
    verifier_admissibilite(chi, "Carte source")
    print(f"\nCarte {chemin_csv} : {chi.shape[1]}x{chi.shape[0]}")

    resultats_maillage = etude_maillage(chi, nom_carte)
    resultats_contraste = etude_contraste(chi, nom_carte)

    dossier = DOSSIER_RESULTATS / nom_carte
    dossier.mkdir(parents=True, exist_ok=True)

    sauvegarder_csv(
        resultats_maillage,
        dossier / "resultats_maillage.csv",
        ["nx", "ny", "T_moy", "T_max", "phi"]
    )
    sauvegarder_csv(
        resultats_contraste,
        dossier / "resultats_contraste.csv",
        ["contraste", "k0", "k1", "T_moy", "T_max", "phi"]
    )
    print(f"Resultats enregistres dans {dossier}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage : python etudes_sensibilite.py "
            "carte_optimale.csv [autre_carte.csv ...]"
        )
        sys.exit(1)

    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    for fichier in sys.argv[1:]:
        lancer_etudes(fichier)
