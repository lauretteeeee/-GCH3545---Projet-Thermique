import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# On importe UNIQUEMENT tes fonctions depuis verification.py
import verification as verif


class PrmMock:
    Lx = 1.0
    Ly = 1.0
    k0 = 1e-2
    k1 = 1.0


def assembler_mock(chi, cl, prm, terme_source=None):
    """Mini-solveur de référence à k uniforme pour tester verification.py."""
    ny, nx = chi.shape
    N = nx * ny
    h = prm.Lx / (nx - 1)
    k = prm.k1 if chi[0, 0] == 1 else prm.k0

    A = lil_matrix((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    def m_idx(i, j):
        return i + nx * j

    # Nœuds intérieurs : -k * (d2T/dx2 + d2T/dy2) = Q(x, y)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            m = m_idx(i, j)
            A[m, m] = 4.0 * k / h**2
            A[m, m_idx(i + 1, j)] = -k / h**2
            A[m, m_idx(i - 1, j)] = -k / h**2
            A[m, m_idx(i, j + 1)] = -k / h**2
            A[m, m_idx(i, j - 1)] = -k / h**2
            b[m] = terme_source(i * h, j * h)

    # Bords Dirichlet / Neumann (formule de Gear d'ordre 2)[cite: 10, 24]
    def bord(i, j, typ, val, di, dj):
        m = m_idx(i, j)
        A[m, :] = 0.0
        if typ == "D":
            A[m, m] = 1.0
            b[m] = val
        else:
            A[m, m] = 3.0
            A[m, m_idx(i + di, j + dj)] = -4.0
            A[m, m_idx(i + 2 * di, j + 2 * dj)] = 1.0
            b[m] = -2.0 * h * val / k

    for i in range(nx):
        bord(i, 0, cl["bas"]["type"][i], cl["bas"]["valeur"][i], 0, 1)
        bord(i, ny - 1, cl["haut"]["type"][i], cl["haut"]["valeur"][i], 0, -1)
    for j in range(ny):
        bord(0, j, cl["gauche"]["type"][j], cl["gauche"]["valeur"][j], 1, 0)
        bord(nx - 1, j, cl["droite"]["type"][j], cl["droite"]["valeur"][j], -1, 0)

    return A.tocsr(), b


# Injection temporaire (Monkey-Patching) dans ton module verification
verif.PRM_DEFAUT = PrmMock()
verif.assembler_systeme = assembler_mock
verif.resoudre = lambda A, b: spsolve(A, b)

if __name__ == "__main__":
    print("=== Test autonome de tes fonctions de verification.py ===")

    # 1. Test unitaire des dérivées et du terme source SymPy
    Tm, Qm, dTm_dx, dTm_dy = verif.generer_solution_manufacturee(k_val=1.0)
    assert np.isclose(Tm(0.5, 0.0), 1.0), "Erreur sur Tm(0.5, 0)"
    assert np.isclose(Qm(0.5, 0.0), 2.0 * np.pi**2), "Erreur sur le laplacien Qm"
    assert np.isclose(dTm_dx(0.0, 0.0), np.pi), "Erreur sur dTm_dx"
    print("[OK] generer_solution_manufacturee : formules SymPy exactes.")

    # 2. Test unitaire de la structure du dictionnaire cl
    cl_test = verif.construire_cl_manufacturees(21, 21, 1.0, Tm, dTm_dx, dTm_dy)
    assert set(cl_test.keys()) == {"bas", "haut", "gauche", "droite"}
    assert len(cl_test["gauche"]["valeur"]) == 21
    print("[OK] construire_cl_manufacturees : format et dimensions valides.")

    # 3. Exécution complète de ton étude de convergence
    hs, L2_1, Linf_1 = verif.etude_convergence(chi_val=1, label="Conducteur (Mock)")
    _, L2_0, Linf_0 = verif.etude_convergence(chi_val=0, label="Isolant (Mock)")

    # Vérification automatique que l'ordre final est bien proche de 2[cite: 24]
    ordre_final = np.log(L2_1[-2] / L2_1[-1]) / np.log(2)
    assert 1.95 <= ordre_final <= 2.05, f"Ordre inattendu : {ordre_final}"
    print(f"\n[SUCCÈS TOTAL] Ton code verification.py est validé (ordre p = {ordre_final:.3f}) !")

if __name__ == "__main__":
    hs, L2_1, Linf_1 = verif.etude_convergence(chi_val=1, label="Conducteur")
    _, L2_0, Linf_0 = verif.etude_convergence(chi_val=0, label="Isolant")

    plt.figure()
    plt.loglog(hs, L2_1, "o-", linewidth=2.5, markersize=8, label=r"$L_2$ ($\chi=1$, $k=k_1$)")
    plt.loglog(hs, L2_0, "s--", linewidth=1.5, markersize=5, label=r"$L_2$ ($\chi=0$, $k=k_0$)")
    plt.loglog(hs, [L2_1[0] * (h / hs[0])**2 for h in hs], "k--", label="Pente 2")
    plt.xlabel("Pas de grille $h$")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("convergence_msm.png", dpi=200)
    plt.show()  