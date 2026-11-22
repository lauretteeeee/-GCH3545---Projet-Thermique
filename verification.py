from solveur import PRM_DEFAUT, assembler_systeme, resoudre
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def generer_solution_manufacturee(k_val) :
    """Retourne les fonctions Python (Tm, Qm, dTm_dx, dTm_dy) pour k constant."""
    x, y = sp.symbols("x y")
    # Fonction lisse non polynomiale
    Tm_expr = sp.sin(sp.pi * x) * sp.cos(sp.pi * y)

    # Terme source de l'ÉDP : Q = -div(k * grad(T)) = -k * (d2T/dx2 + d2T/dy2)
    Qm_expr = -k_val * (sp.diff(Tm_expr, x, 2) + sp.diff(Tm_expr, y, 2))
    dTm_dx_expr = sp.diff(Tm_expr, x)
    dTm_dy_expr = sp.diff(Tm_expr, y)

    # Conversion en fonctions NumPy rapides
    Tm = sp.lambdify((x, y), Tm_expr, "numpy")
    Qm = sp.lambdify((x, y), Qm_expr, "numpy")
    dTm_dx = sp.lambdify((x, y), dTm_dx_expr, "numpy")
    dTm_dy = sp.lambdify((x, y), dTm_dy_expr, "numpy")
    return Tm, Qm, dTm_dx, dTm_dy

def construire_cl_manufacturees(nx, ny, k_val, Tm, dTm_dx, dTm_dy, Lx=1.0, Ly=1.0):
    """Construit le dictionnaire cl (Dirichlet en bas/haut, Neumann à gauche/droite)."""
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Flux sortant qn = -k * dT/dn (normale sortante : -x à gauche, +x à droite). 
    qn_gauche = -k_val * (-dTm_dx(0.0, y))
    qn_droite = -k_val * (+dTm_dx(Lx, y))

    cl = {
        "bas": {"type": np.full(nx, "D"), "valeur": Tm(x, 0.0)},
        "haut": {"type": np.full(nx, "D"), "valeur": Tm(x, Ly)},
        "gauche": {"type": np.full(ny, "N"), "valeur": qn_gauche},
        "droite": {"type": np.full(ny, "N"), "valeur": qn_droite},
    }
    return cl


def etude_convergence(chi_val, label) :
#Exécute la convergence sur 4 grilles pour chi uniforme (0 ou 1)
    k0 = getattr(PRM_DEFAUT, "k0", 1e-2)
    k1 = getattr(PRM_DEFAUT, "k1", 1.0)
    k_val = k1 if chi_val == 1 else k0

    Tm, Qm, dTm_dx, dTm_dy = generer_solution_manufacturee(k_val)

    # 4 grilles successives (n - 1 multiple de 20)
    tailles = [21, 41, 81, 161]
    hs, err_L2, err_Linf = [], [], []

    for n in tailles:
        h = 1.0 / (n - 1)
        chi = np.full((n, n), chi_val, dtype=int)
        cl = construire_cl_manufacturees(n, n, k_val, Tm, dTm_dx, dTm_dy)

        A, b = assembler_systeme(chi, cl, PRM_DEFAUT, terme_source=Qm)
        T_num = resoudre(A, b).reshape((n, n))

        X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        erreur = T_num - Tm(X, Y)

        hs.append(h)
        err_L2.append(np.sqrt(h**2 * np.sum(erreur**2)))
        err_Linf.append(np.max(np.abs(erreur)))

    # Affichage des résultats et ordres p = ln(e_h / e_{h/2}) / ln(2)
    print(f"\n--- Vérification MSM ({label}, k = {k_val}) ---")
    for i in range(len(tailles)):
        if i == 0:
            print(f"n={tailles[i]:3d} | h={hs[i]:.5f} | L2={err_L2[i]:.3e} | Linf={err_Linf[i]:.3e}")
        else:
            p2 = np.log(err_L2[i - 1] / err_L2[i]) / np.log(2)
            pinf = np.log(err_Linf[i - 1] / err_Linf[i]) / np.log(2)
            print(f"n={tailles[i]:3d} | h={hs[i]:.5f} | L2={err_L2[i]:.3e} (p={p2:.2f}) | Linf={err_Linf[i]:.3e} (p={pinf:.2f})")

    return hs, err_L2, err_Linf

if __name__ == "__main__":
    hs, L2_1, Linf_1 = etude_convergence(chi_val=1, label="Conducteur")
    _, L2_0, Linf_0 = etude_convergence(chi_val=0, label="Isolant")

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