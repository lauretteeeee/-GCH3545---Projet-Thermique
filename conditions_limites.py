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

    ny,nx = k.shape

    #pas spatial hx et hy
    hx = prm.Lx/(nx-1)
    hy = prm.Ly/(ny-1)


    def indice(i,j,nx): #traduit les coorodonnees (i,j) en indice m de la matrice 
      m = i + j*nx
      return m


    #parcourir les 4 bords du domaine
    #Gear d'ordre 2 : la derivee est dans la direction normale au bord
    #   bords gauche/droite -> derivee en x -> pas hx
    #   bords haut/bas      -> derivee en y -> pas hy
    for bord in ["gauche", "droite", "haut", "bas"] :
        if bord == "gauche" :
            i = 0
            for j in range(ny) :
                m = indice(i,j,nx)

                type_cl=cl["gauche"]["type"][j]
                valeur_cl=cl["gauche"]["valeur"][j]

                if type_cl=="D":   
                    A_lil[m,:] = 0.0
                    A_lil[m,m] = 1.0
                    b[m] = valeur_cl

                elif type_cl=="N":
                    m1 = indice(i+1,j,nx)
                    m2 = indice(i+2,j,nx)

                    A_lil[m,:]=0.0
                    A_lil[m,m] = 3.0
                    A_lil[m,m1] = -4.0
                    A_lil[m,m2] = 1.0

                    b[m]= -2*hx*valeur_cl/k[j,i]

        elif bord == "droite" :
            i = nx - 1
            for j in range(ny) :
                m = indice(i,j,nx)

                type_cl=cl["droite"]["type"][j]
                valeur_cl=cl["droite"]["valeur"][j]

                if type_cl=="D":   
                    A_lil[m,:] = 0.0
                    A_lil[m,m] = 1.0
                    b[m] = valeur_cl

                elif type_cl=="N":
                    m1 = indice(i-1,j,nx)
                    m2 = indice(i-2,j,nx)
                    A_lil[m,:]=0.0
                    A_lil[m,m] = 3.0
                    A_lil[m,m1] = -4.0
                    A_lil[m,m2] = 1.0

                    b[m]= -2*hx*valeur_cl/k[j,i]

        elif bord == "haut" :
            j = ny - 1
            for i in range(nx) :
                m = indice(i,j,nx)

                type_cl=cl["haut"]["type"][i]
                valeur_cl=cl["haut"]["valeur"][i]

                if type_cl=="D":   
                    A_lil[m,:] = 0.0
                    A_lil[m,m] = 1.0
                    b[m] = valeur_cl

                elif type_cl=="N":
                    m1 = indice(i,j-1,nx)
                    m2 = indice(i,j-2,nx)
                    A_lil[m,:]=0.0
                    A_lil[m,m] = 3.0
                    A_lil[m,m1] = -4.0
                    A_lil[m,m2] = 1.0

                    b[m]= -2*hy*valeur_cl/k[j,i]

        elif bord == "bas" :
            j = 0
            for i in range(nx) :
                m = indice(i,j,nx)

                type_cl=cl["bas"]["type"][i]
                valeur_cl=cl["bas"]["valeur"][i]

                if type_cl=="D":   
                    A_lil[m,:] = 0.0
                    A_lil[m,m] = 1.0
                    b[m] = valeur_cl

                elif type_cl=="N":
                    m1 = indice(i,j+1,nx)
                    m2 = indice(i,j+2,nx)
                    A_lil[m,:]=0.0
                    A_lil[m,m] = 3.0
                    A_lil[m,m1] = -4.0
                    A_lil[m,m2] = 1.0

                    b[m]= -2*hy*valeur_cl/k[j,i]


    # ---------------- COINS ----------------
    # Un coin appartient a deux bords (un vertical : gauche/droite, un horizontal : bas/haut).
    # Les boucles ci-dessus l'ont deja traite deux fois, la derniere ecriture gagnait.
    # Regle (chargé de projet) :
    #   - si l'un des deux bords est Dirichlet : le coin est Dirichlet (valeur imposee)
    #   - si les deux sont Neumann : on ADDITIONNE l'equation de Gear en x et celle en y
    # chaque coin : (i, j, bord vertical, indice sur ce bord, bord horizontal, indice sur ce bord,
    #                sens vers l'interieur en x, sens vers l'interieur en y)
    coins = [(0,    0,    "gauche", 0,    "bas",  0,    +1, +1),
             (nx-1, 0,    "droite", 0,    "bas",  nx-1, -1, +1),
             (0,    ny-1, "gauche", ny-1, "haut", 0,    +1, -1),
             (nx-1, ny-1, "droite", ny-1, "haut", nx-1, -1, -1)]

    for (i, j, bord_v, iv, bord_h, ih, sx, sy) in coins :
        m = indice(i,j,nx)

        type_v = cl[bord_v]["type"][iv]
        valeur_v = cl[bord_v]["valeur"][iv]
        type_h = cl[bord_h]["type"][ih]
        valeur_h = cl[bord_h]["valeur"][ih]

        A_lil[m,:] = 0.0

        if type_v == "D" or type_h == "D" :
            # Dirichlet prioritaire
            A_lil[m,m] = 1.0
            if type_v == "D" :
                b[m] = valeur_v
            else :
                b[m] = valeur_h

        else :
            # Neumann des deux cotes : equation de Gear en x + equation de Gear en y
            #   (3T0 - 4T1x + T2x) + (3T0 - 4T1y + T2y) = -2 hx qx/k - 2 hy qy/k
            m1x = indice(i+sx,j,nx)
            m2x = indice(i+2*sx,j,nx)
            m1y = indice(i,j+sy,nx)
            m2y = indice(i,j+2*sy,nx)

            A_lil[m,m] = 6.0
            A_lil[m,m1x] = -4.0
            A_lil[m,m2x] = 1.0
            A_lil[m,m1y] = -4.0
            A_lil[m,m2y] = 1.0

            b[m] = -2*hx*valeur_v/k[j,i] - 2*hy*valeur_h/k[j,i]

    return A_lil,b



def mesurer_performance(sol, chi, prm):

    ny, nx = chi.shape
    sol_2d = np.reshape(sol, (ny, nx))

    # Pas spatiaux
    hx = prm.Lx / (nx - 1)
    hy = prm.Ly / (ny - 1)

    # Température maximale dans le domaine
    T_max = np.max(sol_2d)

    # --------------------------------
    # Température moyenne
    # --------------------------------

    # Intégration selon x, pour chaque ligne j
    integrale_y = []

    for j in range(ny):
        somme = 0.5 * sol_2d[j, 0]

        for i in range(1, nx - 1):
            somme += sol_2d[j, i]

        somme += 0.5 * sol_2d[j, nx - 1]

        integrale_y.append(hx * somme)

    # Intégration selon y
    somme_totale = 0.5 * integrale_y[0]

    for j in range(1, ny - 1):
        somme_totale += integrale_y[j]

    somme_totale += 0.5 * integrale_y[ny - 1]

    integrale_totale = hy * somme_totale

    T_moy = integrale_totale / (prm.Lx * prm.Ly)

    # --------------------------------
    # Fraction de matériau conducteur
    # --------------------------------

    # Intégration selon x, pour chaque ligne j
    integrale_chi_y = []

    for j in range(ny):
        somme = 0.5 * chi[j, 0]

        for i in range(1, nx - 1):
            somme += chi[j, i]

        somme += 0.5 * chi[j, nx - 1]

        integrale_chi_y.append(hx * somme)

    # Intégration selon y
    somme_chi = 0.5 * integrale_chi_y[0]

    for j in range(1, ny - 1):
        somme_chi += integrale_chi_y[j]

    somme_chi += 0.5 * integrale_chi_y[ny - 1]

    integrale_chi = hy * somme_chi

    phi = integrale_chi / (prm.Lx * prm.Ly)

    return T_moy, T_max, phi

