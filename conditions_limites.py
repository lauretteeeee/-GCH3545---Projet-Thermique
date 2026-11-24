#from solveur import PRM_DEFAUT

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

    #pas spatial hx et h
    hx = prm.Lx/(nx-1)
    hy = prm.Ly/(ny-1)


    def indice(i,j,nx): #traduit les coorodonnees (i,j) en indice m de la matrice 
      m = i + j*nx
      return m


    #parcourir les 4 bords du domaine
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

                   b[m]= -2*hy*valeur_cl/k[j,i]

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

                    b[m]= -2*hy*valeur_cl/k[j,i]
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

                    b[m]= -2*hx*valeur_cl/k[j,i]
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

                    b[m]= -2*hx*valeur_cl/k[j,i]

    return A_lil,b


def mesurer_performance(sol, chi, prm) :

    ny,nx = chi.shape
    sol_2d=np.reshape(sol,(ny,nx))

    #température maximale dans le domaine
    T_max = np.max(sol)

    #température moyenne dans le domaine

    hx=prm.Lx/(nx-1)
    hy=prm.Ly/(ny-1)

    #integration selon y
    integrale_y= []

    for j in range (ny):
        somme= 0.5*sol_2d[j,0]

        for i in range(1,nx-1):
            somme+=sol_2d[j,i]

        somme += 0.5*sol_2d[j,nx-1]      
        integrale_y.append(hy*somme)

    #integration selon x 
    somme_totale= 0.5*integrale_y[0]

    for i in range(1,nx-1):
        somme_totale+=integrale_y[i]
    somme_totale += 0.5*integrale_y[nx-1]

    integrale_totale= hx*somme_totale
    T_moy= integrale_totale/(prm.Lx*prm.Ly)

    #Fraction de matériau conducteur
    
    # Intégration de chi selon y
    integrale_chi_y = []

    for j in range(ny):
        somme = 0.5 * chi[j, 0]

        for i in range(1, nx - 1):
            somme += chi[j, i]

        somme += 0.5 * chi[ny - 1, i]

        integrale_chi_y.append(hy * somme)

    # Intégration selon x
    somme_chi = 0.5 * integrale_chi_y[0]

    for i in range(1, nx - 1):
        somme_chi += integrale_chi_y[i]

    somme_chi += 0.5 * integrale_chi_y[nx - 1]

    integrale_chi = hx * somme_chi

    # Fraction de matériau conducteur
    phi = integrale_chi / (prm.Lx * prm.Ly)

    return T_moy, T_max, phi


