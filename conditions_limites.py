from solveur import PRM_DEFAUT
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

    def indice(i,j,nx): #traduit les coorodonnees (i,j) en indice m de la matrice 
      m = i +nx*j
      return m
    
     #pas spatial hx et hy
    hx = prm.Lx/(prm.nx-1)
    hy = prm.Ly/(prm.ny-1)

    #parcourir les 4 bords du domaine
    for bord in ["gauche", "droite", "haut", "bas"] :
        if bord == "gauche" :
            i = 0
            for j in range(prm.ny) :
                m = indice(i,j,prm.nx)

                type_cl=cl["gauche"]["type"][j]
                valeur_cl=cl["gauche"]["valeur"][j]

                if type_cl=="D":   
                 A_lil[m,:] = 0.0
                 A_lil[m,m] = 1.0
                 b[m] = valeur_cl

                elif type_cl=="N":
                   m1 = indice(i+1,j,prm.nx)
                   m2 = indice(i+2,j,prm.nx)

                   A_lil[m,:]=0.0
                   A_lil[m,m] = 3.0
                   A_lil[m,m1] = -4.0
                   A_lil[m,m2] = 1.0

                   b[m]= -2*hy*valeur_cl/k[i,j]

        elif bord == "droite" :
            i = prm.nx - 1
            for j in range(prm.ny) :
                m = indice(i,j,prm.nx)

                type_cl=cl["droite"]["type"][j]
                valeur_cl=cl["droite"]["valeur"][j]

                if type_cl=="D":   
                    A_lil[m,:] = 0.0
                    A_lil[m,m] = 1.0
                    b[m] = valeur_cl

                elif type_cl=="N":
                    m1 = indice(i-1,j,prm.nx)
                    m2 = indice(i-2,j,prm.nx)
                    A_lil[m,:]=0.0
                    A_lil[m,m] = 3.0
                    A_lil[m,m1] = -4.0
                    A_lil[m,m2] = 1.0

                    b[m]= -2*hy*valeur_cl/k[i,j]
        elif bord == "haut" :
            j = prm.ny - 1
            for i in range(prm.nx) :
                m = indice(i,j,prm.nx)

                type_cl=cl["haut"]["type"][i]
                valeur_cl=cl["haut"]["valeur"][i]

                if type_cl=="D":   
                    A_lil[m,:] = 0.0
                    A_lil[m,m] = 1.0
                    b[m] = valeur_cl

                elif type_cl=="N":
                    m1 = indice(i,j-1,prm.nx)
                    m2 = indice(i,j-2,prm.nx)
                    A_lil[m,:]=0.0
                    A_lil[m,m] = 3.0
                    A_lil[m,m1] = -4.0
                    A_lil[m,m2] = 1.0

                    b[m]= -2*hx*valeur_cl/k[i,j]
        elif bord == "bas" :
            j = 0
            for i in range(prm.nx) :
                m = indice(i,j,prm.nx)

                type_cl=cl["bas"]["type"][i]
                valeur_cl=cl["bas"]["valeur"][i]

                if type_cl=="D":   
                    A_lil[m,:] = 0.0
                    A_lil[m,m] = 1.0
                    b[m] = valeur_cl

                elif type_cl=="N":
                    m1 = indice(i,j+1,prm.nx)
                    m2 = indice(i,j+2,prm.nx)
                    A_lil[m,:]=0.0
                    A_lil[m,m] = 3.0
                    A_lil[m,m1] = -4.0
                    A_lil[m,m2] = 1.0

                    b[m]= -2*hx*valeur_cl/k[i,j]

    return A_lil,b


def mesurer_performance(sol, chi, prm) :

    #température maximale dans le domaine
    T_max = np.max(sol)

    #température moyenne dans le domaine

    hx=prm.Lx/(prm.nx-1)
    hy=prm.Ly/(prm.ny-1)

    #integration selon y
    integrale_y= []

    for i in range (prm.nx):
        somme= 0.5*sol[i,0]
        for j in range(1,prm.ny-1):
            somme+=sol[i,j]
        somme+=0.5*sol[i,prm.ny-1]      
        integrale_y.append(hy*somme)

    #integration selon x 
    somme_totale= 0.5*integrale_y[0]
    for i in range(1,prm.nx-1):
        somme_totale+=integrale_y[i]
    somme_totale+=0.5*integrale_y[prm.nx-1]

    integrale_totale= hx*somme_totale
    T_moy= integrale_totale/(prm.Lx*prm.Ly)

    #Fraction de matériau conducteur
    
    #integration de chi selon y 
    integrale_chi_y=[]

    return T_moy, T_max, phi


