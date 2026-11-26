import numpy as np
from solveur import PRM_DEFAUT, assembler_systeme, resoudre, mesurer_performance
from conditions_limites import creer_cl_thermique
from gestion_cartes import evaluer_carte

def noeuds_autorises(ny, nx) :
    """Retourne un tableau booléen (ny, nx) : True là où on a le droit
    de modifier chi, False sur la bande interdite de 3 nœuds aux bords.
    Dans evaluer_carte(chi) il y a une regle stricte : les 3 premiers noeuds sur chaque
    bord doivent rester identiques, donc creation d'un tableau 'autorise' rempli de False (interdiction),
    puis des True (autorisation) uniquement dans le carre interieur et loin des bords.
    """

    # Contient que des valeurs logiques Vrai/faux
    autorise = np.zeros((ny, nx), dtype = bool) 

    # Pour 3:ny-3 (en vertical) saute les lignes 0, 1, 2 en bas et les 3 dernières lignes en haut (ny-3, ny-2, ny-1)
    # Pour 3:nx-3 (en horizontal) saute les colonnes 0, 1, 2 a gauche et les 3 dernières colonnes a droites (ny-3, ny-2, ny-1)
    autorise[3:ny-3, 3:nx-3] = True 

    return autorise

def calculer_gradient_thermique(sol_2d, prm) :
     """Calcule le carré de la norme du gradient de température |grad(T)|^2
    en chaque nœud de la grille (ny, nx).
    """
     
     # Récuperer le nombre de lignes (ny) et de colonnes (nx) de la carte de température
     ny, nx = sol_2d.shape

     # Calcul de l'espacement entre 2 noeuds voisins
     h = prm.Lx / (nx - 1)

     # Tableau 2D vide de zéros qui stockera les résultats 
     grad_carre = np.zeros((ny, nx))

     for j in range(1, ny-1) :
          for i in range(1, nx-1) : 
               # Derivée partielle horizontale par différence finie centrée d'ordre 2
               dT_dx = (sol_2d[j, i+1] - sol_2d[j, i-1]) / (2*h)

               # Derivée partielle verticale par différence finie centrée d'ordre 2 
               dT_dy = (sol_2d[j+1, i] - sol_2d[j-1, i]) / (2*h)

               # Carré de la norme du gradient : on évite la racine carrée pour alléger le calcul, car la fonction racine est strictement croissante et conserve donc exactement le même classement des noeuds
               grad_carre[j, i] = dT_dx**2 + dT_dy**2

     return grad_carre 

def optimiser_glouton(chi_init, cl, prm, n_ajout=4, n_iter_echange=40) :
     """Optimise la répartition des matériaux par un algorithme glouton en 2 phases :
    1. Croissance constructive depuis chi_init jusqu'à atteindre phi <= phi_max.
    2. Affinage par échange de nœuds (1 <-> 0) à fraction phi constante.

    Entrées :
        - chi_init : carte initiale binaire de forme (ny, nx)
        - cl : dictionnaire des conditions aux limites
        - prm : objet parametres (Lx, Ly, k0, k1, Q, phi_max)
        - n_ajout : nombre de nœuds ajoutés par itération en Phase 1
        - n_iter_echange : nombre maximal d'itérations d'échange en Phase 2

    Sorties :
        - chi_opt : meilleure carte binaire trouvée (ny, nx)
        - hist_Tmoy : liste de l'évolution de T_moy à chaque étape
        - hist_Tmax : liste de l'évolution de T_max à chaque étape
    """
     # Bloc A : Préparation et première évaluation

        # Copier carte de depart et generer gabarit
     chi = chi_init.copy()
     ny, nx = chi.shape
     autorise = noeuds_autorises(ny, nx)

        # Initialiser les listes de suivi
     hist_Tmoy = []
     hist_Tmax = []

        # Évaluer le départ 
     A, b = assembler_systeme(chi, cl, prm)
     sol = resoudre(A, b)
     sol_2d = sol.reshape((ny, nx))
     T_moy, T_max, phi  = mesurer_performance(sol, chi, prm)

        # Enregistrer les performances initiales
     hist_Tmoy.append(T_moy)
     hist_Tmax.append(T_max)

    # Bloc B : Phase de croissance constructive
     
     while phi < prm.phi_max : 
          grad_carre = calculer_gradient_thermique(sol_2d, prm)

        # 1. Repérer toutes les cases libres (chi == 0) situées dans la zone autorisée
        #    On stocke un tuple (gradient, ligne j, colonne i) pour garder la position lors du tri
          candidats = []
          
          for j in range(ny) :
               for i in range(nx) :
                    if autorise[j, i] and chi[j, i] == 0 :
                         candidats.append((grad_carre[j, i], j, i))
        
        # Sécurité : s'il n'y a plus aucune case libre autorisée, arrêt de la croissance
          if len(candidats) == 0 :
               break 
          
        # 2. Trier les candidats du plus grand gradient au plus petit
          candidats.sort(reverse=True)

        # 3. Sauvegarder la carte actuelle avant modification, puis passer les n_ajout 
        #    meilleurs cases isolantes (0) en matériau conducteur (1)
          chi_sauvegarde = chi.copy()
          nb_a_ajouter = min(n_ajout, len(candidats))
          for k in range(nb_a_ajouter) :
               val_gradient, j_best, i_best = candidats[k]
               chi[j_best, i_best] = 1
        
        # 4. Vérifier que la nouvelle carte respecte toujours les critères (dont phi <= 0.4)
        #    Si ce dernier ajout fait dépasser la limite, on annule et on quitte la phase de croissance
          if not evaluer_carte(chi) :
               chi = chi_sauvegarde
               break
          
        # 5. Recalculer le champ de température et les nouvelles performances pour cette carte
        #    (On met à jour sol_2d et phi pour l'itération suivante du while)
          A, b = assembler_systeme(chi, cl, prm)
          sol = resoudre(A, b)
          sol_2d = sol.reshape((ny, nx))
          T_moy, T_max, phi  = mesurer_performance(sol, chi, prm)        

        # Enregistrer les nouvelles performances dans l'historique de convergence
          hist_Tmoy.append(T_moy)
          hist_Tmax.append(T_max)     
          
     pass
