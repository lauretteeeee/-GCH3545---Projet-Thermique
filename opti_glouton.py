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
        - hist_phi : liste de l'évolution de phi à chaque étape
    """
     # BLOC A : PRÉPARATION ET PREMIÈRE ÉVALUATION 

        # Copier carte de depart et generer gabarit
     chi = chi_init.copy()
     ny, nx = chi.shape
     autorise = noeuds_autorises(ny, nx)

        # Initialiser les listes de suivi
     hist_Tmoy = []
     hist_Tmax = []
     hist_phi = []

        # Évaluer le départ 
     A, b = assembler_systeme(chi, cl, prm)
     sol = resoudre(A, b)
     sol_2d = sol.reshape((ny, nx))
     T_moy, T_max, phi  = mesurer_performance(sol, chi, prm)

        # Enregistrer les performances initiales
     hist_Tmoy.append(T_moy)
     hist_Tmax.append(T_max)
     hist_phi.append(phi)

     # BLOC B : PHASE 1 DE CROISSANCE CONSTRUCTIVE
     
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
          hist_phi.append(phi)  

     # BLOC C : PHASE 2 D'AFFINAGE PAR ÉCHANGE (AVEC PHI CONSTANTE)
          
     # Principe : maintenant que le budget de conducteur est atteint (phi <= 0.4),
     # on déplace les nœuds conducteurs les moins utiles (faible gradient) vers les
     # noeuds isolants où la chaleur est encore bloquée (fort gradient)

     k_ech = n_ajout # Nombre de paires de noeuds à échanger à chaque itération

     for ite in range(n_iter_echange) :
          grad_carre = calculer_gradient_thermique(sol_2d, prm)   

          # 1. Lister séparément les cases conductrices (chi == 1) et isolantes (chi == 0)
          #    uniquement dans la zone intérieure autorisée
          conducteurs = []
          isolants = []
          for j in range(ny) :
               for i in range(nx) :
                    if autorise[j, i] :
                         if chi[j, i] == 1 :
                            conducteurs.append((grad_carre[j, i], j, i))
                         else : 
                            isolants.append((grad_carre[j, i], j, i))

          # Sécurité : s'il n'y a plus assez de cases à échanger, arrêt
          if len(conducteurs) == 0 or len(isolants) == 0 :
               break
          
          # 2. Trier les conducteurs par ordre croissant (les moins utiles en premier)
          #    et les isolants par ordre décroissant (les plus critiques en premier)
          conducteurs.sort()
          isolants.sort(reverse = True)

          # 3. Sauvegarder la carte avant de tester l'échange
          chi_sauvegarde = chi.copy()
          nb_a_echanger = min(k_ech, len(conducteurs), len(isolants))

          for k in range(nb_a_echanger) :
               val_grad_cond, j_retire, i_retire = conducteurs[k]
               val_grad_iso, j_ajoute, i_ajoute = isolants[k]
               chi[j_retire, i_retire] = 0 # On retire le conducteur peu utile
               chi[j_ajoute, i_ajoute] = 1 # On le place là où le gradient est fort
        
          # 4. Évaluer la nouvelle carte après l'échange
          A, b = assembler_systeme(chi, cl, prm)
          sol = resoudre(A, b)
          nouveau_Tmoy, nouveau_Tmax, nouveau_phi = mesurer_performance(sol, chi, prm)

          # 5. Critère d'acceptation : on garde l'échange uniquement s'il réduit T_moy
          if nouveau_Tmoy < T_moy :
               T_moy = nouveau_Tmoy
               T_max = nouveau_Tmax
               phi = nouveau_phi
               sol_2d = sol.reshape((ny, nx))
               
               hist_Tmoy.append(T_moy)
               hist_Tmax.append(T_max)
               hist_phi.append(phi) 
          
          else : 
               # Si l'échange dégrade ou n'améliore pas T_moy, on annule la modification
               chi = chi_sauvegarde
               # On réduit le nombre d'échanges simultanés pour un réglage plus précis
               if k_ech > 1 :
                    k_ech = k_ech // 2
               else : 
                    # Un échange de 1 seul noeud n'améliore plus rien : la convergence est atteinte
                    break
                       
     return chi, hist_Tmoy, hist_Tmax, hist_phi


# Esai en attendant les autres fonctions puis à supprimer 
if __name__ == "__main__":
    # Test rapide sur une grille 21x21 en partant d'une grille vide (que des 0)
    ny, nx = 21, 21
    chi_depart = np.zeros((ny, nx), dtype=int)
    cl = creer_cl_thermique(nx, ny)

    chi_opt, hist_Tmoy, hist_Tmax, hist_phi = optimiser_glouton(chi_depart, cl, PRM_DEFAUT, n_ajout=2, n_iter_echange=30)

    print(f"T_moy initiale : {hist_Tmoy[0]:.4f}  -->  T_moy optimisée : {hist_Tmoy[-1]:.4f}")
    print(f"T_max initiale : {hist_Tmax[0]:.4f}  -->  T_max optimisée : {hist_Tmax[-1]:.4f}")
    print(f"Fraction finale phi : {hist_phi[-1]:.4f}")
    print(f"Carte finale admissible (evaluer_carte) : {evaluer_carte(chi_opt)}")
