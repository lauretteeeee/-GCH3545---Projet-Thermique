from gestion_cartes import lire_carte, evaluer_carte

chi = lire_carte("carte_test_non_binaire.csv")
print(chi.shape)
print(evaluer_carte(chi))
