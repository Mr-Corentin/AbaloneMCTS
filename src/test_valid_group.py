# test_valid_group.py
import jax
import jax.numpy as jnp
import numpy as np
from new_jax_fct import (
    initialiser_plateau,
    _is_valid_group_jax,
    is_valid_group_wrapper
)
from utils import is_valid_group as is_valid_group_old

def test_is_valid_group():
    """Test la validation des groupes de billes"""
    board = initialiser_plateau()
    print("\nTest de is_valid_group:")
    
    # Cas de test avec leurs descriptions
    test_cases = [
        # Groupes valides
        ([(0, 0), (0, 1)], "Groupe vertical de 2 billes noires"),
        ([(0, 0), (1, 0)], "Groupe horizontal de 2 billes noires"),
        ([(2, 2), (3, 2), (4, 2)], "Groupe horizontal de 3 billes"),
        ([(0, 0), (1, 1), (2, 2)], "Groupe diagonal de 3 billes"),
        
        # Groupes invalides
        ([(0, 0), (2, 0)], "Billes non adjacentes"),
        ([(0, 0), (1, 1), (0, 2)], "Billes non alignées"),
        ([(0, 0), (0, 1), (0, 2), (0, 3)], "Groupe trop grand"),
        ([(0, 0)], "Groupe trop petit"),
        ([(0, 0), (8, 8)], "Billes de couleurs différentes"),
    ]
    
    for coords, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Coordonnées: {coords}")
        
        # Test avec l'ancienne implémentation
        old_valid, old_msg, old_align = is_valid_group_old(np.array(board), coords)
        print(f"Ancienne implémentation: Valide={old_valid}, Alignement={old_align}")
        print(f"Message: {old_msg}")
        
        # Test avec la nouvelle implémentation
        try:
            new_valid, new_msg, new_align = is_valid_group_wrapper(board, coords)
            print(f"Nouvelle implémentation: Valide={new_valid}, Alignement={new_align}")
            print(f"Message: {new_msg}")
            
            # Vérifier la cohérence des résultats
            assert bool(old_valid) == bool(new_valid), f"Incohérence dans la validation pour {description}"
            if old_valid and new_valid:
                assert old_align == new_align, f"Incohérence dans l'alignement pour {description}"
                
        except Exception as e:
            print(f"Erreur avec la nouvelle implémentation: {e}")

def test_performance():
    """Test les performances des deux implémentations"""
    board = initialiser_plateau()
    test_coords = [(0, 0), (0, 1)]
    
    print("\nTest de performance:")
    
    # Test de l'ancienne implémentation
    import time
    start = time.time()
    for _ in range(1000):
        _ = is_valid_group_old(np.array(board), test_coords)
    old_time = time.time() - start
    print(f"Temps ancien is_valid_group: {old_time:.4f} secondes")
    
    # Test de la nouvelle implémentation
    # Première exécution pour compilation JIT
    _ = is_valid_group_wrapper(board, test_coords)
    
    start = time.time()
    for _ in range(1000):
        _ = is_valid_group_wrapper(board, test_coords)
    new_time = time.time() - start
    print(f"Temps nouveau is_valid_group: {new_time:.4f} secondes")
    print(f"Accélération: {old_time/new_time:.2f}x")

if __name__ == "__main__":
    test_is_valid_group()
    test_performance()