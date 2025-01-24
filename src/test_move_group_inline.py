# test_move_group_inline.py
import jax.numpy as jnp
import numpy as np
from new_jax_fct import (
    initialiser_plateau,
    move_group_inline_jax,
    move_group_inline_wrapper
)
from moves import move_group_inline as move_group_inline_old
def test_move_group_inline():
    """Test les mouvements en ligne des groupes"""
    board = initialiser_plateau()
    print("\nTest de move_group_inline:")
    
    # Cas de test avec leurs descriptions
    test_cases = [
        # Mouvements valides
        ([(4, 0), (5, 1)], "SE", "Groupe vertical vers SE"),
        ([(0, 7), (0, 8)], "NW", "Groupe vertical vers NW"),
        ([(3, 0), (3, 1), (3, 2)], "SW", "Groupe de 3 vertical vers SW")
    ]
    
    for coords, direction, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Coordonnées: {coords}, Direction: {direction}")
        
        # Copier le plateau pour les deux tests
        board_old = np.array(board)
        board_new = jnp.array(board)
        
        # Test avec l'ancienne implémentation
        old_success, old_msg = move_group_inline_old(board_old, coords, direction)
        print(f"Ancienne implémentation: Succès={old_success}")
        print(f"Message: {old_msg}")
        if old_success:
            print("État final ancien:")
            print(board_old)
        
        # Test avec la nouvelle implémentation
        try:
            new_board, new_success, new_msg = move_group_inline_wrapper(board_new, coords, direction)
            print(f"Nouvelle implémentation: Succès={new_success}")
            print(f"Message: {new_msg}")
            if new_success:
                print("État final nouveau:")
                print(np.array(new_board))
            
            # Vérifier la cohérence des résultats
            assert bool(old_success) == bool(new_success), f"Incohérence dans le succès pour {description}"
            if old_success and new_success:
                if not np.array_equal(board_old, np.array(new_board)):
                    print("\nDifférences trouvées :")
                    diff = board_old != np.array(new_board)
                    for i in range(board_old.shape[0]):
                        for j in range(board_old.shape[1]):
                            if diff[i,j]:
                                print(f"Position ({j},{i}): ancien={board_old[i,j]}, nouveau={new_board[i,j]}")
                
        except Exception as e:
            print(f"Erreur avec la nouvelle implémentation: {e}")


def test_performance():
    """Test les performances des deux implémentations"""
    board = initialiser_plateau()
    test_move = ([(2, 2), (3, 2)], "W")
    
    print("\nTest de performance:")
    
    # Test de l'ancienne implémentation
    import time
    start = time.time()
    for _ in range(1000):
        _ = move_group_inline_old(np.array(board), *test_move)
    old_time = time.time() - start
    print(f"Temps ancien move_group_inline: {old_time:.4f} secondes")
    
    # Test de la nouvelle implémentation
    # Première exécution pour compilation JIT
    _ = move_group_inline_wrapper(board, *test_move)
    
    start = time.time()
    for _ in range(1000):
        _ = move_group_inline_wrapper(board, *test_move)
    new_time = time.time() - start
    print(f"Temps nouveau move_group_inline: {new_time:.4f} secondes")
    print(f"Accélération: {old_time/new_time:.2f}x")

if __name__ == "__main__":
    test_move_group_inline()
    test_performance()