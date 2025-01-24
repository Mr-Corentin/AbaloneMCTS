# test_move_piece.py
import jax
import jax.numpy as jnp
import numpy as np
from new_jax_fct import (
    initialiser_plateau,
    move_piece_jax,
    move_piece_wrapper
)
from moves import move_piece as move_piece_old
def test_move_piece():
    """Test les mouvements d'une seule bille"""
    board = initialiser_plateau()
    print("\nTest de move_piece:")
    
    # Cas de test avec leurs descriptions
    test_cases = [
        # Mouvements valides
        ((0, 1, 0, 2), "Mouvement simple vers une case vide"),
        ((2, 2, 2, 3), "Mouvement horizontal"),
        ((4, 1, 5, 2), "Mouvement vers la gauche"),
    ]
    
    for coords, description in test_cases:
        print(f"\nTest: {description}")
        start_x, start_y, end_x, end_y = coords
        print(f"Mouvement: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
        
        # Copier le plateau pour les deux tests
        board_old = np.array(board)
        board_new = jnp.array(board)
        
        print("\nPlateau initial:")
        print(board_old)
        
        # Test avec l'ancienne implémentation
        old_success, old_msg = move_piece_old(board_old, start_x, start_y, end_x, end_y)
        print(f"\nAncienne implémentation: Succès={old_success}")
        print(f"Message: {old_msg}")
        print("État après mouvement (ancien):")
        print(board_old)
        
        # Test avec la nouvelle implémentation
        try:
            new_board, new_success, new_msg = move_piece_wrapper(board_new, start_x, start_y, end_x, end_y)
            print(f"\nNouvelle implémentation: Succès={new_success}")
            print(f"Message: {new_msg}")
            print("État après mouvement (nouveau):")
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
    test_coords = (0, 0, 0, 1)  # Un mouvement simple
    
    print("\nTest de performance:")
    
    # Test de l'ancienne implémentation
    import time
    start = time.time()
    for _ in range(1000):
        _ = move_piece_old(np.array(board), *test_coords)
    old_time = time.time() - start
    print(f"Temps ancien move_piece: {old_time:.4f} secondes")
    
    # Test de la nouvelle implémentation
    # Première exécution pour compilation JIT
    _ = move_piece_wrapper(board, *test_coords)
    
    start = time.time()
    for _ in range(1000):
        _ = move_piece_wrapper(board, *test_coords)
    new_time = time.time() - start
    print(f"Temps nouveau move_piece: {new_time:.4f} secondes")
    print(f"Accélération: {old_time/new_time:.2f}x")

if __name__ == "__main__":
    test_move_piece()
    test_performance()