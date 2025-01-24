import jax
import jax.numpy as jnp
import numpy as np
from board import initialize_board, create_custom_board
from moves import move_single_marble, move_group_inline, move_group_parallel
from legal_moves import get_legal_moves_debug, get_legal_moves, create_player_positions_mask

def debug_legal_moves():
    """Teste la génération des coups légaux sur le plateau initial et sauvegarde les résultats"""
    # Charger les mouvements précalculés
    moves_data = np.load("move_map.npz")
    moves_index = {
        'positions': moves_data['positions'],
        'directions': moves_data['directions'],
        'move_types': moves_data['move_types'],
        'group_sizes': moves_data['group_sizes']
    }
    
    # Créer le plateau initial
    board = initialize_board()
    
    # Obtenir les coups légaux pour le joueur noir (1)
    legal_moves = get_legal_moves(board, moves_index, 1)
    legal_moves_array = np.array(legal_moves)
    
    with open('legal_moves_analysis.txt', 'w') as f:
        # Écrire les statistiques générales
        f.write("=== ANALYSE DES MOUVEMENTS LÉGAUX POUR LE JOUEUR NOIR ===\n\n")
        
        # Statistiques globales
        total_moves = len(moves_index['directions'])
        total_legal_moves = jnp.sum(legal_moves)
        f.write(f"Nombre total de mouvements dans l'index: {total_moves}\n")
        f.write(f"Nombre total de mouvements légaux: {total_legal_moves}\n\n")
        
        # Statistiques par taille de groupe
        for group_size in [1, 2, 3]:
            f.write(f"\n{'='*50}\n")
            f.write(f"GROUPE DE {group_size} BILLE{'S' if group_size > 1 else ''}\n")
            f.write(f"{'='*50}\n\n")
            
            # Filtrer les mouvements pour cette taille de groupe
            size_mask = moves_index['group_sizes'] == group_size
            legal_size_mask = legal_moves_array & size_mask
            moves_indices = np.where(legal_size_mask)[0]
            
            # Compter les types de mouvements pour cette taille
            moves_count = np.sum(legal_size_mask)
            f.write(f"Nombre total de mouvements: {moves_count}\n\n")
            
            if moves_count > 0:
                # Organiser par type de mouvement
                for move_type in range(3):
                    type_name = ["SINGLE", "PARALLEL", "INLINE"][move_type]
                    type_moves = [idx for idx in moves_indices if moves_index['move_types'][idx] == move_type]
                    
                    if type_moves:
                        f.write(f"\n--- Mouvements de type {type_name} ---\n")
                        for idx in type_moves:
                            positions = moves_index['positions'][idx]
                            direction = moves_index['directions'][idx]
                            dir_name = ["NE", "E", "SE", "SW", "W", "NW"][direction]
                            
                            # Ne garder que les positions non-nulles
                            #valid_positions = positions[positions.sum(axis=1) != 0]
                            valid_positions = positions[:group_size]  # Pour affichage uniquement, pas dans les calculs

                            
                            f.write(f"\nMouvement {idx}:\n")
                            f.write(f"Direction: {dir_name}\n")
                            f.write("Positions:\n")
                            for pos in valid_positions:
                                f.write(f"  ({pos[0]}, {pos[1]}, {pos[2]})\n")
                            f.write("-" * 30 + "\n")
                            
            else:
                f.write("Aucun mouvement légal pour cette taille de groupe\n")
            
            f.write("\n")
        
        print(f"Analyse sauvegardée dans 'legal_moves_analysis.txt'")

# if __name__ == "__main__":
#     debug_legal_moves()



def test_push_rules():
    """Teste différents scénarios de poussée"""
    # Charger l'index des mouvements
    moves_data = np.load("move_map.npz")
    moves_index = {
        'positions': moves_data['positions'],
        'directions': moves_data['directions'],
        'move_types': moves_data['move_types'],
        'group_sizes': moves_data['group_sizes']
    }

    def test_scenario(name: str, marbles: list[tuple[tuple[int, int, int], int]]):
        print(f"\n=== Test: {name} ===")
        board = create_custom_board(marbles)
        legal_moves = get_legal_moves(board, moves_index, 1)  # Pour le joueur noir
        
        # Afficher les coups légaux
        legal_moves_array = np.array(legal_moves)
        moves_found = np.where(legal_moves_array)[0]
        
        print(f"Nombre de coups légaux trouvés: {len(moves_found)}")
        if len(moves_found) > 0:
            print("\nDétail des coups:")
            for idx in moves_found:
                move_type = ["SINGLE", "PARALLEL", "INLINE"][moves_index['move_types'][idx]]
                direction = ["NE", "E", "SE", "SW", "W", "NW"][moves_index['directions'][idx]]
                group_size = moves_index['group_sizes'][idx]
                positions = moves_index['positions'][idx][:group_size]
                
                print(f"\nCoup {idx}:")
                print(f"Type: {move_type}")
                print(f"Direction: {direction}")
                print(f"Taille du groupe: {group_size}")
                print(f"Positions: {positions}")
        print("\n" + "="*50)

    # Test 1: Une bille qui en pousse une autre
    test_scenario(
        "Une bille pousse une bille",
        [
            ((0, 0, 0), 1),    # Bille noire
            ((1, -1, 0), -1),  # Bille blanche
        ]
    )

    # Test 2: Deux billes qui en poussent une
    test_scenario(
        "Deux billes poussent une bille",
        [
            ((0, 0, 0), 1),     # Bille noire
            ((1, -1, 0), 1),    # Bille noire
            ((2, -2, 0), -1),   # Bille blanche
        ]
    )

    # Test 3: Poussée vers l'extérieur
    test_scenario(
        "Poussée vers l'extérieur",
        [
            ((0, 0, 0), 1),      # Bille noire
            ((1, -1, 0), 1),     # Bille noire
            ((2, -2, 0), 1),
            ((2, -3, 0), -1)
                    # Bille blanche à pousser dehors
        ]
    )

    # Test 4: Poussée invalide (force insuffisante)
    test_scenario(
        "Poussée invalide (1 contre 2)",
        [
            ((0, 0, 0), 1),      # Bille noire
            ((1, -1, 0), -1),    # Bille blanche
            ((2, -2, 0), -1),    # Bille blanche
        ]
    )

    # Test 5: Poussée bloquée par une bille amie
    test_scenario(
        "Poussée bloquée par une bille amie",
        [
            ((0, 0, 0), 1),      # Bille noire
            ((1, -1, 0), -1),    # Bille blanche
            ((2, -2, 0), 1),     # Bille noire qui bloque
        ]
    )

if __name__ == "__main__": 
    test_push_rules()