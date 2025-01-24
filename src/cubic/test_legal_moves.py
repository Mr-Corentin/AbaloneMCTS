# test_legal_moves.py
import jax.numpy as jnp
import numpy as np
from core import Direction
from board import initialize_board
from legal_moves import create_player_positions_mask, filter_moves_by_positions, get_legal_moves
from moves import move_group_parallel, move_group_inline

def test_player_positions_mask():
    """Teste la création du masque des positions du joueur"""
    print("=== Test du masque des positions du joueur ===")
    
    # Initialiser le plateau
    board = initialize_board()
    radius = 4
    
    # Créer les masques pour les deux joueurs
    mask_player1 = create_player_positions_mask(board, 1, radius)
    mask_player2 = create_player_positions_mask(board, -1, radius)
    
    # Compter les billes de chaque joueur
    count_player1 = jnp.sum(mask_player1)
    count_player2 = jnp.sum(mask_player2)
    
    print(f"\nNombre de billes du joueur 1: {count_player1}")
    print(f"Nombre de billes du joueur -1: {count_player2}")
    
    # Afficher quelques positions pour vérification
    print("\nQuelques positions avec des billes du joueur 1:")
    positions = jnp.where(mask_player1)
    for i in range(min(5, len(positions[0]))):
        x, y, z = positions[0][i] - radius, positions[1][i] - radius, positions[2][i] - radius
        print(f"Position {i+1}: ({x}, {y}, {z})")

    
    return mask_player1, mask_player2

# test_legal_moves.py
def test_moves_filter():
    """Teste le filtrage des mouvements basé sur les positions des billes"""
    print("\n=== Test du filtrage des mouvements ===")
    
    # Obtenir le board et les masques
    board = initialize_board()
    mask_player1 = create_player_positions_mask(board, 1)
    
    # Charger l'index des mouvements
    moves_data = np.load("move_map.npz")
    moves_jax = {
        'positions': jnp.array(moves_data['positions']),
        'directions': jnp.array(moves_data['directions']),
        'move_types': jnp.array(moves_data['move_types']),
        'group_sizes': jnp.array(moves_data['group_sizes'])
    }
    
    # Filtrer les mouvements
    filtered_moves = filter_moves_by_positions(mask_player1, moves_jax)
    
    # Sauvegarder les résultats détaillés
    with open("filtered_moves_details.txt", "w") as f:
        # Statistiques générales
        single_moves = jnp.sum(filtered_moves & (moves_jax['move_types'] == 0))
        double_moves = jnp.sum(filtered_moves & (moves_jax['move_types'] == 1))
        triple_moves = jnp.sum(filtered_moves & (moves_jax['move_types'] == 2))
        
        f.write("=== Statistiques des mouvements filtrés ===\n")
        f.write(f"Mouvements simples : {single_moves}\n")
        f.write(f"Mouvements doubles : {double_moves}\n")
        f.write(f"Mouvements triples : {triple_moves}\n")
        f.write(f"Total             : {jnp.sum(filtered_moves)}\n\n")
        
        # Détails pour chaque type de mouvement
        valid_indices = jnp.where(filtered_moves)[0]
        
        # Groupes de 2 billes
        f.write("\n=== Détails des mouvements de 2 billes ===\n")
        for idx in valid_indices:
            if moves_jax['move_types'][idx] == 1:  # Double moves
                positions = moves_jax['positions'][idx]
                direction = list(Direction)[moves_jax['directions'][idx]].name
                f.write(f"\nIndex {idx}:\n")
                f.write(f"  Positions: {positions[:2].tolist()}\n")
                f.write(f"  Direction: {direction}\n")
                
        # Groupes de 3 billes
        f.write("\n=== Détails des mouvements de 3 billes ===\n")
        for idx in valid_indices:
            if moves_jax['move_types'][idx] == 2:  # Triple moves
                positions = moves_jax['positions'][idx]
                direction = list(Direction)[moves_jax['directions'][idx]].name
                f.write(f"\nIndex {idx}:\n")
                f.write(f"  Positions: {positions[:3].tolist()}\n")
                f.write(f"  Direction: {direction}\n")
    
    # Afficher les statistiques dans la console
    print("\nMouvements filtrés pour le joueur 1:")
    print(f"Mouvements simples : {single_moves}")
    print(f"Mouvements doubles : {double_moves}")
    print(f"Mouvements triples : {triple_moves}")
    print(f"Total             : {jnp.sum(filtered_moves)}")
    print("\nDétails complets sauvegardés dans 'filtered_moves_details.txt'")
    
    return filtered_moves


def test_moves_filter_debug():
    """Test détaillé des mouvements avec focus sur les groupes de 3 billes"""
    print("\n=== Debug détaillé des mouvements ===")
    
    # Initialiser le plateau et le masque
    board = initialize_board()
    mask_player1 = create_player_positions_mask(board, 1)
    
    # Charger l'index
    moves_data = np.load("move_map.npz")
    moves_jax = {
        'positions': jnp.array(moves_data['positions']),
        'directions': jnp.array(moves_data['directions']),
        'move_types': jnp.array(moves_data['move_types']),
        'group_sizes': jnp.array(moves_data['group_sizes'])
    }
    
    # Debug pour un groupe de 3 billes spécifique
    print("\nAnalyse du groupe [[0, 4, -4], [0, 3, -3], [0, 2, -2]]:")
    target_group = [[0, 4, -4], [0, 3, -3], [0, 2, -2]]
    
    # Trouver tous les mouvements dans l'index qui concernent ce groupe
    print("\nMouvements dans l'index pour ce groupe:")
    for idx in range(len(moves_data['positions'])):
        positions = moves_data['positions'][idx][:3]
        if np.array_equal(positions, target_group):
            move_type = moves_data['move_types'][idx]
            direction = list(Direction)[moves_data['directions'][idx]].name
            print(f"\nIndex {idx}:")
            print(f"  Positions: {positions.tolist()}")
            print(f"  Type: {['SINGLE', 'PARALLEL', 'INLINE'][move_type]}")
            print(f"  Direction: {direction}")
            
            # Vérifier si ce mouvement est détecté comme valide
            board_positions = positions + 4  # radius=4
            pieces_check = mask_player1[board_positions[:, 0],
                                      board_positions[:, 1],
                                      board_positions[:, 2]]
            print(f"  Pièces présentes: {pieces_check.tolist()}")
    
    # Filtrer les mouvements
    filtered_moves = filter_moves_by_positions(mask_player1, moves_jax)
    
    # Analyser les mouvements de groupe détectés
    print("\nMouvements de groupe détectés comme valides:")
    valid_indices = jnp.where(filtered_moves)[0]
    for idx in valid_indices:
        if moves_jax['move_types'][idx] == 2:  # Groupes de 3
            positions = moves_jax['positions'][idx]
            direction = list(Direction)[moves_jax['directions'][idx]].name
            print(f"\nIndex {idx}:")
            print(f"  Positions: {positions[:3].tolist()}")
            print(f"  Direction: {direction}")
    
    return filtered_moves


def test_legal_moves():
    """Test détaillé des mouvements légaux sur la position initiale"""
    print("\n=== Test des mouvements légaux ===")
    
    # Initialiser le plateau
    board = initialize_board()
    
    # Charger l'index
    moves_data = np.load("move_map.npz")
    moves_jax = {
        'positions': jnp.array(moves_data['positions']),
        'directions': jnp.array(moves_data['directions']),
        'move_types': jnp.array(moves_data['move_types']),
        'group_sizes': jnp.array(moves_data['group_sizes'])
    }
    
    # Récupérer les mouvements légaux
    legal_moves = get_legal_moves(board, moves_jax, 1)
    
    # Analyse et sauvegarde des résultats
    with open("legal_moves_initial.txt", "w") as f:
        f.write("=== Mouvements légaux pour le joueur 1 (position initiale) ===\n\n")
        
        # Fonction de comptage détaillé
        def count_moves(group_size, move_type):
            return jnp.sum(legal_moves & 
                         (moves_jax['group_sizes'] == group_size) & 
                         (moves_jax['move_types'] == move_type))
        
        # Statistiques détaillées
        f.write("=== Statistiques détaillées ===\n")
        single_moves = count_moves(1, 0)
        parallel_moves_2 = count_moves(2, 1)
        inline_moves_2 = count_moves(2, 2)
        parallel_moves_3 = count_moves(3, 1)
        inline_moves_3 = count_moves(3, 2)
        
        f.write("\nMouvements simples (1 bille):\n")
        f.write(f"  Total: {single_moves}\n")
        
        f.write("\nMouvements de 2 billes:\n")
        f.write(f"  Parallèles: {parallel_moves_2}\n")
        f.write(f"  En ligne  : {inline_moves_2}\n")
        f.write(f"  Total     : {parallel_moves_2 + inline_moves_2}\n")
        
        f.write("\nMouvements de 3 billes:\n")
        f.write(f"  Parallèles: {parallel_moves_3}\n")
        f.write(f"  En ligne  : {inline_moves_3}\n")
        f.write(f"  Total     : {parallel_moves_3 + inline_moves_3}\n")
        
        f.write(f"\nTotal global: {jnp.sum(legal_moves)}\n")
        
        # Debug des mouvements de 2 billes
        f.write("\n=== Debug des mouvements de 2 billes ===\n")
        f.write("\nMouvements candidats de 2 billes dans l'index:\n")
        for idx in range(len(moves_data['positions'])):
            if moves_data['group_sizes'][idx] == 2:
                positions = moves_data['positions'][idx][:2]
                move_type = "PARALLEL" if moves_data['move_types'][idx] == 1 else "INLINE"
                direction = list(Direction)[moves_data['directions'][idx]].name
                is_legal = bool(legal_moves[idx])
                
                f.write(f"\nIndex {idx}:\n")
                f.write(f"  Positions: {positions.tolist()}\n")
                f.write(f"  Type: {move_type}\n")
                f.write(f"  Direction: {direction}\n")
                f.write(f"  Légal: {is_legal}\n")
                
                # Si on a des billes à ces positions
                board_positions = positions + 4  # radius=4
                has_pieces = board[board_positions[:, 0],
                                 board_positions[:, 1],
                                 board_positions[:, 2]] == 1
                f.write(f"  A nos pièces: {has_pieces.tolist()}\n")
        
        # Détails des mouvements légaux
        f.write("\n=== Mouvements légaux ===\n")
        legal_indices = jnp.where(legal_moves)[0]
        for idx in legal_indices:
            positions = moves_jax['positions'][idx]
            group_size = moves_jax['group_sizes'][idx]
            move_type = "SINGLE" if group_size == 1 else ("PARALLEL" if moves_jax['move_types'][idx] == 1 else "INLINE")
            direction = list(Direction)[moves_jax['directions'][idx]].name
            
            f.write(f"\nIndex {idx}:\n")
            f.write(f"  Positions: {positions[:group_size].tolist()}\n")
            f.write(f"  Type: {move_type}\n")
            f.write(f"  Direction: {direction}\n")
    
    # Afficher le résumé dans la console
    print("\nMouvements légaux trouvés:")
    print(f"Mouvements simples    : {single_moves}")
    print(f"Mouvements 2 billes:")
    print(f"  - Parallèles       : {parallel_moves_2}")
    print(f"  - En ligne         : {inline_moves_2}")
    print(f"Mouvements 3 billes:")
    print(f"  - Parallèles       : {parallel_moves_3}")
    print(f"  - En ligne         : {inline_moves_3}")
    print(f"Total                : {jnp.sum(legal_moves)}")
    print("\nDétails complets et debug sauvegardés dans 'legal_moves_initial.txt'")
    
    return legal_moves


def sort_positions(positions):
    """Trie les positions selon un ordre cohérent"""
    return sorted(positions, key=lambda p: (p[0], p[1], p[2]))

def debug_specific_group():
    """Debug détaillé pour trouver et analyser un groupe spécifique"""
    print("\n=== Debug du groupe [[1,1,-2], [2,0,-2]] avec tri ===")
    
    # Charger l'index et initialiser le plateau
    moves_data = np.load("move_map.npz")
    board = initialize_board()
    moves_jax = {
        'positions': jnp.array(moves_data['positions']),
        'directions': jnp.array(moves_data['directions']),
        'move_types': jnp.array(moves_data['move_types']),
        'group_sizes': jnp.array(moves_data['group_sizes'])
    }
    
    # 1. Rechercher le groupe dans l'index
    target_group_original = [[1,1,-2], [2,0,-2]]
    target_group = sort_positions(target_group_original)
    print(f"\n1. Recherche dans l'index:")
    print(f"Groupe original: {target_group_original}")
    print(f"Groupe trié: {target_group}")
    found_indices = []
    
    for idx in range(len(moves_data['positions'])):
        if moves_data['group_sizes'][idx] == 2:
            positions = sort_positions(moves_data['positions'][idx][:2].tolist())
            if positions == target_group:
                found_indices.append(idx)
                move_type = "PARALLEL" if moves_data['move_types'][idx] == 1 else "INLINE"
                direction = list(Direction)[moves_data['directions'][idx]].name
                print(f"\nTrouvé à l'index {idx}:")
                print(f"  Positions originales: {moves_data['positions'][idx][:2].tolist()}")
                print(f"  Positions triées: {positions}")
                print(f"  Type: {move_type}")
                print(f"  Direction: {direction}")
    
    if not found_indices:
        print("Groupe non trouvé dans l'index!")
        return
    
    # 2. Pour chaque occurrence trouvée, suivre le processus complet
    # print("\n2. Analyse du processus de filtrage:")
    # mask_player1 = create_player_positions_mask(board, 1)
    # filtered_moves = filter_moves_by_positions(mask_player1, moves_jax)
    
    # for idx in found_indices:
    #     print(f"\nAnalyse de l'index {idx}:")
    #     positions = moves_jax['positions'][idx][:2]
    #     direction = moves_jax['directions'][idx]
    #     move_type = moves_jax['move_types'][idx]
        
    #     # Vérifier le filtrage initial
    #     print(f"  Passe le filtrage initial: {bool(filtered_moves[idx])}")
        
    #     # Vérifier si les positions contiennent nos pièces
    #     board_positions = positions + 4  # radius=4
    #     has_pieces = board[board_positions[:, 0],
    #                       board_positions[:, 1],
    #                       board_positions[:, 2]] == 1
    #     print(f"  Positions contiennent nos pièces: {has_pieces.tolist()}")
        
    #     # Test direct du mouvement
    #     if move_type == 1:
    #         _, success = move_group_parallel(board, positions, direction, 4)
    #         print(f"  Test move_group_parallel: {bool(success)}")
    #     else:
    #         _, success, _ = move_group_inline(board, positions, direction, 4)
    #         print(f"  Test move_group_inline: {bool(success)}")
        
    #     # Vérifier le résultat final
    #     legal_moves = get_legal_moves(board, moves_jax, 1)
    #     print(f"  Dans les mouvements légaux finals: {bool(legal_moves[idx])}")

# if __name__ == "__main__":
#     debug_specific_group()


# if __name__ == "__main__":
#     legal_moves = test_legal_moves()

# if __name__ == "__main__":
#     filtered_moves = test_moves_filter_debug()
if __name__ == "__main__":
    mask_player1, mask_player2 = test_player_positions_mask()
    filtered_moves = test_moves_filter()

# if __name__ == "__main__":
#     mask_player1, mask_player2 = test_player_positions_mask()