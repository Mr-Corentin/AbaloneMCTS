import jax
import jax.numpy as jnp
from board import create_custom_board, create_board_mask, display_board
from positions import get_valid_neighbors, analyze_group
# from moves import move_group_inline
from legal_moves import get_legal_moves
from core import Direction
import numpy as np

def test_pushing_scenario():
    # Mappage des indices de direction vers leur nom
    DIRECTION_NAMES = ["NE", "E", "SE", "SW", "W", "NW"]
    MOVE_TYPE_NAMES = ["Simple", "Parallèle", "En ligne"]

    # Créer un plateau avec 2 billes noires contre 1 blanche
    marbles = [
        # ((0, 0, 0), 1),    # Première bille noire
        # ((1, -1, 0), 1),   # Deuxième bille noire
        # ((2, -2, 0), -1)   # Bille blanche
        ((1, -1, 0), 1),    # Première bille noire
        ((0, 0, 0), 1),   # Deuxième bille noire
        ((2, -2, 0), -1)
    ]
    board = create_custom_board(marbles)

    # Charger l'index des mouvements
    moves_data = np.load('move_map.npz')
    moves_index = {
        'positions': moves_data['positions'],
        'directions': moves_data['directions'],
        'move_types': moves_data['move_types'],
        'group_sizes': moves_data['group_sizes']
    }

    # Afficher l'état initial
    print("\nTest de scénario de poussée:")
    print("Deux billes noires alignées contre une bille blanche")
    display_board(board)

    # Obtenir et afficher les mouvements légaux pour le joueur noir (1)
    legal_moves = get_legal_moves(board, moves_index, 1)

    # Afficher les mouvements légaux
    print("\nMouvements légaux pour le joueur noir:")
    for i, is_legal in enumerate(legal_moves):
        if is_legal:
            print("\nMouvement", i)
            print(f"Type: {MOVE_TYPE_NAMES[moves_index['move_types'][i]]}")
            print(f"Direction: {DIRECTION_NAMES[moves_index['directions'][i]]}")
            print(f"Taille du groupe: {moves_index['group_sizes'][i]}")
            print("Positions:", moves_index['positions'][i][:moves_index['group_sizes'][i]])

import jax
import jax.numpy as jnp
from board import create_custom_board, display_board
#from moves import move_group_inline_no_jit #as move_group_inline
from moves_cano import move_group_inline, move_group_parallel

def test_inline_move_old():
    # Même configuration que précédemment
    marbles = [
        ((0, 0, 0), 1),    # Première bille noire
        ((1, -1, 0), 1),   # Deuxième bille noire
        ((2, -2, 0), -1)   # Bille blanche
    ]
    board = create_custom_board(marbles)

    print("État initial:")
    display_board(board)

    # Configuration du test
    positions = jnp.array([
        [0, 0, 0],      # première bille
        [1, -1, 0],     # deuxième bille
        [0, 0, 0]       # padding
    ])
    group_size = 2

    # Debug: analyser le groupe
    is_valid, inline_dirs, parallel_dirs = analyze_group(positions, board, group_size)

    # Calculer le vecteur de différence pour le debug
    diff = positions[1] - positions[0]

    print("\nDébug analyze_group:")
    print(f"Différence entre billes: {diff}")
    print(f"Direction est-ouest: {jnp.all(diff == jnp.array([1, -1, 0]))} ou {jnp.all(diff == jnp.array([-1, 1, 0]))}")
    print(f"Direction nord-est/sud-ouest: {jnp.all(diff == jnp.array([1, 0, -1]))} ou {jnp.all(diff == jnp.array([-1, 0, 1]))}")
    print(f"Direction sud-est/nord-ouest: {jnp.all(diff == jnp.array([0, -1, 1]))} ou {jnp.all(diff == jnp.array([0, 1, -1]))}")
    print(f"Directions inline autorisées: {inline_dirs}")

    # Test direction Est (1)
    print("\nTest mouvement vers l'Est (poussée):")
    new_board_east, success_east, billes_sorties_east = move_group_inline(board, positions, 1, group_size)
    print(f"Succès: {success_east}")
    print(f"Billes sorties: {billes_sorties_east}")
    if success_east:
        print("Nouveau plateau:")
        display_board(new_board_east)

    # Test direction Ouest (4)
    print("\nTest mouvement vers l'Ouest (retrait):")
    new_board_west, success_west, billes_sorties_west = move_group_inline(board, positions, 4, group_size)
    print(f"Succès: {success_west}")
    print(f"Billes sorties: {billes_sorties_west}")
    if success_west:
        print("Nouveau plateau:")
        display_board(new_board_west)

def test_inline_move():
    # Configuration avec 2 billes noires et 1 blanche
    marbles = [
        ((0, 0, 0), 1),    # Première bille noire
        ((1, -1, 0), 1),   # Deuxième bille noire
        ((2, -2, 0), -1)   # Bille blanche
    ]
    board = create_custom_board(marbles)

    positions = jnp.array([
        [0, 0, 0],      # première bille
        [1, -1, 0],     # deuxième bille
        [0, 0, 0]       # padding
    ])
    group_size = 2

    print("\n=== État initial ===")
    display_board(board)

    print("\n=== Test mouvement vers l'Ouest (retrait) ===")
    new_board_west, success_west, billes_sorties_west = move_group_inline(board, positions, 4, group_size)
    if success_west:
        print("\nNouveau plateau (Ouest):")
        display_board(new_board_west)

    print("\n=== Test mouvement vers l'Est (poussée) ===")
    new_board_east, success_east, billes_sorties_east = move_group_inline(board, positions, 1, group_size)
    if success_east:
        print("\nNouveau plateau (Est):")
        display_board(new_board_east)


def test_push_scenarios():
    """Teste différents scénarios de poussée"""
    

    # Scénario 0: Déplacement sans poussée
    print("\n=== Test 0: Déplacement sans poussée ===")
    marbles = [
        ((0, 2, -2), 1),    # Première bille noire
        ((1, 1, -2), 1),   # Deuxième bille noire
        ((2, 0, -2), 1)   # Troisième bille noire
    ]
    test_push_movement(marbles, [[0, 2, -2], [1, 1, -2], [2, 0, -2]], 3, 1)  # direction Est

    # Scénario 1: Poussée normale 
    print("\n=== Test 1: Poussée normale (Est) ===")
    marbles = [
        ((0, 0, 0), 1),    # Première bille noire
        ((1, -1, 0), 1),   # Deuxième bille noire
        ((2, -2, 0), -1)   # Bille blanche
    ]
    test_push_movement(marbles, [[0, 0, 0], [1, -1, 0], [0, 0, 0]], 2, 1)  # direction Est

    # Scénario 2: Poussée avec sortie de plateau
    print("\n=== Test 2: Poussée avec sortie de plateau ===")
    marbles = [
        ((2, -2, 0), 1),    # Première bille noire
        ((3, -3, 0), 1),    # Deuxième bille noire
        ((4, -4, 0), -1)    # Bille blanche (sera poussée hors du plateau)
    ]
    test_push_movement(marbles, [[2, -2, 0], [3, -3, 0], [0, 0, 0]], 2, 1)  # direction Est

    # Scénario 3: Poussée sur l'axe SE-NW
    print("\n=== Test 3: Poussée sur l'axe SE-NW ===")
    marbles = [
        ((0, -2, 2), 1),    # Première bille noire
        ((0, -1, 1), 1),    # Deuxième bille noire
        ((0, 0, 0), -1)     # Bille blanche
    ]
    test_push_movement(marbles, [[0, -2, 2], [0, -1, 1], [0, 0, 0]], 2, 5)  # direction NW

    # Scénario 4: Poussée avec bille amie derrière (doit échouer)
    print("\n=== Test 4: Poussée avec bille amie derrière (devrait échouer) ===")
    marbles = [
        ((0, 0, 0), 1),    # Première bille noire
        ((1, -1, 0), 1),   # Deuxième bille noire
        ((2, -2, 0), -1),  # Bille blanche
        ((3, -3, 0), 1)    # Bille noire derrière (bloque la poussée)
    ]
    test_push_movement(marbles, [[0, 0, 0], [1, -1, 0], [0, 0, 0]], 2, 1)  # direction Est

    # Scénario 5: Poussée 2 vs 2 (doit échouer)
    print("\n=== Test 5: Poussée 2 vs 2 (devrait échouer) ===")
    marbles = [
        ((0, 0, 0), 1),    # Première bille noire
        ((1, -1, 0), 1),   # Deuxième bille noire
        ((2, -2, 0), -1),  # Première bille blanche
        ((3, -3, 0), -1)   # Deuxième bille blanche
    ]
    test_push_movement(marbles, [[0, 0, 0], [1, -1, 0], [0, 0, 0]], 2, 1)  # direction Est

    print("\n=== Test 6: Poussée normale (Est) 3 billes ===")
    marbles = [
        ((1, -1, 0), 1),    # Première bille noire
        ((2, -2, 0), 1),   # Deuxième bille noire
        ((3, -3, 0), 1),   # Troisième bille noire
        ((4,-4,0), -1)     # Bille blanche
    ]
    test_push_movement(marbles, [[1, -1, 0], [2, -2, 0], [3, -3, 0]], 3, 1)  # direction Est

    # Scénario 7: Déplacement sans poussée
    print("\n=== Test 7: Déplacement bloquée par bille ===")
    marbles = [
        ((0, 2, -2), 1),    # Première bille noire
        ((1, 1, -2), 1),   # Deuxième bille noire
        ((2, 0, -2), 1),   # Troisième bille noire
        ((3, -1, -2), 1)
    ]
    test_push_movement(marbles, [[0, 2, -2], [1, 1, -2], [2, 0, -2]], 3, 1)  # direction Est

# def test_push_movement(marbles, positions, group_size, direction):
#     """
#     Fonction utilitaire pour tester un mouvement spécifique

#     Args:
#         marbles: Liste des billes à placer sur le plateau
#         positions: Positions du groupe qui bouge
#         group_size: Taille du groupe
#         direction: Direction du mouvement
#     """
#     board = create_custom_board(marbles)

#     print("État initial:")
#     display_board(board)

#     positions = jnp.array(positions)
#     new_board, success, billes_sorties = move_group_inline(board, positions, direction, group_size)

#     print(f"Succès: {success}")
#     print(f"Billes sorties: {billes_sorties}")
#     if success:
#         print("\nNouveau plateau:")
#         display_board(new_board)

import time
from moves_cano import move_group_inline_old

def test_push_movement(marbles, positions, group_size, direction):
    """
    Fonction utilitaire pour tester un mouvement spécifique avec mesure de temps

    Args:
        marbles: Liste des billes à placer sur le plateau
        positions: Positions du groupe qui bouge
        group_size: Taille du groupe
        direction: Direction du mouvement
    """
    board = create_custom_board(marbles)
    positions = jnp.array(positions)

    print("État initial:")
    display_board(board)

    # Test de la nouvelle version
    start_time = time.time()
    for _ in range(1000):  # Faire plusieurs itérations pour avoir une meilleure mesure
        new_board, success, billes_sorties = move_group_inline(board, positions, direction, group_size)
        # Force l'exécution avec block_until_ready()
        new_board.block_until_ready()
        success.block_until_ready()
        billes_sorties.block_until_ready()
    end_time = time.time()
    new_version_time = end_time - start_time

    # Test de l'ancienne version
    start_time = time.time()
    for _ in range(1000):
        old_board, old_success, old_billes = move_group_inline_old(board, positions, direction, group_size)
        # Force l'exécution
        old_board.block_until_ready()
        old_success.block_until_ready()
        old_billes.block_until_ready()
    end_time = time.time()
    old_version_time = end_time - start_time

    print(f"\nComparaison des temps d'exécution (100 itérations):")
    print(f"Nouvelle version: {new_version_time:.6f} secondes")
    print(f"Ancienne version: {old_version_time:.6f} secondes")
    print(f"Amélioration: {((old_version_time - new_version_time) / old_version_time) * 100:.4f}%")

    print(f"\nRésultats nouvelle version:")
    print(f"Succès: {success}")
    print(f"Billes sorties: {billes_sorties}")
    print("\nNouveau plateau:")
    display_board(new_board)
    # if success:
    #     print("\nNouveau plateau:")
    #     display_board(new_board)

    # Vérifier que les résultats sont identiques
    old_board, old_success, old_billes = move_group_inline_old(board, positions, direction, group_size)
    results_match = (
        jnp.array_equal(new_board, old_board) and 
        success == old_success and 
        billes_sorties == old_billes
    )
    print(f"\nRésultats identiques: {results_match}")

# if __name__ == "__main__":
#     test_push_scenarios()

def test_performance():
    import time

    # Préparer les données de test
    marbles = [
        ((0, 0, 0), 1),    # Première bille noire
        ((1, -1, 0), 1),   # Deuxième bille noire
        ((2, -2, 0), -1)   # Bille blanche
    ]
    board = create_custom_board(marbles)
    positions = jnp.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    group_size = 2
    direction = 1  # Est

    # Test version non-JIT
    n_iterations = 1000
    start_time = time.time()
    for _ in range(n_iterations):
        move_group_inline_no_jit(board, positions, direction, group_size)
    no_jit_time = time.time() - start_time

    # Test version JIT
    # Première exécution pour compiler
    move_group_inline(board, positions, direction, group_size)

    # Mesure du temps après compilation
    start_time = time.time()
    for _ in range(n_iterations):
        move_group_inline(board, positions, direction, group_size)
    jit_time = time.time() - start_time

    print(f"\nTest de performance sur {n_iterations} itérations:")
    print(f"Version sans JIT: {no_jit_time:.4f} secondes")
    print(f"Version avec JIT: {jit_time:.4f} secondes")
    print(f"Accélération: {no_jit_time/jit_time:.2f}x")

from board import initialize_board
from legal_moves_cano import get_legal_moves as get_legal_moves_cano
def test_initial_legal_moves():
    # Créer le plateau initial
    board = initialize_board()
    print("\nPlateau initial:")
    display_board(board)

    # Charger l'index des mouvements
    moves_data = np.load('move_map.npz')
    moves_index = {
        'positions': moves_data['positions'],
        'directions': moves_data['directions'],
        'move_types': moves_data['move_types'],
        'group_sizes': moves_data['group_sizes']
    }

    # Obtenir les mouvements légaux pour le joueur noir (1)
    legal_moves = get_legal_moves_cano(board, moves_index)

    # Compter les mouvements par taille de groupe
    single_moves = jnp.sum(legal_moves & (moves_index['group_sizes'] == 1))
    double_moves = jnp.sum(legal_moves & (moves_index['group_sizes'] == 2))
    triple_moves = jnp.sum(legal_moves & (moves_index['group_sizes'] == 3))
    total_moves = jnp.sum(legal_moves)

    # Sauvegarder les résultats dans un fichier texte
    with open('initial_legal_moves.txt', 'w') as f:
        f.write("Mouvements légaux pour le joueur noir à partir de la position initiale:\n\n")
        f.write(f"Mouvements avec 1 bille: {single_moves}\n")
        f.write(f"Mouvements avec 2 billes: {double_moves}\n")
        f.write(f"Mouvements avec 3 billes: {triple_moves}\n")
        f.write(f"\nNombre total de mouvements légaux: {total_moves}\n")

        # Sauvegarder les détails de chaque mouvement
        f.write("\nDétail des mouvements:\n")
        for i in range(len(legal_moves)):
            if legal_moves[i]:
                f.write(f"\nMouvement {i+1}:\n")
                f.write(f"Taille du groupe: {moves_index['group_sizes'][i]}\n")
                f.write(f"Positions: {moves_index['positions'][i]}\n")
                f.write(f"Direction: {moves_index['directions'][i]}\n")
                f.write(f"Type: {moves_index['move_types'][i]}\n")

    # Afficher les résultats
    print("\nNombre de mouvements légaux pour le joueur noir:")
    print(f"1 bille: {single_moves} mouvements")
    print(f"2 billes: {double_moves} mouvements")
    print(f"3 billes: {triple_moves} mouvements")
    print(f"\nTotal: {total_moves} mouvements")
    print("\nLes détails ont été sauvegardés dans 'initial_legal_moves.txt'")


# Exemple de test
def test_inline_order():
    # Créer un plateau simple avec 2 billes alignées
    marbles = [
        ((0, 2, -2), 1),
        ((0, 3, -3), 1),
    ]
    board = create_custom_board(marbles)
    display_board(board)

    # Tester les deux ordres possibles
    positions1 = jnp.array([[0, 2, -2], [0, 3, -3], [0, 0, 0]])  # ordre 1
    positions2 = jnp.array([[0, 3, -3], [0, 2, -2], [0, 0, 0]])  # ordre 2

    # Tester les mouvements dans différentes directions
    directions = [0, 1,2,3,4,5]  # Est et Ouest
    for direction in directions:
        print(f"\nDirection: {direction}")
        print("Test ordre 1:")
        new_board1, success1, _ = move_group_inline(board, positions1, direction, 2)
        print(f"Succès: {success1}")
        if success1:
            display_board(new_board1)

        print("\nTest ordre 2:")
        new_board2, success2, _ = move_group_inline(board, positions2, direction, 2)
        print(f"Succès: {success2}")
        if success2:
            display_board(new_board2)


def test_inline_order():
    # Créer un plateau simple avec 2 billes alignées
    marbles = [
        ((0, 0, 0), 1),
        ((1, -1, 0), 1),
        ((2,-2,0), -1)
    ]
    board = create_custom_board(marbles)
    display_board(board)

    # Tester les deux ordres possibles
    positions1 = jnp.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])  # ordre 1
    # positions2 = jnp.array([[1, -1, 0], [0, 0, 0], [0, 0, 0]])  # ordre 2

    # Tester les mouvements dans différentes directions
    directions = [1,4]  # Est et Ouest
    for direction in directions:
        print(f"\nDirection: {direction}")
        print("Test ordre 1:")
        new_board1, success1, _ = move_group_inline(board, positions1, direction, 2)
        print(f"Succès: {success1}")
        if success1:
            display_board(new_board1)
            
        # print("\nTest ordre 2:")
        # new_board2, success2, _ = move_group_inline_no_jit(board, positions2, direction, 2)
        # print(f"Succès: {success2}")
        # if success2:
        #     display_board(new_board2)

def test_simple_moves():
    # Configuration initiale avec 2 billes alignées
    marbles = [
        ((0, 2, -2), 1),    # Première bille noire
        ((1, 1, -2), 1),   # Deuxième bille noire
        ((2, 0, -2), 1)
    ]
    #board = create_custom_board(marbles)
    board = initialize_board()
    
    print("État initial:")
    display_board(board)
    
    positions = jnp.array([
        [0, 2, -2],      # première bille
        [1, 1, -2],     # deuxième bille
        [2, 0, -2]       # padding
    ])
    group_size = 3

    positions = jnp.array([
        [2, 0, -2],      # première bille
        [1, 1, -2],     # deuxième bille
        [0, 2, -2]       # padding
    ])
    group_size = 3

    # Test mouvement vers l'ouest
    print("\nTest mouvement vers l'Ouest:")
    new_board_west, success_west, _ = move_group_inline(board, positions, 4, group_size)
    print(f"Succès: {success_west}")
    if success_west:
        display_board(new_board_west)
    print("\nTest mouvement vers l'Est:")
    new_board_west, success_west, _ = move_group_inline(board, positions, 1, group_size)
    print(f"Succès: {success_west}")
    if success_west:
        display_board(new_board_west)


def test_collision_amie():
    # Créer un plateau avec des billes alignées et une bille amie sur le chemin
    marbles = [
        ((1, 2, -3), 1),    # Première bille du groupe
        ((2, 1, -3), 1),    # Deuxième bille du groupe
        ((3, 0, -3), 1),    # Troisième bille du groupe
        ((4, -1, -3), 1),   # Bille amie qui bloque
    ]
    board = create_custom_board(marbles)
    
    print("\nÉtat initial:")
    display_board(board)
    
    positions = jnp.array([
        [1, 2, -3],     # première bille
        [2, 1, -3],     # deuxième bille
        [3, 0, -3]      # troisième bille
    ])
    group_size = 3
    
    # Test mouvement vers l'est (devrait échouer)
    print("\nTest mouvement vers l'Est (devrait échouer):")
    new_board_east, success_east, _ = move_group_inline(board, positions, 1, group_size)
    print(f"Succès: {success_east}")

    #second test
    marbles = [
        ((2, 2, -4), 1),    # Première bille du groupe
        ((1, 3, -4), 1),    # Deuxième bille du groupe
        ((3, 1, -4), 1),    # Troisième bille du groupe
        ((0, 4, -4), 1),   # Bille amie qui bloque
    ]
    board = create_custom_board(marbles)
    
    print("\nÉtat initial:")
    display_board(board)
    
    positions = jnp.array([
        [1, 3, -4],     # première bille
        [2, 2, -4],     # deuxième bille
        [0, 0, 0]      # troisième bille
    ])
    group_size = 2
    
    # Test mouvement vers l'est (devrait échouer)
    print("\nTest mouvement vers l'Est (devrait échouer):")
    new_board_east, success_east, _ = move_group_inline(board, positions, 1, group_size)
    print(f"Succès: {success_east}")
    display_board(new_board_east)
    print("\nTest mouvement vers l'Ouest (devrait échouer):")
    new_board_west, success_west, _ = move_group_inline(board, positions, 4, group_size)
    print(f"Succès: {success_west}")
    display_board(new_board_west)

import jax.numpy as jnp

def test_head_positions():
    """
    Test différents cas de groupes de billes pour vérifier la détection de head_position
    """
    # Test cases avec différentes configurations
    test_cases = [
        # Groupe de 2 aligné E-W
        {
            "positions": jnp.array([[1, 3, -4], [2, 2, -4], [0, 0, 0]]),
            "group_size": 2,
            "directions": [1, 4],  # Test E puis W
            "expected_heads": [[2, 2, -4], [1, 3, -4]]  # [pour E, pour W]
        },
        # Groupe de 2 aligné NE-SW
        {
            "positions": jnp.array([[2, 0, -2], [3, 0, -3], [0, 0, 0]]),
            "group_size": 2,
            "directions": [0, 3],  # Test NE puis SW
            "expected_heads": [[3, 0, -3], [2, 0, -2]]
        },
        # Groupe de 3 aligné E-W
        {
            "positions": jnp.array([[1, 3, -4], [2, 2, -4], [3, 1, -4]]),
            "group_size": 3,
            "directions": [1, 4],  # Test E puis W
            "expected_heads": [[3, 1, -4], [1, 3, -4]]
        },
        # Groupe de 3 aligné SE-NW
        {
            "positions": jnp.array([[2, 0, -2], [2, 1, -3], [2, 2, -4]]),
            "group_size": 3,
            "directions": [2, 5],  # Test SE puis NW
            "expected_heads": [[2, 0, -2], [2, 2, -4]]
        }
    ]

    # Fonction qui implémente la logique de head_position
    def get_head_position(positions, positions_mask, direction):
        scores = jnp.where(
            direction == 1,  # Est (x+1,y-1,z) : priorité x puis -y
            positions[:, 0] * 100 - positions[:, 1],
            jnp.where(
                direction == 4,  # Ouest (x-1,y+1,z) : priorité -x puis y
                -positions[:, 0] * 100 + positions[:, 1],
                jnp.where(
                    direction == 0,  # Nord-Est (x+1,y,z-1) : priorité x puis -z
                    positions[:, 0] * 100 - positions[:, 2],
                    jnp.where(
                        direction == 3,  # Sud-Ouest (x-1,y,z+1) : priorité -x puis z
                        -positions[:, 0] * 100 + positions[:, 2],
                        jnp.where(
                            direction == 2,  # Sud-Est (x,y-1,z+1) : priorité -y puis z
                            -positions[:, 1] * 100 + positions[:, 2],
                            # Nord-Ouest (x,y+1,z-1) : priorité y puis -z
                            positions[:, 1] * 100 - positions[:, 2]
                        )
                    )
                )
            )
        )

        # scores = jnp.where(
        #     direction == 1,  # Est
        #     positions[:, 0] * 100 - positions[:, 1],
        #     jnp.where(
        #         direction == 4,  # Ouest
        #         -positions[:, 0] * 100 + positions[:, 1],
        #         jnp.where(
        #             direction == 0,  # Nord-Est
        #             positions[:, 1] * 100 - positions[:, 2],
        #             jnp.where(
        #                 direction == 3,  # Sud-Ouest
        #                 -positions[:, 1] * 100 + positions[:, 2],
        #                 jnp.where(
        #                     direction == 2,  # Sud-Est
        #                     -positions[:, 2] * 100 + positions[:, 0],
        #                     # Nord-Ouest (direction == 5)
        #                     positions[:, 2] * 100 - positions[:, 0]
        #                 )
        #             )
        #         )
        #     )
        # )
        scores = jnp.where(positions_mask, scores, -jnp.inf)
        head_index = jnp.argmax(scores)
        return positions[head_index]

    # Exécuter les tests
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i + 1}:")
        positions = test_case["positions"]
        group_size = test_case["group_size"]
        direction_taken = test_case['directions']
        positions_mask = jnp.arange(positions.shape[0]) < group_size
        
        print(f"Groupe de {group_size} billes:")
        print("Positions:", positions[positions_mask])
        
        for j, direction in enumerate(test_case["directions"]):
            head_pos = get_head_position(positions, positions_mask, direction)
            expected = jnp.array(test_case["expected_heads"][j])
            print(f"\nDirection {direction}:")
            print("Head position calculée:", head_pos)
            print("Head position attendue:", expected)
            print("Correct:", jnp.array_equal(head_pos, expected))
from legal_moves import debug_specific_moves
def test_debug_specific_moves():
    # Charger l'index des mouvements
    moves_data = np.load('move_map.npz')
    moves_index = {
        'positions': moves_data['positions'],
        'directions': moves_data['directions'],
        'move_types': moves_data['move_types'],
        'group_sizes': moves_data['group_sizes']
    }

    # Créer un plateau de test
    board = initialize_board()
    
    # Débugger nos mouvements spécifiques
    print("\nDébugging des mouvements spécifiques:")
    results = debug_specific_moves(board, moves_index, 1, [1425, 1428])
    
    # Afficher un résumé final
    print("\nRésumé final:")
    for move_idx, is_legal in results.items():
        print(f"Mouvement {move_idx} : {'légal' if is_legal else 'illégal'}")


def test_custom_board_moves():
    # Définir les positions des billes
    marbles = [
        # Groupe de 3 billes noires alignées
        ((0, 2, -2), 1),  # Première bille noire
        ((1, 1, -2), 1),  # Deuxième bille noire
        ((2, 0, -2), 1),  # Troisième bille noire
        
        # Billes blanches à pousser
        ((3, -1, -2), -1),  # Bille blanche à pousser
        ((4, -2, -2), -1),  # Bille blanche au bord
        
        # Ajoutons peut-être un autre groupe pour tester d'autres situations
        ((0, 0, 0), 1),   # Bille noire isolée
        ((1, 0, -1), -1)  # Bille blanche adjacente
    ]
    
    # Créer le plateau
    board = create_custom_board(marbles)
    
    print("\nTest sur plateau personnalisé:")
    display_board(board)
    
    # Charger l'index des mouvements
    moves_data = np.load('move_map.npz')
    moves_index = {
        'positions': moves_data['positions'],
        'directions': moves_data['directions'],
        'move_types': moves_data['move_types'],
        'group_sizes': moves_data['group_sizes']
    }
    
    legal_moves = get_legal_moves(board, moves_index, 1)

    # Compter les mouvements par taille de groupe
    single_moves = jnp.sum(legal_moves & (moves_index['group_sizes'] == 1))
    double_moves = jnp.sum(legal_moves & (moves_index['group_sizes'] == 2))
    triple_moves = jnp.sum(legal_moves & (moves_index['group_sizes'] == 3))
    total_moves = jnp.sum(legal_moves)

    # Sauvegarder les résultats dans un fichier texte
    with open('initial_legal_moves_2.txt', 'w') as f:
        f.write("Mouvements légaux pour le joueur noir à partir de la position initiale:\n\n")
        f.write(f"Mouvements avec 1 bille: {single_moves}\n")
        f.write(f"Mouvements avec 2 billes: {double_moves}\n")
        f.write(f"Mouvements avec 3 billes: {triple_moves}\n")
        f.write(f"\nNombre total de mouvements légaux: {total_moves}\n")

        # Sauvegarder les détails de chaque mouvement
        f.write("\nDétail des mouvements:\n")
        for i in range(len(legal_moves)):
            if legal_moves[i]:
                f.write(f"\nMouvement {i+1}:\n")
                f.write(f"Taille du groupe: {moves_index['group_sizes'][i]}\n")
                f.write(f"Positions: {moves_index['positions'][i]}\n")
                f.write(f"Direction: {moves_index['directions'][i]}\n")
                f.write(f"Type: {moves_index['move_types'][i]}\n")

    # Afficher les résultats
    print("\nNombre de mouvements légaux pour le joueur noir:")
    print(f"1 bille: {single_moves} mouvements")
    print(f"2 billes: {double_moves} mouvements")
    print(f"3 billes: {triple_moves} mouvements")
    print(f"\nTotal: {total_moves} mouvements")
    print("\nLes détails ont été sauvegardés dans 'initial_legal_moves_2.txt'")
from legal_moves_cano import get_legal_moves as get_legal_moves_cano
def test_custom_board_moves_cano():
    # Définir les positions des billes
    # Note : On garde les mêmes valeurs car dans create_custom_board
    # 1 représente déjà le joueur courant
    marbles = [
        # Groupe de 3 billes du joueur courant alignées
        ((0, 2, -2), 1),  
        ((1, 1, -2), 1),  
        ((2, 0, -2), 1),  
        
        # Billes adverses à pousser
        ((3, -1, -2), -1),  
        ((4, -2, -2), -1),  
        
        # Autres situations
        ((0, 0, 0), 1),   
        ((1, 0, -1), -1)  
    ]
    
    # Créer le plateau
    board = create_custom_board(marbles)
    #board = initialize_board()
    
    print("\nTest sur plateau personnalisé:")
    display_board(board)
    
    # Charger l'index des mouvements
    moves_data = np.load('move_map.npz')
    moves_index = {
        'positions': moves_data['positions'],
        'directions': moves_data['directions'],
        'move_types': moves_data['move_types'],
        'group_sizes': moves_data['group_sizes']
    }
    
    # Plus besoin de spécifier le joueur
    legal_moves = get_legal_moves_cano(board, moves_index)
    single_moves = jnp.sum(legal_moves & (moves_index['group_sizes'] == 1))
    double_moves = jnp.sum(legal_moves & (moves_index['group_sizes'] == 2))
    triple_moves = jnp.sum(legal_moves & (moves_index['group_sizes'] == 3))
    total_moves = jnp.sum(legal_moves)

    # Sauvegarder les résultats dans un fichier texte
    with open('initial_legal_moves_cano.txt', 'w') as f:
        f.write("Mouvements légaux pour le joueur noir à partir de la position initiale:\n\n")
        f.write(f"Mouvements avec 1 bille: {single_moves}\n")
        f.write(f"Mouvements avec 2 billes: {double_moves}\n")
        f.write(f"Mouvements avec 3 billes: {triple_moves}\n")
        f.write(f"\nNombre total de mouvements légaux: {total_moves}\n")

        # Sauvegarder les détails de chaque mouvement
        f.write("\nDétail des mouvements:\n")
        for i in range(len(legal_moves)):
            if legal_moves[i]:
                f.write(f"\nMouvement {i+1}:\n")
                f.write(f"Taille du groupe: {moves_index['group_sizes'][i]}\n")
                f.write(f"Positions: {moves_index['positions'][i]}\n")
                f.write(f"Direction: {moves_index['directions'][i]}\n")
                f.write(f"Type: {moves_index['move_types'][i]}\n")

    # Afficher les résultats
    print("\nNombre de mouvements légaux pour le joueur noir:")
    print(f"1 bille: {single_moves} mouvements")
    print(f"2 billes: {double_moves} mouvements")
    print(f"3 billes: {triple_moves} mouvements")
    print(f"\nTotal: {total_moves} mouvements")
    print("\nLes détails ont été sauvegardés dans 'initial_legal_moves_cano.txt'")

def test_debugs_moves():

    #board = create_custom_board(marbles)
    marbles = [
        # Groupe de 3 billes noires alignées
        ((0, 2, -2), 1),  # Première bille noire
        ((1, 1, -2), 1),  # Deuxième bille noire
        ((2, 0, -2), 1),  # Troisième bille noire
        
        # Billes blanches à pousser
        ((3, -1, -2), -1),  # Bille blanche à pousser
        ((4, -2, -2), -1),  # Bille blanche au bord
        
        # Ajoutons peut-être un autre groupe pour tester d'autres situations
        ((0, 0, 0), 1),   # Bille noire isolée
        ((1, 0, -1), -1)  # Bille blanche adjacente
    ]
    
    # Créer le plateau
    board = create_custom_board(marbles)
    
    
    print("État initial:")
    display_board(board)
    
    positions = jnp.array([
        [0, 2, -2],      # première bille
        [1, 1, -2],     # deuxième bille
        [0, 0, 0]       # padding
    ])
    group_size = 2


    print("\nTest mouvement vers NE:")
    new_board_west, success_west= move_group_parallel(board, positions, 0, group_size)
    print(f"Succès: {success_west}")
    if success_west:
        display_board(new_board_west)

if __name__ == "__main__":
    #test_debug_specific_moves()
    #test_initial_legal_moves()
   # test_custom_board_moves_cano()

    test_initial_legal_moves()
    test_push_scenarios()
    #test_debugs_moves()


# if __name__ == "__main__":
#     test_head_positions()

# if __name__ == "__main__":
#     test_simple_moves()

# if __name__ == "__main__":
#     #test_inline_order()
#     # test_performance()
#     test_initial_legal_moves()
#     #test_simple_moves()
#     #test_collision_amie()
#    # test_push_scenarios()



# if __name__ == "__main__":
#      test_push_scenarios()
# if __name__ == "__main__":
#     test_initial_legal_moves()


# if __name__ == "__main__":
#     test_performance()