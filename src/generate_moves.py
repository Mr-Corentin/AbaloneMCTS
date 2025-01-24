# src/generate_moves.py
import jax.numpy as jnp
from mcts_utils import get_neighbors, get_cell_content_jax, initialiser_plateau, get_all_valid_groups_jax

def generate_all_single_moves():
    """
    Génère tous les mouvements possibles pour une seule bille
    """
    moves = set()
    cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
    
    for y, n_cells in enumerate(cells_per_row):
        for x in range(n_cells):
            board = jnp.full((9, 9), jnp.nan)
            for i, n in enumerate(cells_per_row):
                for j in range(n):
                    board = board.at[i, j].set(0)
            board = board.at[y, x].set(1)
            
            neighbors = get_neighbors(board, x, y)
            for direction in neighbors.keys():
                moves.add(((x, y), direction))
    
    print(f"Nombre de mouvements à une bille : {len(moves)}")
    return moves

def generate_all_two_marble_moves():
    """
    Génère tous les mouvements possibles pour deux billes adjacentes
    """
    moves = set()
    cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
    
    for y1, n_cells in enumerate(cells_per_row):
        for x1 in range(n_cells):
            board = jnp.full((9, 9), jnp.nan)
            for i, n in enumerate(cells_per_row):
                for j in range(n):
                    board = board.at[i, j].set(0)
            board = board.at[y1, x1].set(1)
            
            neighbors = get_neighbors(board, x1, y1)
            for (x2, y2) in neighbors.values():
                if (x2, y2) > (x1, y1):  # Ne traiter que si la deuxième coordonnée est "plus grande"
                    board = board.at[y2, x2].set(1)
                    group = tuple(sorted([(x1, y1), (x2, y2)]))  # Trier les coordonnées
                    
                    for direction in ['NW', 'NE', 'E', 'SE', 'SW', 'W']:
                        moves.add((group, direction))
                    
                    board = board.at[y2, x2].set(0)
    
    print(f"Nombre de mouvements à deux billes : {len(moves)}")
    return moves

def generate_all_three_marble_moves():
    """
    Génère tous les mouvements possibles pour trois billes adjacentes alignées
    """
    moves = set()
    cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
    
    for y1, n_cells in enumerate(cells_per_row):
        for x1 in range(n_cells):
            board = jnp.full((9, 9), jnp.nan)
            for i, n in enumerate(cells_per_row):
                for j in range(n):
                    board = board.at[i, j].set(0)
            board = board.at[y1, x1].set(1)
            
            neighbors1 = get_neighbors(board, x1, y1)
            for direction1, (x2, y2) in neighbors1.items():
                if (x2, y2) > (x1, y1):  # Vérifier l'ordre
                    board = board.at[y2, x2].set(1)
                    
                    neighbors2 = get_neighbors(board, x2, y2)
                    for direction2, (x3, y3) in neighbors2.items():
                        if direction2 == direction1 and (x3, y3) > (x2, y2):  # Vérifier l'alignement et l'ordre
                            board = board.at[y3, x3].set(1)
                            
                            group = tuple(sorted([(x1, y1), (x2, y2), (x3, y3)]))  # Trier les coordonnées
                            for direction in ['NW', 'NE', 'E', 'SE', 'SW', 'W']:
                                moves.add((group, direction))
                            
                            board = board.at[y3, x3].set(0)
                    board = board.at[y2, x2].set(0)
    
    print(f"Nombre de mouvements à trois billes : {len(moves)}")
    return moves

def main():
    single_moves = generate_all_single_moves()
    print("\nExemples de mouvements à une bille:")
    for move in list(single_moves)[:5]:
        print(f"Position: {move[0]}, Direction: {move[1]}")
    two_marble_moves = generate_all_two_marble_moves()
    
    print("\nExemples de mouvements à deux billes:")
    for move in list(two_marble_moves)[:5]:
        print(f"Groupe: {move[0]}, Direction: {move[1]}")

    three_marble_moves = generate_all_three_marble_moves()
    
    print("\nExemples de mouvements à trois billes:")
    for move in list(three_marble_moves)[:5]:
        print(f"Groupe: {move[0]}, Direction: {move[1]}")
    

if __name__ == "__main__":
    main()