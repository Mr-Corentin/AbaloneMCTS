# src/jax_board.py
import jax
import jax.numpy as jnp

class JaxAbaloneBoard:
    def __init__(self):
        self.cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
        # On initialise avec None pour les cases invalides
        # Note: avec JAX on pourrait utiliser float('nan') au lieu de None
        self.board = self.initialize_board()
        self.current_player = 1  # 1 pour noir (commence), -1 pour blanc
    
    def initialize_board(self):
        # Initialiser le plateau avec des NaN
        board = jnp.full((9, 9), float('nan'))
        
        # Remplir les cases valides avec 0 (vide)
        def fill_valid_cells(board):
            for y, n_cells in enumerate(self.cells_per_row):
                for x in range(n_cells):
                    board = board.at[y, x].set(0)
            return board
        
        # Placer les billes noires (en haut, valeur 1)
        def place_black_marbles(board):
            # Ligne 0
            board = board.at[0, :5].set(1)
            # Ligne 1
            board = board.at[1, :6].set(1)
            # Ligne 2 (3 billes)
            board = board.at[2, 2:5].set(1)
            return board
            
        # Placer les billes blanches (en bas, valeur -1)
        def place_white_marbles(board):
            # Ligne 8
            board = board.at[8, :5].set(-1)
            # Ligne 7
            board = board.at[7, :6].set(-1)
            # Ligne 6 (3 billes)
            board = board.at[6, 2:5].set(-1)
            return board
            
        # Appliquer les transformations
        board = fill_valid_cells(board)
        board = place_black_marbles(board)
        board = place_white_marbles(board)
        
        return board
    
    def display(self):
        """Affiche le plateau"""
        for y, n_cells in enumerate(self.cells_per_row):
            # Indentation
            print(" " * abs(4 - y), end="")
            
            for x in range(n_cells):
                cell = self.board[y, x]
                if jnp.isnan(cell):
                    continue
                elif cell == 0:
                    print("O ", end="")
                elif cell == 1:
                    print("B ", end="")  # Noir
                elif cell == -1:
                    print("W ", end="")  # Blanc
            print()

    def get_canonical_form(self):
        """Retourne la forme canonique du plateau"""
        return self.board * self.current_player
    
# # Test dans votre terminal ou notebook
# if __name__ == "__main__":
#     # Créer une instance du plateau
#     board = JaxAbaloneBoard()
    
#     print("Plateau initial:")
#     board.display()
    
#     print("\nForme canonique (du point de vue du joueur actuel):")
#     canonical_board = board.get_canonical_form()
#     print(canonical_board)