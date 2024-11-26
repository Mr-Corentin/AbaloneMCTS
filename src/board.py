# board.py
import numpy as np

def create_board():
    """
    Crée et initialise le plateau de jeu d'Abalone
    Returns:
        numpy.array: Le plateau initialisé avec les billes
    """
    board = np.full((9, 9), None)
    cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
    
    for y, n_cells in enumerate(cells_per_row):
        for x in range(n_cells):
            board[y][x] = 0
    
    # Placement des billes blanches (en haut)
    for x in range(5):  # Ligne 0
        board[0][x] = 1
    for x in range(6):  # Ligne 1
        board[1][x] = 1
    # Ligne 2 : 2 espaces, 3 blanches, 2 espaces
    for x in range(2, 5):
        board[2][x] = 1
        
    # Placement des billes noires (en bas)
    for x in range(5):  # Ligne 8
        board[8][x] = -1
    for x in range(6):  # Ligne 7
        board[7][x] = -1
    # Ligne 6 : 2 espaces, 3 noires, 2 espaces
    for x in range(2, 5):
        board[6][x] = -1
    
    return board

def display_board(board):
    """
    Affiche le plateau de jeu
    Args:
        board: Le plateau à afficher
    """
    cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
    
    for y, n_cells in enumerate(cells_per_row):
        # Calculer l'indentation pour cette ligne
        indent = abs(4 - y)
        
        print(" " * indent, end="")
        
        for x in range(n_cells):
            if board[y][x] == 0:
                print("O ", end="")  # Case vide
            elif board[y][x] == 1:
                print("W ", end="")  # Bille blanche (white)
            elif board[y][x] == -1:
                print("B ", end="")  # Bille noire (black)
        print()

def get_cell_content(board, x, y):
    """
    Retourne le contenu d'une case du plateau
    Args:
        board: Le plateau de jeu
        x, y: Coordonnées de la case
    Returns:
        str: 'W' pour une bille blanche, 'B' pour une bille noire, 'O' pour une case vide
             ou un message d'erreur si les coordonnées sont invalides
    """
    # Vérifier si les coordonnées sont dans les limites du plateau
    if x < 0 or y < 0 or x > 8 or y > 8:
        return "Invalid coordinates: out of bounds"
    
    # Vérifier si la case existe pour cette ligne (selon le nombre de cases par ligne)
    cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
    if x >= cells_per_row[y]:
        return "Invalid coordinates: cell doesn't exist"
    
    if board[y][x] == 1:
        return "W"
    elif board[y][x] == -1:
        return "B"
    elif board[y][x] == 0:
        return "O"
    else:
        return "Invalid cell"