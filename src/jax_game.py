# src/jax_game.py
import jax
import jax.numpy as jnp
from .jax_board import JaxAbaloneBoard
from .jax_utils import get_cell_content_jax, is_valid_group, get_neighbors
from .jax_moves import push_marbles, make_move
from itertools import combinations

class JaxAbaloneGame:
    def __init__(self):
        self.board = JaxAbaloneBoard().board  # On récupère juste le plateau
        self.current_player = 1  # 1 (noir) commence, -1 pour blanc
        self.white_marbles_out = 0
        self.black_marbles_out = 0
    
    def switch_player(self):
        self.current_player = -self.current_player
    
    def get_current_player_name(self):
        return "Noir" if self.current_player == 1 else "Blanc"
    
    def is_game_over(self):
        """Vérifie si un joueur a gagné"""
        if self.white_marbles_out >= 6:
            return True, "Les Noirs ont gagné!"
        if self.black_marbles_out >= 6:
            return True, "Les Blancs ont gagné!"
        return False, None
    
    def check_move_ownership(self, coordinates):
        """Vérifie si les billes appartiennent au joueur actuel"""
        for x, y in coordinates:
            content = get_cell_content_jax(self.board, x, y)
            if content != self.current_player:
                return False
        return True
    
    def get_canonical_form(self):
        """Retourne la forme canonique du plateau"""
        return self.board * self.current_player
    # def make_move(self, coordinates, direction):
    #    """
    #    Effectue un mouvement (déplacement ou poussée)
    #    Returns:
    #        tuple: (succès, message)
    #    """
    #    # Vérifier si c'est game over
    #    is_over, message = self.is_game_over()
    #    if is_over:
    #        return False, message
       
    #    # Vérifier que les billes appartiennent au bon joueur
    #    if not self.check_move_ownership(coordinates):
    #        return False, "Ces billes ne vous appartiennent pas"
       
    #    # Essayer d'abord une poussée
    #    new_board, success, message, billes_sorties = push_marbles(self.board, coordinates, direction)
    #    if success:
    #        self.board = new_board
    #        # Mettre à jour le compte des billes sorties
    #        if self.current_player == 1:
    #            self.white_marbles_out += billes_sorties
    #        else:
    #            self.black_marbles_out += billes_sorties
           
    #        # Vérifier si quelqu'un a gagné
    #        is_over, win_message = self.is_game_over()
    #        if is_over:
    #            return True, f"{message} {win_message}"
           
    #        self.switch_player()
    #        return True, message
       
    #    # Si ce n'est pas une poussée valide, essayer un déplacement normal
    #    if len(coordinates) <= 3:
    #        new_board, success, message = make_move(self.board, coordinates, direction)
    #        if success:
    #            self.board = new_board
    #            self.switch_player()
    #            return True, message
       
    #    return False, "Mouvement invalide"

    def make_move(self, coordinates, direction):
        """
        Effectue un mouvement (déplacement ou poussée)
        Returns:
            tuple: (succès, message)
        """
        # Vérifier si c'est game over
        is_over, message = self.is_game_over()
        if is_over:
            return False, message

        # Vérifier que les billes appartiennent au bon joueur
        if not self.check_move_ownership(coordinates):
            return False, "Ces billes ne vous appartiennent pas"

        # Essayer d'abord une poussée
        new_board, success, message, billes_sorties = push_marbles(self.board, coordinates, direction)
        if success:
            self.board = new_board
            # Mettre à jour le compte des billes sorties
            if self.current_player == 1:  # Noir joue
                self.black_marbles_out += billes_sorties  # Bille blanche sortie
            else:  # Blanc joue
                self.white_marbles_out += billes_sorties  # Bille noire sortie

            # Vérifier si quelqu'un a gagné
            is_over, win_message = self.is_game_over()
            if is_over:
                return True, f"{message} {win_message}"

            self.switch_player()
            return True, message

        # Si ce n'est pas une poussée valide, essayer un déplacement normal
        if len(coordinates) <= 3:
            new_board, success, message = make_move(self.board, coordinates, direction)
            if success:
                self.board = new_board
                self.switch_player()
                return True, message

        return False, "Mouvement invalide"

    
    def display(self):
        """Affiche l'état actuel du jeu"""
        cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
        print(f"\nTour des {self.get_current_player_name()}s")
        print(f"Billes sorties - Blanches: {self.white_marbles_out}, Noires: {self.black_marbles_out}")
        
        for y, n_cells in enumerate(cells_per_row):
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

    def test_move(self, coordinates, direction):
        """
        Teste si un mouvement est valide sans modifier l'état du jeu
        Returns:
            bool: True si le mouvement est valide
        """
        # Vérifier si les billes appartiennent au bon joueur
        if not self.check_move_ownership(coordinates):
            return False

        # Essayer d'abord une poussée
        new_board, success, _, _ = push_marbles(self.board, coordinates, direction)
        if success:
            return True

        # Essayer un déplacement normal
        if len(coordinates) <= 3:
            new_board, success, _ = make_move(self.board, coordinates, direction)
            if success:
                return True

        return False
    
    def get_all_valid_groups(self):
       """
       Trouve tous les groupes valides de billes pour le joueur actuel
       Returns:
           list: Liste de tuples de coordonnées représentant les groupes valides
       """
       valid_groups = []
       
       # Trouver toutes les billes du joueur actuel
       player_marbles = []
       for y in range(9):
           for x in range(9):
               content = get_cell_content_jax(self.board, x, y)
               if content == self.current_player:
                   player_marbles.append((x, y))
       
       # D'abord ajouter les groupes de 1 bille
       valid_groups.extend([(marble,) for marble in player_marbles])

       # Tester les combinaisons de 2 et 3 billes
       for size in range(2, 4):
           for marble_group in combinations(player_marbles, size):
               is_valid, msg, alignment = is_valid_group(self.board, marble_group)
               if is_valid:
                   valid_groups.append(marble_group)

       return valid_groups
   
    def get_all_legal_moves(self):
        """
        Récupère tous les coups légaux possibles pour le joueur actuel
        Returns:
            list: Liste de tuples (groupe, direction) représentant chaque coup possible
        """
        legal_moves = []
        valid_groups = self.get_all_valid_groups()
        
        for group in valid_groups:
            if len(group) == 1:
                x, y = group[0]
                neighbors = get_neighbors(self.board, x, y)
                for direction in neighbors.keys():
                    original_board = self.board
                    if self.test_move([group[0]], direction):
                        legal_moves.append((group, direction))
                    self.board = original_board
            else:
                for direction in ['NW', 'NE', 'E', 'SE', 'SW', 'W']:
                    original_board = self.board
                    if self.test_move(list(group), direction):
                        legal_moves.append((group, direction))
                    self.board = original_board
        
        return legal_moves

    def get_state_tensor(self):
        """
        Crée le tenseur d'état pour le réseau neuronal
        Returns:
            jax.Array: Le tenseur représentant l'état du jeu
        """
        # Retourne la forme canonique pour l'instant
        # À modifier selon les besoins du réseau neuronal
        return self.get_canonical_form()