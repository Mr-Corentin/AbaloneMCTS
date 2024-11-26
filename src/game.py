from .board import create_board, display_board
from .utils import get_cell_content
from .moves import push_marbles, make_move

class AbaloneGame:
    def __init__(self):
        self.board = create_board()
        self.current_player = 'W'  # W commence toujours
        self.white_marbles_out = 0
        self.black_marbles_out = 0
    
    def switch_player(self):
        self.current_player = 'B' if self.current_player == 'W' else 'W'
    
    def get_current_player_name(self):
        return "Blanc" if self.current_player == 'W' else "Noir"
    
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
            if get_cell_content(self.board, x, y) != self.current_player:
                return False
        return True
    
    def make_move(self, coordinates, direction):
        """
        Effectue un mouvement (déplacement ou poussée)
        """
        # Vérifier si c'est game over
        is_over, message = self.is_game_over()
        if is_over:
            return False, message
        
        # Vérifier que les billes appartiennent au bon joueur
        if not self.check_move_ownership(coordinates):
            return False, "Ces billes ne vous appartiennent pas"
        
        # Essayer d'abord une poussée
        success, message, billes_sorties = push_marbles(self.board, coordinates, direction)
        if success:
            # Mettre à jour le compte des billes sorties
            if self.current_player == 'W':
                self.black_marbles_out += billes_sorties
            else:
                self.white_marbles_out += billes_sorties
            
            # Vérifier si quelqu'un a gagné
            is_over, win_message = self.is_game_over()
            if is_over:
                return True, f"{message} {win_message}"
            
            self.switch_player()
            return True, message
        
        # Si ce n'est pas une poussée valide, essayer un déplacement normal
        if len(coordinates) <= 3:
            success, message = make_move(self.board, coordinates, direction)
            if success:
                self.switch_player()
                return True, message
        
        return False, "Mouvement invalide"
    
    def display(self):
        """Affiche l'état actuel du jeu"""
        print(f"\nTour des {self.get_current_player_name()}s")
        print(f"Billes sorties - Blanches: {self.white_marbles_out}, Noires: {self.black_marbles_out}")
        display_board(self.board)


def play_game():
    game = AbaloneGame()
    
    while True:
        game.display()
        
        print(f"\nTour des {game.get_current_player_name()}s")
        
        try:
            coord_input = input("Entrez les coordonnées des billes (format: 'x1,y1 x2,y2 ...' ou 'q' pour quitter): ")
            
            if coord_input.lower() == 'q':
                print("Partie terminée!")
                break
            
            # Convertir l'entrée en liste de coordonnées
            coordinates = []
            coord_pairs = coord_input.strip().split()
            
            for coord in coord_pairs:
                x, y = map(int, coord.split(','))
                coordinates.append((x, y))
            
            direction = input("Entrez la direction (NW/NE/E/SE/SW/W): ").upper()
            if direction not in ['NW', 'NE', 'E', 'SE', 'SW', 'W']:
                print("Direction invalide!")
                continue
            
            # Effectuer le mouvement
            success, message = game.make_move(coordinates, direction)
            print(message)
            
            # Vérifier si la partie est terminée
            is_over, end_message = game.is_game_over()
            if is_over:
                game.display()
                print(end_message)
                break
                
        except ValueError:
            print("Format invalide! Utilisez: 'x1,y1 x2,y2' (ex: '2,3 2,4')")
        except IndexError:
            print("Coordonnées hors limites!")
        except Exception as e:
            print(f"Erreur: {str(e)}")