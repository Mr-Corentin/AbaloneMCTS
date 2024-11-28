# from src.game import play_game

# def main():
#     """
#     Point d'entrée principal du jeu Abalone
#     """
#     print("=== Bienvenue dans Abalone ===")
    
#     play_game()

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nPartie interrompue. Au revoir!")
#     except Exception as e:
#         print(f"\nUne erreur est survenue : {e}")

from src.jax_game import JaxAbaloneGame

def play_game():
    game = JaxAbaloneGame()
    
    while True:
        game.display()
        
        try:
            coord_input = input("Entrez les coordonnées des billes (format: 'x1,y1 x2,y2 ...' ou 'q' pour quitter): ")
            
            if coord_input.lower() == 'q':
                print("Partie terminée!")
                break
            
            coordinates = []
            coord_pairs = coord_input.strip().split()
            
            for coord in coord_pairs:
                x, y = map(int, coord.split(','))
                coordinates.append((x, y))
            
            direction = input("Entrez la direction (NW/NE/E/SE/SW/W): ").upper()
            if direction not in ['NW', 'NE', 'E', 'SE', 'SW', 'W']:
                print("Direction invalide!")
                continue
            
            success, message = game.make_move(coordinates, direction)
            print(message)
            
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

if __name__ == "__main__":
    play_game()
