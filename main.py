from src.game import play_game

def main():
    """
    Point d'entrée principal du jeu Abalone
    """
    print("=== Bienvenue dans Abalone ===")
    
    play_game()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPartie interrompue. Au revoir!")
    except Exception as e:
        print(f"\nUne erreur est survenue : {e}")