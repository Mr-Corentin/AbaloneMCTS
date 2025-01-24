from move_map import AbaloneMoveMap

def test_move_mapping():
    move_map = AbaloneMoveMap()
    print(f"Nombre total de mouvements possibles : {move_map.total_moves}")
    
def test_move_map():
    move_map = AbaloneMoveMap()
    
    # Afficher un échantillon
    move_map.display_move_map_sample()
    
    # Tester quelques index spécifiques
    test_indices = [0, 10, 100]
    print("\nTest de quelques index spécifiques:")
    for idx in test_indices:
        move_info = move_map.get_move_info(idx)
        if move_info:
            print(f"\nMouvement {idx}:")
            for key, value in move_info.items():
                print(f"  {key}: {value}")
    
    # Sauvegarder la map complète
    move_map.save_move_map()

if __name__ == "__main__":
    test_move_mapping()
    # move_map = AbaloneMoveMap()

    # move_map.find_moves_by_group([(0, 0), (1, 0)])

    # move_map.find_moves_by_group([(1, 0), (0, 0)])

    # # Test pour trois billes
    # move_map.find_moves_by_group([(0, 0), (1, 0), (2, 0)])


    # move_map.find_moves_by_group([(3, 5), (4, 5), (5, 5)])

    # move_map.find_moves_by_group([(2, 5), (2, 6), (2, 7)])
    test_move_map()



    