from board import create_board, get_cell_content
from utils import is_valid_group, get_neighbors
from moves import push_marbles, make_move

class AbaloneMoveMap:
    def __init__(self):
        self.move_to_index = {}
        self.index_to_move = {}
        self.total_moves = 0
        self.cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
        
        # Compteurs par type de mouvement
        self.single_moves = 0
        self.double_moves = 0
        self.triple_moves = 0
        
        # Initialisation du plateau de test
        self.test_board = create_board()
        self._initialize_empty_board()
        
        # Génération des mouvements
        self._generate_all_moves()
        self._print_stats()
    
    def _initialize_empty_board(self):
        """Initialise un plateau vide"""
        for y, cells in enumerate(self.cells_per_row):
            for x in range(cells):
                self.test_board[y][x] = 0

    def _cell_exists(self, x, y):
        """Vérifie si une cellule existe sur le plateau"""
        if x < 0 or y < 0 or y >= len(self.cells_per_row):
            return False
        return x < self.cells_per_row[y]

    def _normalize_group(self, coordinates):
        """Normalise l'ordre des coordonnées dans un groupe"""
        return sorted(coordinates, key=lambda pos: (pos[1], pos[0]))

    def _is_move_valid(self, coordinates, direction):
        """Vérifie si un mouvement est valide"""
        # Vérifier que toutes les positions existent
        if not all(self._cell_exists(x, y) for x, y in coordinates):
            return False

        # Créer un plateau de test avec le groupe
        test_board = create_board()
        for y, cells in enumerate(self.cells_per_row):
            for x in range(cells):
                test_board[y][x] = 0
        
        # Placer les billes du groupe
        for x, y in coordinates:
            test_board[y][x] = 1

        # Pour une seule bille, vérifier si la direction est valide
        if len(coordinates) == 1:
            x, y = coordinates[0]
            if direction not in get_neighbors(test_board, x, y):
                return False

        # Vérifier si le groupe est valide (pour 2+ billes)
        if len(coordinates) > 1:
            is_valid, _, _ = is_valid_group(test_board, coordinates)
            if not is_valid:
                return False

        # Tester le mouvement
        is_push, _, _ = push_marbles(test_board, list(coordinates), direction)
        if is_push:
            return True
            
        return make_move(test_board, coordinates, direction)[0]

    def _add_move(self, coordinates, direction):
        """Ajoute un mouvement à la map si valide"""
        coordinates = self._normalize_group(coordinates)
        
        if self._is_move_valid(coordinates, direction):
            move_str = f"{coordinates}:{direction}"
            if move_str not in self.move_to_index:
                self.move_to_index[move_str] = self.total_moves
                self.index_to_move[self.total_moves] = move_str
                self.total_moves += 1
                
                if len(coordinates) == 1:
                    self.single_moves += 1
                elif len(coordinates) == 2:
                    self.double_moves += 1
                else:
                    self.triple_moves += 1

    def _generate_single_marble_moves(self):
        """Génère les mouvements pour une bille"""
        for y, cells in enumerate(self.cells_per_row):
            for x in range(cells):
                for direction in get_neighbors(self.test_board, x, y):
                    self._add_move([(x, y)], direction)

    def _generate_double_marble_moves(self):
        """Génère les mouvements pour deux billes"""
        for y, cells in enumerate(self.cells_per_row):
            for x in range(cells):
                for direction, (x2, y2) in get_neighbors(self.test_board, x, y).items():
                    if (x2, y2) != (x, y):
                        group = [(x, y), (x2, y2)]
                        for move_dir in ['NW', 'NE', 'E', 'SE', 'SW', 'W']:
                            self._add_move(group, move_dir)

    def _generate_triple_marble_moves(self):
        """Génère les mouvements pour trois billes"""
        for y, cells in enumerate(self.cells_per_row):
            for x in range(cells):
                neighbors1 = get_neighbors(self.test_board, x, y)
                for _, (x2, y2) in neighbors1.items():
                    if (x2, y2) == (x, y):
                        continue
                    
                    neighbors2 = get_neighbors(self.test_board, x2, y2)
                    for _, (x3, y3) in neighbors2.items():
                        if (x3, y3) in [(x, y), (x2, y2)]:
                            continue
                        
                        group = [(x, y), (x2, y2), (x3, y3)]
                        for move_dir in ['NW', 'NE', 'E', 'SE', 'SW', 'W']:
                            self._add_move(group, move_dir)

    def _generate_all_moves(self):
        """Génère tous les mouvements possibles"""
        self._generate_single_marble_moves()
        self._generate_double_marble_moves()
        self._generate_triple_marble_moves()

    def _print_stats(self):
        """Affiche les statistiques des mouvements"""
        print("\nStatistiques des mouvements possibles:")
        print(f"Mouvements d'une bille   : {self.single_moves}")
        print(f"Mouvements de deux billes: {self.double_moves}")
        print(f"Mouvements de trois billes: {self.triple_moves}")
        print(f"Nombre total de mouvements: {self.total_moves}")

    def get_index(self, move_str):
        """Retourne l'index d'un mouvement"""
        return self.move_to_index.get(move_str)

    def get_move(self, index):
        """Retourne le mouvement correspondant à un index"""
        return self.index_to_move.get(index)
    
    def find_moves_by_group(self, coordinates):
        """Trouve tous les mouvements pour un groupe de billes"""
        moves_found = []
        coordinates = self._normalize_group(coordinates)
        
        for index, move_str in self.index_to_move.items():
            group_str = move_str.split(':')[0]
            if group_str == str(coordinates):
                direction = move_str.split(':')[1]
                moves_found.append((direction, index))
        
        if moves_found:
            print(f"\nMouvements trouvés pour le groupe {coordinates}:")
            for direction, index in sorted(moves_found, key=lambda x: x[0]):
                print(f"Index {index}: Direction {direction}")
            print(f"Total: {len(moves_found)} mouvements possibles")
        else:
            print(f"\nAucun mouvement trouvé pour le groupe {coordinates}")
        
        return moves_found
    
    
    def display_move_map_sample(self, n_samples=5):
        """Affiche un échantillon de la map pour chaque type de mouvement"""
        print("\nÉchantillon de la bijective map :")
        print("-" * 50)
        
        # Pour chaque type de mouvement
        for move_size in [1, 2, 3]:
            print(f"\nMouvements de {move_size} bille(s):")
            count = 0
            for index, move_str in self.index_to_move.items():
                group = eval(move_str.split(':')[0])  # Convertit la string en liste de tuples
                if len(group) == move_size:
                    direction = move_str.split(':')[1]
                    print(f"Index {index:4d}: Groupe {group}, Direction {direction}")
                    count += 1
                    if count >= n_samples:
                        break
                        
    def get_move_info(self, index):
        """Retourne les informations détaillées d'un mouvement"""
        if index not in self.index_to_move:
            return None
            
        move_str = self.index_to_move[index]
        group_str, direction = move_str.split(':')
        group = eval(group_str) 
        
        return {
            'index': index,
            'group': group,
            'group_size': len(group),
            'direction': direction,
            'original_string': move_str
        }

    def save_move_map(self, filename="move_map.txt"):
        """Sauvegarde la map complète dans un fichier"""
        with open(filename, 'w') as f:
            f.write(f"Nombre total de mouvements: {self.total_moves}\n")
            f.write(f"Mouvements d'une bille: {self.single_moves}\n")
            f.write(f"Mouvements de deux billes: {self.double_moves}\n")
            f.write(f"Mouvements de trois billes: {self.triple_moves}\n\n")
            
            for index in sorted(self.index_to_move.keys()):
                move_info = self.get_move_info(index)
                f.write(f"Index {index:4d}: {move_info['group']} -> {move_info['direction']}\n")