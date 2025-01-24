import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Set

@dataclass(frozen=True)
class CubeCoord:
    x: int
    y: int
    z: int
    
    def __post_init__(self):
        if self.x + self.y + self.z != 0:
            raise ValueError(f"Invalid cube coordinates: {self.x}+{self.y}+{self.z}≠0")
            
    def __add__(self, other):
        return CubeCoord(self.x + other.x, self.y + other.y, self.z + other.z)

class Direction(Enum):
    NE = CubeCoord(1, 0, -1)   # Nord-Est
    E = CubeCoord(1, -1, 0)    # Est
    SE = CubeCoord(0, -1, 1)   # Sud-Est
    SW = CubeCoord(-1, 0, 1)   # Sud-Ouest
    W = CubeCoord(-1, 1, 0)    # Ouest
    NW = CubeCoord(0, 1, -1)   # Nord-Ouest

class AbaloneBoard:
    def __init__(self):
        self.radius = 4
        self.positions: Set[CubeCoord] = set()
        self.initialize_board()

    def initialize_board(self):
        """Initialise le plateau avec les coordonnées cubiques"""
        for z in range(-self.radius, self.radius + 1):
            x_min = max(-self.radius, -z - self.radius)
            x_max = min(self.radius, -z + self.radius)
            for x in range(x_min, x_max + 1):
                y = -x - z
                if abs(y) <= self.radius:
                    self.positions.add(CubeCoord(x, y, z))

    def is_valid_position(self, coord: CubeCoord) -> bool:
        return coord in self.positions

    def generate_adjacent_pairs(self) -> List[Tuple[CubeCoord, CubeCoord]]:
        """Génère toutes les paires de cases adjacentes"""
        pairs = []
        processed = set()
        
        for pos1 in self.positions:
            for dir in Direction:
                pos2 = pos1 + dir.value
                if self.is_valid_position(pos2):
                    pair_key = tuple(sorted([(pos1.x, pos1.y, pos1.z), (pos2.x, pos2.y, pos2.z)]))
                    if pair_key not in processed:
                        pairs.append((pos1, pos2))
                        processed.add(pair_key)
        return pairs

    def generate_triplets(self) -> List[Tuple[CubeCoord, CubeCoord, CubeCoord]]:
        """Génère tous les triplets de cases alignées"""
        triplets = []
        processed = set()
        
        for pos1 in self.positions:
            for dir in Direction:
                pos2 = pos1 + dir.value
                if not self.is_valid_position(pos2):
                    continue
                pos3 = pos2 + dir.value
                if not self.is_valid_position(pos3):
                    continue
                
                triplet_key = tuple(sorted([(p.x, p.y, p.z) for p in [pos1, pos2, pos3]]))
                if triplet_key not in processed:
                    triplets.append((pos1, pos2, pos3))
                    processed.add(triplet_key)
        
        return triplets

    def count_all_moves(self):
        """Calcule tous les mouvements possibles"""
        # Mouvements pour 1 bille
        total_single = sum(len([dir for dir in Direction 
                              if self.is_valid_position(pos + dir.value)]) 
                          for pos in self.positions)
        
        # Mouvements pour 2 billes
        moves_two = []
        for pos1, pos2 in self.generate_adjacent_pairs():
            for dir in Direction:
                new_pos1 = pos1 + dir.value
                new_pos2 = pos2 + dir.value
                if (self.is_valid_position(new_pos1) and 
                    self.is_valid_position(new_pos2)):
                    moves_two.append({
                        'start': (pos1, pos2),
                        'move_dir': dir,
                        'end': (new_pos1, new_pos2)
                    })

        # Mouvements pour 3 billes
        moves_three = []
        for pos1, pos2, pos3 in self.generate_triplets():
            for dir in Direction:
                new_pos1 = pos1 + dir.value
                new_pos2 = pos2 + dir.value
                new_pos3 = pos3 + dir.value
                if (self.is_valid_position(new_pos1) and 
                    self.is_valid_position(new_pos2) and
                    self.is_valid_position(new_pos3)):
                    moves_three.append({
                        'start': (pos1, pos2, pos3),
                        'move_dir': dir,
                        'end': (new_pos1, new_pos2, new_pos3)
                    })

        return {
            'single': total_single,
            'double': (len(moves_two), moves_two),
            'triple': (len(moves_three), moves_three),
            'total': total_single + len(moves_two) + len(moves_three)
        }

    def save_pairs_to_file(self, pairs: List[Tuple[CubeCoord, CubeCoord]], filename="pairs.txt"):
        with open(filename, 'w') as f:
            f.write(f"Nombre total de paires: {len(pairs)}\n\n")
            for i, (pos1, pos2) in enumerate(pairs, 1):
                f.write(f"Paire {i}:\n")
                f.write(f"  Position 1: ({pos1.x},{pos1.y},{pos1.z})\n")
                f.write(f"  Position 2: ({pos2.x},{pos2.y},{pos2.z})\n\n")

    def save_triplets_to_file(self, triplets: List[Tuple[CubeCoord, CubeCoord, CubeCoord]], filename="triplets.txt"):
        with open(filename, 'w') as f:
            f.write(f"Nombre total de triplets: {len(triplets)}\n\n")
            for i, (pos1, pos2, pos3) in enumerate(triplets, 1):
                f.write(f"Triplet {i}:\n")
                f.write(f"  Position 1: ({pos1.x},{pos1.y},{pos1.z})\n")
                f.write(f"  Position 2: ({pos2.x},{pos2.y},{pos2.z})\n")
                f.write(f"  Position 3: ({pos3.x},{pos3.y},{pos3.z})\n")
                for dir in Direction:
                    if pos2 == pos1 + dir.value and pos3 == pos2 + dir.value:
                        f.write(f"  Direction: {dir.name}\n")
                        break
                f.write("\n")

    def save_moves_to_file(self, moves_details, filename="moves.txt"):
        with open(filename, 'w') as f:
            f.write(f"Nombre total de mouvements: {len(moves_details)}\n\n")
            for i, move in enumerate(moves_details, 1):
                f.write(f"Mouvement {i}:\n")
                f.write(f"  Depart: {' et '.join(str((p.x,p.y,p.z)) for p in move['start'])}\n")
                f.write(f"  Direction: {move['move_dir'].name}\n")
                f.write(f"  Arrivee: {' et '.join(str((p.x,p.y,p.z)) for p in move['end'])}\n\n")

if __name__ == "__main__":
    board = AbaloneBoard()
    
    # Générer et sauvegarder les configurations
    pairs = board.generate_adjacent_pairs()
    triplets = board.generate_triplets()
    board.save_pairs_to_file(pairs)
    board.save_triplets_to_file(triplets)
    
    # Calculer tous les mouvements
    results = board.count_all_moves()
    
    # Afficher les résultats
    print("\nRésultats détaillés:")
    print("-" * 40)
    print(f"Mouvements pour 1 bille  : {results['single']}")
    print(f"Mouvements pour 2 billes : {results['double'][0]}")
    print(f"Mouvements pour 3 billes : {results['triple'][0]}")
    print("-" * 40)
    print(f"Total des mouvements     : {results['total']}")
    
    # Sauvegarder les mouvements
    board.save_moves_to_file(results['double'][1], "moves_2_billes.txt")
    board.save_moves_to_file(results['triple'][1], "moves_3_billes.txt")