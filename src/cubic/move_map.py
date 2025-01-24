import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Tuple, Set, Dict

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
    
    def to_array(self):
        return [self.x, self.y, self.z]

class Direction(Enum):
    NE = CubeCoord(1, 0, -1)   # Nord-Est
    E = CubeCoord(1, -1, 0)    # Est
    SE = CubeCoord(0, -1, 1)   # Sud-Est
    SW = CubeCoord(-1, 0, 1)   # Sud-Ouest
    W = CubeCoord(-1, 1, 0)    # Ouest
    NW = CubeCoord(0, 1, -1)   # Nord-Ouest

class MoveType(IntEnum):
    SINGLE = 0
    PARALLEL = 1
    INLINE = 2

@dataclass
class Move:
    positions: List[CubeCoord]
    direction: Direction
    move_type: MoveType

class ModernAbaloneMoveMap:
    def __init__(self):
        self.radius = 4
        self.positions: Set[CubeCoord] = set()
        self.moves: Dict[int, Move] = {}
        self.total_moves = 0
        
        # Initialiser le plateau et générer les mouvements
        self.initialize_board()
        self._generate_all_moves()
    
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

    def _add_move(self, positions: List[CubeCoord], direction: Direction, move_type: MoveType):
        """Ajoute un mouvement à la map"""
        positions = list(positions)  # Convertit tuple en liste si nécessaire
        self.moves[self.total_moves] = Move(positions, direction, move_type)
        self.total_moves += 1

    def _generate_single_moves(self):
        """Génère les mouvements pour une seule bille"""
        for pos in self.positions:
            for dir in Direction:
                new_pos = pos + dir.value
                if self.is_valid_position(new_pos):
                    self._add_move([pos], dir, MoveType.SINGLE)

    def _get_alignment_direction(self, positions: List[CubeCoord]) -> Direction:
        """
        Détermine la direction d'alignement d'un groupe de billes
        Returns None si les positions ne forment pas un groupe aligné
        """
        if len(positions) <= 1:
            return None
            
        # Calculer le vecteur entre les deux premières positions
        diff = CubeCoord(
            positions[1].x - positions[0].x,
            positions[1].y - positions[0].y,
            positions[1].z - positions[0].z
        )
        
        # Trouver la direction correspondante
        for dir in Direction:
            if (dir.value.x == diff.x and 
                dir.value.y == diff.y and 
                dir.value.z == diff.z):
                return dir
        return None

    def _get_opposite_direction(self, dir: Direction) -> Direction:
        """Retourne la direction opposée"""
        opposites = {
            Direction.NE: Direction.SW,
            Direction.E: Direction.W,
            Direction.SE: Direction.NW,
            Direction.SW: Direction.NE,
            Direction.W: Direction.E,
            Direction.NW: Direction.SE
        }
        return opposites[dir]

    def _generate_pair_moves(self):
        """Génère les mouvements pour les paires de billes"""
        pairs = self.generate_adjacent_pairs()
        
        for pos1, pos2 in pairs:
            # Déterminer la direction d'alignement du groupe
            positions = [pos1, pos2]
            align_dir = self._get_alignment_direction(positions)
            
            # Pour chaque direction possible
            for dir in Direction:
                # Calculer les nouvelles positions
                new_pos1 = pos1 + dir.value
                new_pos2 = pos2 + dir.value
                
                # Vérifier si le mouvement est valide
                if self.is_valid_position(new_pos1) and self.is_valid_position(new_pos2):
                    # Déterminer le type de mouvement
                    is_inline = (dir == align_dir or dir == self._get_opposite_direction(align_dir))
                    move_type = MoveType.INLINE if is_inline else MoveType.PARALLEL
                    self._add_move([pos1, pos2], dir, move_type)

    def _generate_triplet_moves(self):
        """Génère les mouvements pour les triplets de billes"""
        triplets = self.generate_triplets()
        
        for pos1, pos2, pos3 in triplets:
            # Déterminer la direction d'alignement du groupe
            positions = [pos1, pos2, pos3]
            align_dir = self._get_alignment_direction(positions)
            
            # Pour chaque direction possible
            for dir in Direction:
                # Calculer les nouvelles positions
                new_pos1 = pos1 + dir.value
                new_pos2 = pos2 + dir.value
                new_pos3 = pos3 + dir.value
                
                # Vérifier si le mouvement est valide
                if (self.is_valid_position(new_pos1) and 
                    self.is_valid_position(new_pos2) and 
                    self.is_valid_position(new_pos3)):
                    # Déterminer le type de mouvement
                    is_inline = (dir == align_dir or dir == self._get_opposite_direction(align_dir))
                    move_type = MoveType.INLINE if is_inline else MoveType.PARALLEL
                    self._add_move([pos1, pos2, pos3], dir, move_type)
                    
    def _generate_all_moves(self):
        """Génère tous les mouvements possibles"""
        self._generate_single_moves()
        self._generate_pair_moves()
        self._generate_triplet_moves()

    def save_to_numpy(self, filename: str = "move_map.npz"):
        """Sauvegarde la map dans un format numpy"""
        max_group_size = max(len(m.positions) for m in self.moves.values())
        
        # Création des tableaux numpy
        positions = np.zeros((len(self.moves), max_group_size, 3), dtype=np.int8)
        directions = np.zeros(len(self.moves), dtype=np.int8)
        move_types = np.zeros(len(self.moves), dtype=np.int8)
        group_sizes = np.zeros(len(self.moves), dtype=np.int8)
        
        # Remplissage des tableaux
        for idx, move in self.moves.items():
            # Coordonnées
            for i, pos in enumerate(move.positions):
                positions[idx, i] = [pos.x, pos.y, pos.z]
            
            # Direction (utilisez l'index de l'enum)
            directions[idx] = list(Direction).index(move.direction)
            
            # Type de mouvement
            move_types[idx] = move.move_type
            
            # Taille du groupe
            group_sizes[idx] = len(move.positions)
        
        # Sauvegarde
        np.savez_compressed(
            filename,
            positions=positions,
            directions=directions,
            move_types=move_types,
            group_sizes=group_sizes
        )
        print(f"\nMap sauvegardée dans {filename}")

    def print_stats(self):
        """Affiche les statistiques des mouvements"""
        stats = {
            MoveType.SINGLE: 0,
            MoveType.PARALLEL: 0,
            MoveType.INLINE: 0
        }
        
        for move in self.moves.values():
            stats[move.move_type] += 1
        
        print("\nStatistiques des mouvements possibles:")
        print(f"Mouvements simples: {stats[MoveType.SINGLE]}")
        print(f"Mouvements parallèles: {stats[MoveType.PARALLEL]}")
        print(f"Mouvements en ligne: {stats[MoveType.INLINE]}")
        print(f"Total: {self.total_moves}")


