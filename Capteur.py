import math
import pygame
import numpy as np

class Capteur:
    def __init__(self, rayon):
        self.rayon = rayon
        self.normalized_directions = []
        
        # Pré-calcul des directions (inchangé)
        nb_rayons = 16
        for i in range(nb_rayons):
            angle_rad = math.radians(i * (360 / nb_rayons))
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            self.normalized_directions.append((dx, dy))
    
    def get_rayon(self):
        return self.rayon

    # --- VERSION ULTRA-RAPIDE (C-Optimized) ---
    def scan_lidar(self, position, obstacles):
        ''' 
        Scan optimisé utilisant clipline au lieu de boucles Python.
        Attend une LISTE d'objets Obstacle (pas une fonction de callback).
        '''
        x, y = position
        sensors = []
        
        for dx, dy in self.normalized_directions:
            end_x = x + dx * self.rayon
            end_y = y + dy * self.rayon
            
            min_dist = self.rayon
            
            # On teste l'intersection avec tous les obstacles
            for obs in obstacles:
                # clipline retourne le segment de ligne coupé par le rectangle
                clipped = obs.rect.clipline(x, y, end_x, end_y)
                
                if clipped:
                    # Le premier point du segment est le point d'impact
                    impact_x, impact_y = clipped[0]
                    dist = math.hypot(impact_x - x, impact_y - y)
                    if dist < min_dist:
                        min_dist = dist
            
            sensors.append(min_dist / self.rayon)
            
        return sensors