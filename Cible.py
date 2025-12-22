# Cible.py
import numpy as np
import pygame

class Cible:
    def __init__(self, x, y, vitesse=2, map_width=800, map_height=600):
        self.x = x
        self.y = y
        self.vitesse = vitesse
        self.map_width = map_width
        self.map_height = map_height

    def get_position(self):
        return (self.x, self.y)

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def move_random(self, obstacles):
        # Mouvement aléatoire, mais évite les obstacles simples
        dx = np.random.uniform(-self.vitesse, self.vitesse)
        dy = np.random.uniform(-self.vitesse, self.vitesse)
        
        # Test collision (similaire à ton drone)
        test_rect = pygame.Rect((self.x + dx) - 5, (self.y + dy) - 5, 10, 10)
        if not any(obs.rect.colliderect(test_rect) for obs in obstacles):
            self.x += dx
            self.y += dy
        
        # Clamping aux bords
        self.x = max(0, min(self.x, self.map_width))
        self.y = max(0, min(self.y, self.map_height))