import numpy as np
import pygame
import math

class Cible:
    def __init__(self, x, y, vitesse=5, map_width=800, map_height=600):
        self.x = x
        self.y = y
        self.vitesse = vitesse # Augmente ça à 3.0 ou 4.0 dans DroneEnv si c'est encore trop lent
        self.map_width = map_width
        self.map_height = map_height
        
        # --- AJOUT INERTIE ---
        self.dx = 0
        self.dy = 0
        self.change_timer = 0 # Compteur pour garder la même direction

    def get_position(self):
        return (self.x, self.y)

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def move_random(self, obstacles):
        # 1. EST-CE QU'ON CHANGE DE DIRECTION ?
        # Si le timer est fini, on choisit un nouveau cap
        if self.change_timer <= 0:
            angle = np.random.uniform(0, 2 * math.pi)
            self.dx = math.cos(angle) * self.vitesse
            self.dy = math.sin(angle) * self.vitesse
            
            # On garde ce cap pendant 30 à 100 frames (0.5 à 1.5 seconde)
            self.change_timer = np.random.randint(30, 100)
            
        # On décrémente le timer
        self.change_timer -= 1
        
        # 2. TENTATIVE DE DÉPLACEMENT
        next_x = self.x + self.dx
        next_y = self.y + self.dy
        
        # 3. GESTION OBSTACLES & MURS
        # On crée le rect futur
        test_rect = pygame.Rect(next_x - 5, next_y - 5, 10, 10)
        
        collision = False
        
        # Vérif Murs (Rebondir au lieu de glisser c'est plus sympa pour une cible)
        if next_x < 0 or next_x > self.map_width:
            self.dx *= -1 # Rebond horizontal
            collision = True
            
        if next_y < 0 or next_y > self.map_height:
            self.dy *= -1 # Rebond vertical
            collision = True
            
        # Vérif Obstacles
        if any(obs.rect.colliderect(test_rect) for obs in obstacles):
            # Si on fonce dans un mur, on inverse le cap immédiatement et on reset le timer
            self.dx *= -1
            self.dy *= -1
            self.change_timer = 0 # Pour qu'elle cherche une nouvelle issue vite
            collision = True
        
        if not collision:
            self.x += self.dx
            self.y += self.dy