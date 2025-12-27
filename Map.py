import numpy as np
import pygame
from Obstacle import Obstacle

class Map:
    def __init__(self, size_min=(800, 600), size_max=(1200, 800)):
        self.size_min = size_min
        self.size_max = size_max
        self.width = 800
        self.height = 600
        self.obstacles = []
        self.walls = []

    def reset_random(self, difficulty=None):
        '''
        Régénère une carte aléatoire avec des dimensions et obstacles variés
        '''
        # 1. Dimensions aléatoires
        self.width = np.random.randint(self.size_min[0], self.size_max[0])
        self.height = np.random.randint(self.size_min[1], self.size_max[1])
        
        # 2. Gestion de la difficulté (si pas spécifiée, aléatoire)
        if difficulty is None:
            # 20% niv 0, 30% niv 1, 30% niv 2, 20% niv 3
            difficulty = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])

        self.obstacles = []
        
        # 3. Génération des obstacles internes
        # Niveau 0 : Aucun obstacle
        if difficulty == 1:
            nb_obs = np.random.randint(2, 4)
        elif difficulty == 2:
            nb_obs = np.random.randint(3, 7)
            self._generate_obstacles(nb_obs)
        elif difficulty == 3:
            nb_obs = np.random.randint(5, 9)
            self._generate_obstacles(nb_obs)

        # 4. Ajout des murs (toujours présents)
        self._add_walls()

    def _generate_obstacles(self, nb_obstacles):
        ''' Méthode privée pour placer les obstacles sans chevauchement '''
        attempts = 0
        while len(self.obstacles) < nb_obstacles and attempts < 50:
            w = np.random.randint(50, 200)
            h = np.random.randint(50, 200)
            x = np.random.randint(20, self.width - w - 20)
            y = np.random.randint(20, self.height - h - 20)
            
            new_obs = Obstacle(x, y, w, h)
            
            # Vérif chevauchement
            if not any(o.rect.colliderect(new_obs.rect) for o in self.obstacles):
                self.obstacles.append(new_obs)
            attempts += 1

    def _add_walls(self, epaisseur=20):
        ''' Ajoute les murs autour de la map '''
        self.walls = [
            Obstacle(0, 0, self.width, epaisseur), # Haut
            Obstacle(0, self.height - epaisseur, self.width, epaisseur), # Bas
            Obstacle(0, 0, epaisseur, self.height), # Gauche
            Obstacle(self.width - epaisseur, 0, epaisseur, self.height) # Droite
        ]

    def get_all_obstacles(self):
        ''' Retourne murs + obstacles pour la gestion des collisions '''
        return self.obstacles + self.walls

    def check_collision(self, rect):
        ''' Vérifie si un rectangle touche un obstacle ou un mur '''
        all_obs = self.get_all_obstacles()
        return any(o.rect.colliderect(rect) for o in all_obs)

    def draw(self, screen):
        ''' Dessine la map '''
        # Fond
        screen.fill((20, 20, 20))
        # Obstacles
        for obs in self.get_all_obstacles():
            obs.draw(screen)