import pygame

class Obstacle:
    def __init__(self, x, y, width, height, color=(100, 100, 100)):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color # On stocke la couleur
    
    def get_position(self):
        return self.rect.x, self.rect.y
    
    def get_size(self):
        return self.rect.width, self.rect.height
    
    def draw(self, screen):
        # On utilise la couleur choisie à l'initialisation
        pygame.draw.rect(screen, self.color, self.rect)