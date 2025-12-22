import pygame

class Obstacle:
    def __init__(self, x, y, width, height, color=(100, 100, 100)):
        ''' 
        Constructeur de la classe Obstacle
        '''
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color # On stocke la couleur
    
    def get_position(self):
        ''' 
        Renvoie la position (x, y) de l'obstacle
        '''
        return self.rect.x, self.rect.y
    
    def get_size(self):
        ''' 
        Renvoie la taille (largeur, hauteur) de l'obstacle
        '''
        return self.rect.width, self.rect.height
    
    def draw(self, screen):
        ''' 
        Dessine l'obstacle sur la surface donnée
        '''
        # On utilise la couleur choisie à l'initialisation
        pygame.draw.rect(screen, self.color, self.rect)