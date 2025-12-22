from Capteur import Capteur

class Drone:
    def __init__(self, id, x, y, capteur, vitesse=1):
        ''' 
        Constructeur de la classe Drone
        '''
        self.id = id
        self.x = x
        self.y = y
        self.capteur = capteur
        self.vitesse = vitesse

    def get_id(self):
        ''' 
        Renvoie l'ID du drone
        '''
        return self.id

    def get_x(self):
        ''' 
        Renvoie la position x du drone
        '''
        return self.x

    def get_y(self):
        ''' 
        Renvoie la position y du drone
        '''
        return self.y

    def get_position(self):
        ''' 
        Renvoie la position (x, y) du drone
        '''
        return (self.x, self.y)

    def get_capteur(self):
        ''' 
        Renvoie le capteur du drone
        '''
        return self.capteur

    def get_vitesse(self):
        ''' 
        Renvoie la vitesse du drone
        '''
        return self.vitesse

    def set_x(self, x):
        ''' 
        Modifie la position x du drone
        '''
        self.x = x

    def set_y(self, y):
        ''' 
        Modifie la position y du drone
        '''
        self.y = y

    def set_position(self, x, y):
        ''' 
        Modifie la position (x, y) du drone
        '''
        self.x = x
        self.y = y

    def set_capteur(self, capteur):
        ''' 
        Modifie le capteur du drone
        '''
        self.capteur = capteur

    def set_vitesse(self, vitesse):
        ''' 
        Modifie la vitesse du drone
        '''
        self.vitesse = vitesse

    def action(self, action):
        ''' 
        Modifie la position du drone en fonction de l'action
        '''
        if action=="forward":
            self.set_y(self.get_y()+self.vitesse)
        elif action=="backward":
            self.set_y(self.get_y()-self.vitesse)
        elif action=="left":
            self.set_x(self.get_x()-self.vitesse)
        elif action=="right":
            self.set_x(self.get_x()+self.vitesse)

    def move_vector(self, dx, dy, map_width, map_height):
        ''' 
        Modifie la position du drone en fonction du vecteur (dx, dy)
        '''
        # 1. Appliquer le mouvement
        self.x += dx
        self.y += dy
        
        # 2. Bloquer aux bords (Clamping) pour ne pas sortir de la carte
        if self.x < 0: self.x = 0
        if self.x > map_width: self.x = map_width
        if self.y < 0: self.y = 0
        if self.y > map_height: self.y = map_height