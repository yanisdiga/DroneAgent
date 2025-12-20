from Capteur import Capteur

class Drone:
    def __init__(self, id, x, y, capteur, vitesse=1):
        self.id = id
        self.x = x
        self.y = y
        self.capteur = capteur
        self.vitesse = vitesse

    def get_id(self):
        return self.id

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_position(self):
        return (self.x, self.y)

    def get_capteur(self):
        return self.capteur

    def get_vitesse(self):
        return self.vitesse

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def set_capteur(self, capteur):
        self.capteur = capteur

    def set_vitesse(self, vitesse):
        self.vitesse = vitesse

    def action(self, action):
        if action=="forward":
            self.set_y(self.get_y()+self.vitesse)
        elif action=="backward":
            self.set_y(self.get_y()-self.vitesse)
        elif action=="left":
            self.set_x(self.get_x()-self.vitesse)
        elif action=="right":
            self.set_x(self.get_x()+self.vitesse)

    def move_vector(self, dx, dy, map_width, map_height):
        # 1. Appliquer le mouvement
        self.x += dx
        self.y += dy
        
        # 2. Bloquer aux bords (Clamping) pour ne pas sortir de la carte
        if self.x < 0: self.x = 0
        if self.x > map_width: self.x = map_width
        if self.y < 0: self.y = 0
        if self.y > map_height: self.y = map_height