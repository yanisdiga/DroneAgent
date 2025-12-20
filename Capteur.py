import math
import pygame

class Capteur:
    def __init__(self, rayon):
        self.rayon = rayon
        self.normalized_directions = []
        
        # On génère 16 angles (de 0 à 360 degrés)
        nb_rayons = 16
        for i in range(nb_rayons):
            angle_rad = math.radians(i * (360 / nb_rayons))
            # Attention en informatique : Y est inversé souvent, ou cos/sin classiques
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            self.normalized_directions.append((dx, dy))
    
    def get_rayon(self):
        return self.rayon

    def set_rayon(self, rayon):
        self.rayon = rayon

    def scan(self, me, environment):
        neighbors = []
        me_x, me_y = me.get_position()
        for env in environment:
            if env.get_id() != me.get_id():
                env_x, env_y = env.get_position()
                dx = env_x - me_x
                dy = env_y - me_y
                # Distance euclidienne
                distance = math.sqrt(dx**2 + dy**2)
                if distance <= self.rayon:
                    neighbors.append(env)
        return neighbors
    
    def scan_lidar(self, position, check_collision_fn):
        x, y = position
        sensors = []
        
        # Taille du drone (pour la hitbox du lidar)
        drone_size = 10 
        half_size = drone_size // 2

        for dx, dy in self.normalized_directions:
            dist_obs = self.rayon
            
            # On avance pas à pas le long du rayon
            # Pas de 5 pixels (compromis précision/vitesse)
            for step in range(1, int(self.rayon), 5):
                check_x = x + dx * step
                check_y = y + dy * step
                
                # On crée un rect virtuel centré sur le point du rayon
                test_rect = pygame.Rect(check_x - half_size, check_y - half_size, drone_size, drone_size)
                
                # check_collision_fn doit maintenant accepter un Rect, pas juste x,y
                if check_collision_fn(test_rect):
                    dist_obs = step
                    break
            
            sensors.append(dist_obs / self.rayon)
        return sensors