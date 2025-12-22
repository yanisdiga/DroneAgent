import numpy as np
from Obstacle import Obstacle

def random_map_size():
    ''' 
    Génère une taille de carte aléatoire
    '''
    width_map = np.random.randint(800, 1200)
    height_map = np.random.randint(600, 800)
    return (width_map, height_map)

def create_random_obstacle(map_size):
    ''' 
    Génère un obstacle aléatoire
    '''
    width_map, height_map = map_size
    width_obstacle = np.random.randint(50, 200)
    height_obstacle = np.random.randint(50, 200)
    x = np.random.randint(0, width_map - width_obstacle)
    y = np.random.randint(0, height_map - height_obstacle)
    return Obstacle(x, y, width_obstacle, height_obstacle)

def create_random_obstacles(map_size, num_obstacles):
    ''' 
    Génère n obstacles aléatoires
    '''
    obstacles = []
    width_map, height_map = map_size
    
    # On tente de placer num_obstacles
    for _ in range(num_obstacles):
        # On essaie plusieurs fois de trouver une place libre pour cet obstacle
        for _ in range(10): # 10 tentatives max par obstacle
            w = np.random.randint(50, 200)
            h = np.random.randint(50, 200)
            x = np.random.randint(20, width_map - w - 20)
            y = np.random.randint(20, height_map - h - 20)
            
            new_obs = Obstacle(x, y, w, h)
            
            # Vérification de chevauchement avec les obstacles existants
            # On utilise .rect.colliderect car c'est natif et gère les chevauchements partiels
            overlap = any(new_obs.rect.colliderect(o.rect) for o in obstacles)
            
            if not overlap:
                obstacles.append(new_obs)
                break # Place trouvée, on passe à l'obstacle suivant
                
    return obstacles

def dist_segment_point(p1, p2, p3):
    """
    Calcule la distance minimale entre le point p3 et le segment [p1, p2].
    p1: Ancienne position du drone
    p2: Nouvelle position du drone
    p3: Position de la cible
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Si le drone n'a pas bougé
    if np.all(p1 == p2):
        return np.linalg.norm(p3 - p1)

    # Projection vectorielle pour trouver le point le plus proche sur la droite
    l2 = np.sum((p1 - p2)**2)
    t = np.sum((p3 - p1) * (p2 - p1)) / l2
    
    # On borne t entre 0 et 1 pour rester sur le segment (pas la droite infinie)
    t = max(0, min(1, t))
    
    projection = p1 + t * (p2 - p1)
    return np.linalg.norm(p3 - projection)

def add_walls(largeur_carte, hauteur_carte, epaisseur):
    ep = epaisseur # Epaisseur
    murs = [
        Obstacle(0, 0, largeur_carte, ep),
        Obstacle(0, hauteur_carte - ep, largeur_carte, ep),
        Obstacle(0, 0, ep, hauteur_carte),
        Obstacle(largeur_carte - ep, 0, ep, hauteur_carte)
    ]
    return murs