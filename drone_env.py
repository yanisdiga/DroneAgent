import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from Obstacle import Obstacle
from Drone import Drone
from Capteur import Capteur

def random_map_size():
    width_map = np.random.randint(800, 1200)
    height_map = np.random.randint(600, 800)
    return (width_map, height_map)

def create_random_obstacle(map_size):
    width_map, height_map = map_size
    width_obstacle = np.random.randint(50, 200)
    height_obstacle = np.random.randint(50, 200)
    x = np.random.randint(0, width_map - width_obstacle)
    y = np.random.randint(0, height_map - height_obstacle)
    return Obstacle(x, y, width_obstacle, height_obstacle)

def create_random_obstacles(map_size, num_obstacles):
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

class DroneEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(DroneEnv, self).__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # --- CONFIGURATION CENTRALE ---
        self.CONFIG = {
            "MAP_SIZE_MIN": (800, 600),
            "MAP_SIZE_MAX": (1200, 800),
            "MAX_STEPS": 2000,
            "RAYON_LIDAR": 80,
            "RAYON_CAPTURE": 20,
            "VITESSE": 5,
            "REWARD_TARGET": 1000.0,
            "REWARD_EXPLORATION": 0.5,
        }

        self.largeur_carte, self.hauteur_carte = self.CONFIG["MAP_SIZE_MIN"]
        self.max_steps = self.CONFIG["MAX_STEPS"]
        
        # --- Intégration de tes Classes POO ---
        # On crée un Capteur avec le rayon défini
        self.mon_capteur = Capteur(rayon=self.CONFIG["RAYON_LIDAR"]) 
        # On crée le Drone (position temporaire 0,0)
        self.drone_agent = Drone(id=1, x=0, y=0, capteur=self.mon_capteur, vitesse=self.CONFIG["VITESSE"])
        
        self.obstacles = []
        self.rayon_capture = self.CONFIG["RAYON_CAPTURE"]

        # --- Espaces d'Action et d'Observation ---
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation identique
        low_obs = np.array([0]*20, dtype=np.float32)
        high_obs = np.array([1]*20, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

    def _add_walls(self):
        ep = 20 # Epaisseur
        self.murs = [
            Obstacle(0, 0, self.largeur_carte, ep),
            Obstacle(0, self.hauteur_carte - ep, self.largeur_carte, ep),
            Obstacle(0, 0, ep, self.hauteur_carte),
            Obstacle(self.largeur_carte - ep, 0, ep, self.hauteur_carte)
        ]
        self.obstacles.extend(self.murs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        niveau = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]) 
        
        if niveau == 1:
            self.obstacles = []
        elif niveau == 2:
            self.obstacles = create_random_obstacles((self.largeur_carte, self.hauteur_carte), np.random.randint(3, 7))
        elif niveau == 3:
            self.obstacles = create_random_obstacles((self.largeur_carte, self.hauteur_carte), np.random.randint(7, 15))

        # 1. Map aléatoire
        self.largeur_carte, self.hauteur_carte = random_map_size()
        
        # 2. Obstacles aléatoires
        self.obstacles = create_random_obstacles((self.largeur_carte, self.hauteur_carte), np.random.randint(3, 7))
        self._add_walls() # Ajout des murs
        
        # 3. Position Drone (On utilise ton objet Drone !)
        while True:
            start_x = np.random.uniform(50, self.largeur_carte-50)
            start_y = np.random.uniform(50, self.hauteur_carte-50)
            
            # Vérif collision au spawn
            drone_rect = pygame.Rect(start_x-5, start_y-5, 10, 10)
            if not any(obs.rect.colliderect(drone_rect) for obs in self.obstacles):
                # On met à jour TON objet Drone
                self.drone_agent.set_position(start_x, start_y)
                break

        # 4. Position Cible
        while True:
            self.cible_pos = np.array([np.random.uniform(50, self.largeur_carte-50), np.random.uniform(50, self.hauteur_carte-50)])
            cible_rect = pygame.Rect(self.cible_pos[0]-5, self.cible_pos[1]-5, 10, 10)
            # CORRECTION BUG: ajout de .rect
            if not any(obs.rect.colliderect(cible_rect) for obs in self.obstacles):
                break
        
        # On récupère la pos du drone sous forme d'array numpy pour les calculs mathématiques
        drone_pos_array = np.array(self.drone_agent.get_position())
        self.zones_visitees = [drone_pos_array.copy()]
        self.current_step = 0

        diff = self.cible_pos - drone_pos_array
        self.distance_precedente = np.linalg.norm(diff)
        
        return self._get_obs(), {}

    def step(self, action):
        # Initialisation reward (Coût de la vie/temps)
        reward = -0.1 
        terminated = False
        truncated = False

        # Sauvegarde la position AVANT mouvement
        prev_pos = np.array(self.drone_agent.get_position())

        # --- 1. Calcul de la vitesse voulue ---
        vitesse = self.drone_agent.get_vitesse()
        dx = action[0] * vitesse
        dy = action[1] * vitesse
        
        curr_x, curr_y = self.drone_agent.get_position()

        # --- 2. Physique de Glissement (Wall Sliding) ---
        test_rect_x = pygame.Rect((curr_x + dx) - 5, curr_y - 5, 10, 10)
        collision_x = any(obs.rect.colliderect(test_rect_x) for obs in self.obstacles)

        test_rect_y = pygame.Rect(curr_x - 5, (curr_y + dy) - 5, 10, 10)
        collision_y = any(obs.rect.colliderect(test_rect_y) for obs in self.obstacles)

        # Pénalité de collision (Moins violente qu'avant, c'est bien)
        if collision_x or collision_y:
            reward -= 1.0
        
        move_x = 0 if collision_x else dx
        move_y = 0 if collision_y else dy

        self.drone_agent.move_vector(move_x, move_y, self.largeur_carte, self.hauteur_carte)

        # Position APRÈS mouvement
        new_pos = np.array(self.drone_agent.get_position())

        # Calcul Proprioception (Vitesse ressentie)
        real_velocity = (new_pos - prev_pos) / self.drone_agent.get_vitesse()
        self.current_velocity = real_velocity

        vitesse_reelle_norm = np.linalg.norm(real_velocity)
        
        # ANTI-CAMPING : Si le drone fait du surplace (bloqué ou hésitant)
        # Seuil 0.05 car ta vitesse max est normalisée
        if vitesse_reelle_norm < 0.05:
            reward -= 0.5 # Aïe ! Bouge de là !

        drone_pos_array = np.array(self.drone_agent.get_position())

        # --- 3. LOGIQUE D'EXPLORATION (CORRIGÉE) ---
        # J'ai supprimé le bloc "Mise à jour plus fréquente" qui causait le bug.
        
        # On ne vérifie pas à chaque frame si la liste est vide (optimisation)
        if not self.zones_visitees:
            dist_last = 9999
        else:
            dist_last = np.linalg.norm(drone_pos_array - self.zones_visitees[-1])

        rayon_capteur = self.drone_agent.get_capteur().get_rayon() 
        
        # On ne lance le calcul coûteux que si on s'est éloigné de la dernière zone connue
        if dist_last > rayon_capteur:
            # On vérifie qu'on est loin de TOUTES les zones passées
            # (Note : Pour optimiser encore plus, tu pourrais ne vérifier que les 100 dernières)
            distances = [np.linalg.norm(drone_pos_array - z) for z in self.zones_visitees]
            
            if all(d > rayon_capteur for d in distances):
                # C'EST UNE NOUVELLE ZONE !
                reward += 0.5 
                
                # On l'ajoute à la mémoire MAINTENANT
                self.zones_visitees.append(drone_pos_array.copy())
                
                # Gestion mémoire
                if len(self.zones_visitees) > 1000:
                    self.zones_visitees.pop(0)

        # --- 4. Reward Shaping & Victoire ---
        # On utilise le 'CCD' (Continuous Collision Detection) pour ne pas rater la cible
        
        # Distance à l'arrêt (comme avant)
        dist_arret = np.linalg.norm(self.cible_pos - drone_pos_array)
        
        # Distance minimale durant le trajet (NOUVEAU)
        dist_trajet = dist_segment_point(prev_pos, drone_pos_array, self.cible_pos)
        
        # On gagne si on s'arrête dessus OU si on l'a traversée
        # (On prend le min des deux pour être sûr)
        dist_reelle = min(dist_arret, dist_trajet)

        # Guidage terminal (Chaud/Froid)
        # On utilise dist_arret pour le guidage car on veut qu'il s'arrête dessus idéalement
        if dist_arret <= rayon_capteur:
            if dist_arret < self.distance_precedente:
                bonus = 10.0 / (dist_arret + 1.0) 
                reward += bonus
            else:
                reward -= 0.1 
        
        self.distance_precedente = dist_arret

       # --- 5. PRIME DE VISÉE (Target Lock) 🔫 ---
        # On utilise dist_arret (calculé plus haut)
        if dist_arret <= rayon_capteur: 
            vec_cible = self.cible_pos - drone_pos_array
            vec_vitesse = np.array([dx, dy]) 
            
            norm_cible = np.linalg.norm(vec_cible)
            norm_vitesse = np.linalg.norm(vec_vitesse)
            
            if norm_cible > 0 and norm_vitesse > 0:
                # Produit scalaire
                alignement = np.dot(vec_cible / norm_cible, vec_vitesse / norm_vitesse)
                
                # --- CHANGEMENT ICI ---
                # 1. On récompense plus largement l'alignement (plus facile à déclencher)
                if alignement > 0.5:
                    # On regarde si la voie est libre devant (les rayons centraux du Lidar)
                    # Tes indices centraux dépendent de ton Capteur, disons qu'on prend le max de danger global
                    danger_max = max(self._get_obs()[4:20]) # Indices des lidars dans l'obs
                    
                    facteur = 1.0
                    if danger_max > 0.7: # Si un mur est tout proche
                        facteur = 0.0 # On coupe la prime de visée ! Il doit d'abord survivre.
                    
                    # On réduit aussi le 3.0 à 1.0 ou 1.5 pour qu'il soit moins obsédé
                    reward += alignement * 1.5 * facteur

        # VICTOIRE : On vérifie si on a touché la zone à un moment du trajet
        if dist_reelle <= self.rayon_capture:
            reward += 1000.0
            terminated = True

        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # 1. RADAR (C'est OK : un capteur radio peut donner la direction de la cible)
        # On garde le vecteur normalisé vers la cible, mais PAS la position absolue du drone.
        # Note : Si tu veux être HARDCORE, tu retires aussi ça et tu ne lui donnes
        # que la distance (signal fort/faible), mais ça rend la navigation très dure.
        # Gardons le vecteur directionnel local pour l'instant.
        
        drone_pos = np.array(self.drone_agent.get_position())
        diff = self.cible_pos - drone_pos
        dist = np.linalg.norm(diff)
        rayon_radar = self.drone_agent.get_capteur().get_rayon()
        
        radar_info = [0.0, 0.0] # dx, dy vers la cible (local)
        if dist <= rayon_radar:
            # AVANT (Erreur) : [diff[0]/rayon_radar, ...] -> Le signal faiblit en approchant
            # MAINTENANT : On normalise par la distance actuelle pour avoir un vecteur de taille 1
            if dist > 0:
                radar_info = [diff[0]/dist, diff[1]/dist]
            else:
                radar_info = [0.0, 0.0] # Sécurité division par zéro (sur la cible)
            
        # 2. LIDAR (Ta vision actuelle)
        def check_collision(rect_test):
            return any(obs.rect.colliderect(rect_test) for obs in self.obstacles)

        lidar_distances = self.drone_agent.capteur.scan_lidar(self.drone_agent.get_position(), check_collision)
        
        # IMPORTANT : On garde les distances brutes pour le dessin (render)
        self.last_sensors = lidar_distances 

        # --- AMÉLIORATION ICI ---
        # On inverse les valeurs pour le cerveau :
        # Distance 1.0 (Loin) -> Proximité 0.0 (Tout va bien)
        # Distance 0.0 (Proche) -> Proximité 1.0 (DANGER !)
        lidar_danger = [1.0 - d for d in lidar_distances]

        # 3. PROPRIOCEPTION
        if not hasattr(self, 'current_velocity'):
            self.current_velocity = np.array([0.0, 0.0])
        vel_x, vel_y = self.current_velocity

        # On donne 'lidar_danger' au lieu de 'lidar_distances' à l'IA
        obs = np.array([vel_x, vel_y] + radar_info + lidar_danger, dtype=np.float32)
        
        return obs

    def render(self):
        # (Ton render était bon, juste s'assurer d'utiliser self.drone_agent.get_position())
        if self.render_mode is None: return
        if self.screen is None or self.screen.get_width() != self.largeur_carte:
            pygame.init()
            self.screen = pygame.display.set_mode((self.largeur_carte, self.hauteur_carte))
            self.clock = pygame.time.Clock()

        self.screen.fill((20, 20, 20))
        for obs in self.obstacles:
            obs.draw(self.screen)
        
        # Récupération rayon depuis l'objet Capteur
        r = int(self.drone_agent.get_capteur().get_rayon())
        
        for centre in self.zones_visitees:
            s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (40, 40, 70, 128), (r, r), r)
            self.screen.blit(s, (centre[0]-r, centre[1]-r))

        pos_int = (int(self.drone_agent.get_x()), int(self.drone_agent.get_y()))
        
        # --- Visualization Lidar ---
        if hasattr(self, 'last_sensors'):
            for i, dist_norm in enumerate(self.last_sensors):
                dist_pixel = dist_norm * r # r est le rayon du capteur
                
                # On récupère la direction déjà normalisée depuis le capteur
                dx, dy = self.drone_agent.capteur.normalized_directions[i]
                
                # Plus besoin de normaliser ici !
                end_x = pos_int[0] + dx * dist_pixel
                end_y = pos_int[1] + dy * dist_pixel
                
                # Couleur: Rouge si obstacle touché (dist < 1.0), Vert si rien (dist == 1.0)
                # On met une tolérance car float
                color = (0, 255, 0) if dist_norm >= 0.99 else (255, 0, 0)
                
                pygame.draw.line(self.screen, color, pos_int, (end_x, end_y), 2)
        
        pygame.draw.circle(self.screen, (80, 80, 80), pos_int, r, 1) # Radar
        pygame.draw.circle(self.screen, (255, 0, 0), self.cible_pos.astype(int), 10) 
        pygame.draw.circle(self.screen, (0, 255, 0), pos_int, 8) # Drone
        
        pygame.display.flip()
        self.clock.tick(60)