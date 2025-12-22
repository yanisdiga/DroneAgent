import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from Obstacle import Obstacle
from Drone import Drone
from Capteur import Capteur
import Utils as utils

class DroneEnv(gym.Env):
    def __init__(self, drone, capteur, render_mode=None):
        '''
        Constructeur de l'environnement
        '''
        super(DroneEnv, self).__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # --- CONFIGURATION CENTRALE ---
        self.CONFIG = {
            "MAP_SIZE_MIN": (800, 600),
            "MAP_SIZE_MAX": (1200, 800),
            "MAX_STEPS": 2000,
            "PATIENCE": 500,
            "RAYON_LIDAR": capteur.rayon,
            "RAYON_CAPTURE": 20,
            "VITESSE": drone.vitesse,
            "REWARD_TARGET": 1000.0,
            "REWARD_EXPLORATION": 0.5,
            "ENNUI_FACTOR": 0.01,  # Combien ça fait mal par step
            "ENNUI_CAP": 50,       # Plafond du multiplicateur (Max -0.5)
        }

        self.largeur_carte, self.hauteur_carte = self.CONFIG["MAP_SIZE_MIN"]
        self.max_steps = self.CONFIG["MAX_STEPS"]
        
        # --- Intégration de tes Classes POO ---
        # On crée un Capteur avec le rayon défini
        self.mon_capteur = capteur
        # On crée le Drone (position temporaire 0,0)
        self.drone_agent = drone
        
        self.obstacles = []
        self.rayon_capture = self.CONFIG["RAYON_CAPTURE"]

        # --- Espaces d'Action et d'Observation ---
        # 2 Vitesse + 2 Radar + 1 Proximité + 16 Lidar + 4 Exploration = 25
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(25,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation identique
        low_obs = np.array([-1.0]*25, dtype=np.float32) # Correction: -1.0 car vitesse/radar peuvent être négatifs
        high_obs = np.array([1.0]*25, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        '''
        Fonction de reset de l'environnement permettant de réinitialiser l'état de l'environnement
        '''
        super().reset(seed=seed)

        niveau = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]) 
        
        if niveau == 1:
            self.obstacles = []
        elif niveau == 2:
            self.obstacles = utils.create_random_obstacles((self.largeur_carte, self.hauteur_carte), np.random.randint(1, 3))
        elif niveau == 3:
            self.obstacles = utils.create_random_obstacles((self.largeur_carte, self.hauteur_carte), np.random.randint(3, 5))

        # 1. Map aléatoire
        self.largeur_carte, self.hauteur_carte = utils.random_map_size()
        
        # 2. Obstacles aléatoires
        self.obstacles = utils.create_random_obstacles((self.largeur_carte, self.hauteur_carte), np.random.randint(3, 7))
        self.obstacles.extend(utils.add_walls(self.largeur_carte, self.hauteur_carte, 20))
        
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
        
        # --- INITIALISATION DE LA GRILLE D'EXPLORATION ---
        # Taille des cellules pour le 'Fog of War'
        cell_size = 50 
        self.grid_w = (self.largeur_carte // cell_size) + 1
        self.grid_h = (self.hauteur_carte // cell_size) + 1
        self.visited_grid = np.zeros((self.grid_w, self.grid_h), dtype=bool)

        # On récupère la pos du drone sous forme d'array numpy pour les calculs mathématiques
        drone_pos_array = np.array(self.drone_agent.get_position())
        self.zones_visitees = [drone_pos_array.copy()]

        self.current_step = 0
        self.steps_since_discovery = 0
        self.consecutive_visited_steps = 0

        diff = self.cible_pos - drone_pos_array
        self.distance_precedente = np.linalg.norm(diff)
        
        return self._get_obs(), {}

    def step(self, action):
        '''
        Fonction de step de l'environnement permettant de mettre à jour l'état de l'environnement
        '''
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
        rayon_capteur = self.drone_agent.get_capteur().get_rayon() 

        # --- 3. LOGIQUE D'EXPLORATION (GRID / FOG OF WAR) ---
        # On convertit la position en coordonnées de grille (Cellule de 50x50 pixels)

        grid_x = int(curr_x // 50)
        grid_y = int(curr_y // 50)
        
        is_new_cell = False # Drapeau pour savoir si on a exploré ce tour-ci

        grid_x = int(curr_x // 50)
        grid_y = int(curr_y // 50)
        
        is_new_cell = False 

        if 0 <= grid_x < self.grid_w and 0 <= grid_y < self.grid_h:
            if not self.visited_grid[grid_x, grid_y]:
                # --- DÉCOUVERTE (C'est la fête !) ---
                self.visited_grid[grid_x, grid_y] = True
                reward += 0.5 
                is_new_cell = True 
                self.zones_visitees.append(drone_pos_array.copy())
                
                # Le drone est content, il oublie son ennui
                self.consecutive_visited_steps = 0 
            else:
                # --- DÉJÀ VU (La pression monte) ---
                self.consecutive_visited_steps += 1
                
                # Calcul de la pénalité progressive
                # On cape le multiplicateur pour éviter le suicide (Max 50)
                facteur = min(self.consecutive_visited_steps, self.CONFIG["ENNUI_CAP"])
                
                # Pénalité = -0.01 * facteur
                # Ex: au bout de 50 steps, il perd -0.5 par mouvement !
                reward -= self.CONFIG["ENNUI_FACTOR"] * facteur
        
        if is_new_cell:
            self.steps_since_discovery = 0 # <-- On reset le compteur, bravo !
        else:
            self.steps_since_discovery += 1 # <-- On s'impatiente...
        # --- 4. Reward Shaping & Victoire ---
        # On utilise le 'CCD' (Continuous Collision Detection) pour ne pas rater la cible
        
        # Distance à l'arrêt (comme avant)
        dist_arret = np.linalg.norm(self.cible_pos - drone_pos_array)
        
        # Distance minimale durant le trajet (NOUVEAU)
        dist_trajet = utils.dist_segment_point(prev_pos, drone_pos_array, self.cible_pos)
        
        # On gagne si on s'arrête dessus OU si on l'a traversée
        # (On prend le min des deux pour être sûr)
        dist_reelle = min(dist_arret, dist_trajet)

        # Guidage terminal (Chaud/Froid)
        # On utilise dist_arret pour le guidage car on veut qu'il s'arrête dessus idéalement
        if dist_arret <= rayon_capteur:
            
            # 1. L'EFFET AIMANT (MAGNETISM)
            # Dès qu'il la voit, il gagne des points juste pour rester à proximité.
            # C'est supérieur au bonus d'exploration (+0.5), donc il ne voudra plus partir.
            reward += 2.0 
            
            # 2. APPROCHE AGRESSIVE
            if dist_arret < self.distance_precedente:
                # On booste le gain quand il s'approche
                # Formule exponentielle : plus il est près, plus ça rapporte
                bonus = 30.0 / (dist_arret + 1.0) 
                reward += bonus
            else:
                # 3. INTERDICTION DE RECULER
                # C'est ICI que tout change.
                # Avant, tu avais : "if not is_new_cell: reward -= 0.1"
                # Maintenant : ON PUNIT TOUT LE TEMPS.
                # Même s'il y a une case inexplorée derrière lui, s'il recule alors qu'il voit la cible : PUNITION.
                reward -= 5.0
        
        self.distance_precedente = dist_arret

        # --- 5. PRIME DE VISÉE (Target Lock) 🔫 ---
        # On utilise dist_arret (calculé plus haut)
        
        target_detected = (dist_arret <= rayon_capteur)
        
        if target_detected:
            # --- HUNTER MODE ACTIF 🦁 ---
            if is_new_cell:
                # ANNULATION du bonus d'exploration : On ne veut pas qu'il soit distrait !
                reward -= 0.5 
            
            vec_cible = self.cible_pos - drone_pos_array
            vec_vitesse = np.array([dx, dy]) 
            
            norm_cible = np.linalg.norm(vec_cible)
            norm_vitesse = np.linalg.norm(vec_vitesse)
            
            if norm_cible > 0 and norm_vitesse > 0:
                # Produit scalaire
                alignement = np.dot(vec_cible / norm_cible, vec_vitesse / norm_vitesse)
                
                # --- CHANGEMENT ICI ---
                # On recompense massivement l'alignement quand on est proches
                if alignement > 0.5:
                    danger_max = max(self._get_obs()[4:20]) 
                    
                    facteur = 1.0
                    if danger_max > 0.85: 
                        facteur = 0.5 # On est un peu plus téméraire (0.2 -> 0.5)
                    
                    # BOOST MASSIF : 1.5 -> 5.0
                    reward += alignement * 5.0 * facteur
                    
                    # BONUS DE PROXIMITÉ (Plus on est près, plus c'est rentable)
                    # Ex: à 10px -> (80 - 10) * 0.1 = +7 points !
                    reward += (rayon_capteur - dist_arret) * 0.1

        # VICTOIRE : On vérifie si on a touché la zone à un moment du trajet
        if dist_reelle <= self.rayon_capture:
            reward += 1000.0
            terminated = True

        # --- AJOUT : ARRÊT PRÉMATURÉ (TIMEOUT DE PATIENCE) ---
        if self.steps_since_discovery >= self.CONFIG["PATIENCE"]:
            truncated = True # On coupe l'épisode
            reward -= 5.0 # Petite punition pour dire "Tu étais trop lent/bloqué"

        self.current_step += 1
        # Max steps arret
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        '''
        Fonction de récupération des observations
        '''
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

        lidar_distances = self.drone_agent.capteur.scan_lidar(self.drone_agent.get_position(), self.obstacles)
        
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
        
        # --- NOUVEAU : CAPTEURS D'EXPLORATION (4 valeurs) ---
        curr_x, curr_y = self.drone_agent.get_position()
        gx = int(curr_x // 50) # Coordonnée grille actuelle
        gy = int(curr_y // 50)
        
        # On vérifie les 4 voisins (Haut, Bas, Gauche, Droite)
        # 0.0 = Inconnu (Bon), 1.0 = Déjà visité (Ennuyeux) ou Hors Map (Mur)
        explo_sensors = []
        
        offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Nord, Sud, Ouest, Est
        
        for dx, dy in offsets:
            nx, ny = gx + dx, gy + dy
            
            # Si hors de la carte, on considère comme "visité" pour ne pas qu'il y aille
            if not (0 <= nx < self.grid_w and 0 <= ny < self.grid_h):
                explo_sensors.append(1.0)
            # Sinon, on regarde si c'est True (visité) ou False (nouveau)
            elif self.visited_grid[nx, ny]:
                explo_sensors.append(1.0)
            else:
                explo_sensors.append(0.0) # C'est nouveau !

        # Assemblage final (25 valeurs)
        # Ajout du Capteur de Proximité Radar (0.0 = Loin/Pas vu, 1.0 = Dessus)
        radar_proximity = 0.0
        if dist <= rayon_radar:
            radar_proximity = max(0.0, 1.0 - (dist / rayon_radar))
            
        obs = np.array([vel_x, vel_y] + radar_info + [radar_proximity] + lidar_danger + explo_sensors, dtype=np.float32)
        
        return obs

    def render(self):
        '''
        Fonction de render de l'environnement permettant l'affichage
        de la carte et des obstacles avec pygame.
        '''
        if self.render_mode == "human":
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
        else:
            return