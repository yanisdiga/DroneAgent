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
            # MODIFICATION 1 : MOINS DE PRESSION
            # On divise la punition par 10. Le drone a le droit de prendre son temps pour viser.
            "ENNUI_FACTOR": 0.005,  
            "ENNUI_CAP": 50,
            "DECAY_RATE": 0.998,  # (Quasi permanent) Vitesse de refroidissement des traces (plus c'est bas, plus ça disparait vite)
            "STUCK_THRESHOLD": 10, # Nombre de frames bloqué avant de secouer
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
        
        # --- MODIFICATION 2 : VITESSE ALÉATOIRE ---
        vitesse_aleatoire = np.random.uniform(3.0, 7.0)
        self.drone_agent.set_vitesse(vitesse_aleatoire)

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
        
        # --- NOUVELLE LOGIQUE EXPLORATION (HEATMAP) ---
        cell_size = 50 
        self.grid_w = (self.largeur_carte // cell_size) + 1
        self.grid_h = (self.hauteur_carte // cell_size) + 1
        
        # Grille de "Chaleur" (Float) au lieu de Bool
        self.heatmap = np.zeros((self.grid_w, self.grid_h), dtype=np.float32)

        self.stuck_counter = 0 # Compteur pour le Shake

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
        reward = -0.01
        terminated = False
        truncated = False

        # Sauvegarde la position AVANT mouvement
        prev_pos = np.array(self.drone_agent.get_position())

        # --- 1. REFROIDISSEMENT DE LA CARTE (Time Decay) ---
        # Les anciennes traces s'effacent petit à petit (Mémoire organique).
        self.heatmap *= self.CONFIG["DECAY_RATE"]

        # --- 2. CALCUL DU MOUVEMENT & GESTION "STUCK" ---
        vitesse = self.drone_agent.get_vitesse()
        dx = action[0] * vitesse
        dy = action[1] * vitesse
        
        curr_x, curr_y = self.drone_agent.get_position()

        # Test de collision prédictif (Pour le Wall Sliding et le Stuck Counter)
        test_rect_x = pygame.Rect((curr_x + dx) - 5, curr_y - 5, 10, 10)
        col_x = any(obs.rect.colliderect(test_rect_x) for obs in self.obstacles)
        
        test_rect_y = pygame.Rect(curr_x - 5, (curr_y + dy) - 5, 10, 10)
        col_y = any(obs.rect.colliderect(test_rect_y) for obs in self.obstacles)
        
        # Gestion du compteur de blocage
        if col_x or col_y:
            reward -= 0.05 # Petite punition collision

        # --- LE "SHAKE" (SECOUSSE) 🌪️ ---
        # Si bloqué depuis X frames, on ignore le réseau de neurones et on force un mouvement aléatoire
        if self.stuck_counter > self.CONFIG["STUCK_THRESHOLD"]:
            # On secoue fort (bruit aléatoire pur) pour se décoincer
            dx += np.random.uniform(-vitesse, vitesse) * 2.0
            dy += np.random.uniform(-vitesse, vitesse) * 2.0
            # Punition pour inciter à ne pas se retrouver coincé
            reward -= 0.1

        # Application physique (Wall Sliding)
        # Si collision, on annule le mouvement sur l'axe concerné
        move_x = 0 if col_x else dx
        move_y = 0 if col_y else dy

        # EXCEPTION : Si on est en mode "Shake", on force le mouvement même si collision détectée
        # (L'objet Drone.py gère le clamping dans la carte, donc il ne sortira pas)
        if self.stuck_counter > self.CONFIG["STUCK_THRESHOLD"]:
            move_x = dx
            move_y = dy

        self.drone_agent.move_vector(move_x, move_y, self.largeur_carte, self.hauteur_carte)

        # --- 3. PROPRIOCEPTION & ANTI-CAMPING ---
        new_pos = np.array(self.drone_agent.get_position())
        dist_parcourue = np.linalg.norm(new_pos - prev_pos)

        # On considère qu'on est "Stuck" SEULEMENT si on a parcouru moins de 0.5 pixel
        # Même si on touche un mur (col_x=True), tant qu'on avance (sliding), on n'est PAS stuck.
        if dist_parcourue < 0.5:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0 # On bouge, tout va bien !
        
        # Vitesse ressentie
        real_velocity = (new_pos - prev_pos) / self.drone_agent.get_vitesse()
        self.current_velocity = real_velocity
        vitesse_reelle_norm = np.linalg.norm(real_velocity)
        
        # Si le drone fait du surplace (hors mode Shake)
        if vitesse_reelle_norm < 0.05 and self.stuck_counter < self.CONFIG["STUCK_THRESHOLD"]:
            reward -= 0.5 # Aïe ! Bouge de là !

       # --- 4. EXPLORATION INTELLIGENTE (HEATMAP) ---
        grid_x = int(new_pos[0] // 50)
        grid_y = int(new_pos[1] // 50)
        
        if 0 <= grid_x < self.grid_w and 0 <= grid_y < self.grid_h:
            current_heat = self.heatmap[grid_x, grid_y]
            
            # Formule : On récompense la nouveauté, mais on ne punit plus le passage !
            # Si heat est 0.0 -> Reward +0.5
            # Si heat est 1.0 -> Reward 0.0 (Neutre)
            reward_explo = (1.0 - current_heat) * 0.5
            
            if reward_explo > 0.05: 
                reward += reward_explo # C'est frais, bravo !
                self.steps_since_discovery = 0 
            else:
                # C'est une zone déjà visitée.
                # ON NE PUNIT PAS DIRECTEMENT (pour lui permettre de traverser un couloir connu)
                # La seule punition est celle du temps qui passe (-0.01 par step)
                self.steps_since_discovery += 1 
            
            # On marque le territoire (On remet la chaleur à 1.0)
            self.heatmap[grid_x, grid_y] = 1.0
        else:
            self.steps_since_discovery += 1 # Hors map (ne devrait pas arriver)

        # PUNITIF PROGRESSIF : Si ça fait trop longtemps qu'il n'a rien découvert
        if self.steps_since_discovery > 150:
             reward -= 0.05 # Là on commence à s'énerver : "Bouge de là !"

        # --- 5. LOGIQUE CIBLE & VICTOIRE (TARGET HANDLING) ---
        drone_pos_array = new_pos
        rayon_capteur = self.drone_agent.get_capteur().get_rayon()
        
        # Distances
        dist_arret = np.linalg.norm(self.cible_pos - drone_pos_array)
        
        # On vérifie si on voit la cible
        target_detected = (dist_arret <= rayon_capteur)
        
        if target_detected:
            # --- A. L'EFFET AIMANT & GUIDAGE ---
            # 1. Magnetism : Dès qu'il la voit, il gagne des points pour rester
            reward += 2.0 
            
            # 2. Approche Agressive vs Recul
            if dist_arret < self.distance_precedente:
                # Bonus exponentiel pour l'approche
                bonus = 30.0 / (dist_arret + 1.0) 
                reward += bonus
            else:
                # INTERDICTION DE RECULER si on voit la cible
                reward -= 5.0
            
            # --- B. PRIME DE VISÉE (HUNTER MODE) ---
            vec_cible = self.cible_pos - drone_pos_array
            vec_vitesse = np.array([dx, dy]) 
            norm_cible = np.linalg.norm(vec_cible)
            norm_vitesse = np.linalg.norm(vec_vitesse)
            
            if norm_cible > 0 and norm_vitesse > 0:
                alignement = np.dot(vec_cible / norm_cible, vec_vitesse / norm_vitesse)
                
                if alignement > 0.5:
                    # On vérifie s'il y a du danger proche (pour récompenser l'audace)
                    # Note: on récupère les capteurs via scan_lidar direct pour éviter boucle infinie _get_obs
                    # Mais pour simplifier ici, on suppose que l'audace est constante ou on garde ta logique
                    reward += alignement * 5.0 
                    reward += (rayon_capteur - dist_arret) * 0.1 # Bonus proximité
        
        self.distance_precedente = dist_arret

        # --- 6. VICTOIRE STABILISÉE ---
        # On ne gagne plus instantanément, il faut rester dessus
        
        if dist_arret < self.CONFIG["RAYON_CAPTURE"]:
            self.steps_on_target += 1
            reward += 1.0 # Félicitations pour la stabilité
        else:
            self.steps_on_target = 0 # Reset s'il sort de la zone
            
        # La victoire n'arrive que s'il est resté stable 30 frames (0.5 seconde)
        if self.steps_on_target > 30:
            reward += 1000.0
            terminated = True
            print(f"🎯 CIBLE CAPTURÉE ET STABILISÉE au step {self.current_step}!")

        # --- 7. TIMEOUTS ---
        # Patience (si pas d'exploration depuis longtemps)
        if self.steps_since_discovery >= self.CONFIG["PATIENCE"]:
            truncated = True 
            reward -= 5.0 # Punition "Tu es trop lent/bloqué"

        self.current_step += 1
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
        
        # --- CAPTEURS D'EXPLORATION (Heatmap Sensors) ---
        curr_x, curr_y = self.drone_agent.get_position()
        gx = int(curr_x // 50)
        gy = int(curr_y // 50)
        
        explo_sensors = []
        offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)] # N, S, O, E
        
        for dx, dy in offsets:
            nx, ny = gx + dx, gy + dy
            if not (0 <= nx < self.grid_w and 0 <= ny < self.grid_h):
                explo_sensors.append(1.0) # Mur = Comme si c'était visité (repoussant)
            else:
                # On lit la Heatmap (float) :
                # 0.9 = Je viens de passer là (ne pas y aller)
                # 0.1 = Je suis passé il y a longtemps (ok pour y retourner)
                # 0.0 = Jamais visité (Fonce !)
                explo_sensors.append(self.heatmap[nx, ny])

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
            
            # --- Visualisation Heatmap (Olfactive) ---
            # On dessine des carrés rouges là où c'est "Chaud"
            for x in range(self.grid_w):
                for y in range(self.grid_h):
                    heat = self.heatmap[x, y]
                    if heat > 0.01: # Si la case est un peu chaude
                        # Couleur : Rouge transparent selon l'intensité
                        # heat va de 0 à 1. Alpha max 150 pour voir à travers.
                        alpha = int(heat * 150)
                        s = pygame.Surface((50, 50), pygame.SRCALPHA)
                        s.fill((255, 50, 50, alpha)) # Rouge
                        self.screen.blit(s, (x * 50, y * 50))

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