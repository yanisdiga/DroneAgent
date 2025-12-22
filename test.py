# test.py
import pygame
import numpy as np
from sb3_contrib import RecurrentPPO # Import Contrib
from drone_env import DroneEnv
from Drone import Drone
from Capteur import Capteur

capteur = Capteur(rayon=80)
drone = Drone(id=1, x=0, y=0, capteur=capteur, vitesse=3)
env = DroneEnv(drone, capteur, render_mode="human")

print("Chargement du modèle LSTM...")
model = RecurrentPPO.load("drone_model_final") # Charge le bon fichier
print("Modèle chargé ! 🧠")

obs, _ = env.reset()

# INITIALISATION DE LA MÉMOIRE DU DRONE
# Le LSTM a besoin d'un état initial (vide au début)
# (1 = nb_envs, 2*lstm_hidden_size pour le LSTM state)
# Le plus simple est de laisser None au début, ou :
lstm_states = None 
num_envs = 1 
# Si tu veux être propre : lstm_states = np.zeros((num_envs, 2 * 64)) mais None marche souvent

episodes = 5

for ep in range(episodes):
    print(f"--- Épisode {ep + 1} ---")
    terminated = False
    truncated = False
    score = 0
    
    # Reset de la mémoire à chaque nouvel épisode ! (Important)
    lstm_states = None
    
    while not terminated and not truncated:
        # IMPORTANT : On passe 'lstm_states' et on récupère le NOUVEAU 'lstm_states'
        # C'est comme ça que le drone se souvient du passé.
        # On utilise deterministic=True pour voir le "vrai" comportement appris (sans bruit aléatoire)
        action, _states = model.predict(obs, state=lstm_states, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        
        env.render()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                episodes = 0

    print(f"Score: {score:.2f}")
    obs, _ = env.reset()

pygame.quit()