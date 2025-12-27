# test.py
import pygame
import numpy as np
from sb3_contrib import RecurrentPPO
from drone_env import DroneEnv
from Drone import Drone
from Capteur import Capteur

capteur = Capteur(rayon=80)
drone = Drone(id=1, x=0, y=0, capteur=capteur, vitesse=3)
env = DroneEnv(drone, capteur, render_mode="human")

model = "drone_model_final" # // drone_model_final

print("Chargement du modèle : " + model + " ...")
try:
    model = RecurrentPPO.load(model)
    print("Modèle chargé ! 🧠")
except:
    print("Erreur : Modèle : "+ model +" non trouvé. Vérifie le nom du fichier.")
    exit()

# --- SÉLECTION DU NIVEAU ---
# 0: Vide (Murs uniquement)
# 1: Facile (Quelques obstacles)
# 2: Moyen (Obstacles standards)
# 3: Difficile (Beaucoup d'obstacles)
# R: Random"

difficulty_option = 2

episodes = 5
lstm_states = None 

for ep in range(episodes):
    print(f"--- Épisode {ep + 1} ---")

    if difficulty_option is not None:
        obs, _ = env.reset(options={"difficulty": difficulty_option})
    else:
        obs, _ = env.reset() # Random par défaut
        
    terminated = False
    truncated = False
    score = 0
    lstm_states = None # Reset mémoire
    
    while not terminated and not truncated:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                episodes = 0
                break # Sortie propre

    print(f"Score: {score:.2f}")

pygame.quit()