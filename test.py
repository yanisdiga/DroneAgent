import pygame
from stable_baselines3 import PPO
from drone_env import DroneEnv

# 1. Charger l'environnement en mode VISUEL
env = DroneEnv(render_mode="human")

# 2. Charger le cerveau qu'on vient d'entraîner
# (Assure-toi que le fichier "drone_model_v1.zip" est bien dans le dossier)
print("Chargement du modèle...")
try:
    model = PPO.load("drone_model_final")
    print("Modèle chargé avec succès ! 🧠")
except:
    print("Erreur : Fichier 'drone_model_final.zip' introuvable.")
    exit()

# 3. Lancer la boucle de démonstration
obs, _ = env.reset()
episodes = 5

for ep in range(episodes):
    print(f"--- Lancement Épisode {ep + 1} ---")
    terminated = False
    truncated = False
    score = 0
    
    while not terminated and not truncated:
        # L'IA prédit l'action (deterministic=True signifie qu'elle ne teste plus, elle joue le mieux possible)
        action, _states = model.predict(obs, deterministic=True)
        
        # On applique l'action
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        
        # Affichage
        env.render()
        
        # Gestion de la fermeture fenêtre pour éviter le crash
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                truncated = True
                episodes = 0 # Stop tout

    print(f"Épisode terminé. Score final : {score:.2f}")
    obs, _ = env.reset()

print("Démonstration terminée.")
pygame.quit()