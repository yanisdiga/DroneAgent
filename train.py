import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from drone_env import DroneEnv
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# --- Callback pour voir la progression ---
class ScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ScoreCallback, self).__init__(verbose)
        self.best_reward = -float('inf')
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Avec le multi-processing, 'infos' contient les infos de TOUS les environnements (8)
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_count += 1
                    current_reward = info["episode"]["r"]
                    print(f"Épisode {self.episode_count} fini | Score: {current_reward:.2f}")
                    
                    if current_reward > self.best_reward:
                        self.best_reward = current_reward
                        self.model.save("drone_model_best") 
                        print(f"🔥 NEW RECORD: {self.best_reward:.2f} (Épisode {self.episode_count})")
                    
                    # Petit log périodique pour savoir que ça tourne
                    elif self.episode_count % 100 == 0:
                        print(f"Épisode {self.episode_count} | Score récent: {current_reward:.2f}")
                        
        return True

if __name__ == '__main__': # Indispensable sous Windows
    # 1. CRÉATION PARALLÈLE (8 environnements)
    # make_vec_env s'occupe de créer les 8 instances, de mettre les Monitor, 
    # et de les emballer dans un SubprocVecEnv.
    env = make_vec_env(DroneEnv, n_envs=8, vec_env_cls=SubprocVecEnv)

    # 2. EMPILAGE D'IMAGES (Mémoire)
    # On applique directement le FrameStack sur l'environnement vectorisé
    env = VecFrameStack(env, n_stack=4)
    
    # 3. CRÉATION DU MODÈLE
    # ent_coef=0.05 est très bien pour forcer l'exploration
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0, 
        device="cpu", 
        ent_coef=0.05, 
        learning_rate=0.0003, 
        tensorboard_log="./drone_tensorboard/"
    )

    print("🚀 Début de l'entraînement TURBO (8 Cœurs + Mémoire 4 frames)...")
    callback = ScoreCallback()

    # 4. LANCEMENT
    # Note : 500 000 steps total divisé par 8 envs = 62 500 steps par env.
    # C'est très rapide. Tu peux monter à 1 000 000 ou 2 000 000 si besoin.
    model.learn(total_timesteps=1000000, callback=callback)

    # 5. SAUVEGARDE FINALE
    model.save("drone_model_final")
    env.close() # Toujours propre de fermer les processus
    print(f"✅ Entraînement fini ! Modèle final enregistré.")