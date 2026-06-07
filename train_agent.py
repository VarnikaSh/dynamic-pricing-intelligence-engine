import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from stable_baselines3 import PPO
from pricing_env import(
    MultiProductPricingEnv
)

env = MultiProductPricingEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)
model.learn(
    total_timesteps=100000
)
model.save(
    "pricing_agent_v35"
)
print(
    "Training Complete"
)