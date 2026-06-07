import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI
from stable_baselines3 import PPO
import numpy as np

app = FastAPI()

model = PPO.load(
    "pricing_agent_v35"
)

@app.get("/recommend_price")

def recommend_price(
    inventory: int,
    competitor_price: int,
    demand: int
):
    state = np.array([
        inventory,
        competitor_price,
        demand,
        1.0,
        30
    ])
    action, _ = model.predict(
        state,
        deterministic=True
    )
    price = 10 + action * 5
    return{
        "recommended_price":
        int(price)
    }