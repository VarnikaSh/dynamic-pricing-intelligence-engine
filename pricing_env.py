import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MultiProductPricingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.products = pd.read_csv(
            "products.csv"
        )
        self.n_products = len(
            self.products
        )
        self.action_space = spaces.MultiDiscrete(
            [20] * self.n_products
        )
        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(self.n_products * 4,),
            dtype=np.float32
        )
    def reset(self, seed=None, options=None):
        self.revenue_history = []
        super().reset(seed=seed)
        self.day = 0
        self.inventory = (
            self.products["inventory"]
            .values
            .astype(float)
        )
        return self._get_state(), {}
    def _get_state(self):
        state = []
        for i in range(self.n_products):
            state.extend([
                self.inventory[i],
                self.products.iloc[i]["base_price"],
                self.products.iloc[i]["base_demand"],
                self.day
            ])
        return np.array(
            state,
            dtype=np.float32
        )
    def step(self, action):
        total_profit = 0
        for i in range(self.n_products): 
            price = 10 + action[i] * 5
            base_demand = (
                self.products.iloc[i]
                ["base_demand"]
            )
            competitor_price = (
                price
                +
                np.random.randint(
                    -5,
                    5
                )
            )
            customer_type = np.random.choice(
                [
                    "budget",
                    "premium",
                    "loyal"
                ]
            )
            season_factor = 1 + 0.3 * np.sin(self.day / 30)
            holiday_boost = 1
            elasticity = np.random.uniform(
                0.8,
                2.0
            )
            if np.random.random() < 0.05:
                holiday_boost = 1.5
            demand = (
                base_demand *
                season_factor *
                holiday_boost *
                (
                    competitor_price /
                    max(price, 1)
                )** elasticity
            )
            if customer_type == "budget":
                demand *= 1.3
            elif customer_type == "premium":
                demand *= 0.8
            elif customer_type == "loyal":
                demand *= 1.1 
            trend_factor = (
                1 +
                0.2 * 
                np.sin(
                    self.day / 50
                )
            )          
            demand *= trend_factor 
            demand = max(
                demand,
                0
            )
            if np.random.random() < 0.03:
                self.inventory[i] *= 0.8
            sales = min(
                demand,
                self.inventory[i]
            )
            revenue = (
                sales * price
            )
            self.inventory[i] -= sales
            if self.inventory[i] < 50:
                self.inventory[i] += 300
                reorder_cost = 200
            else:
                reorder_cost = 0     
            self.revenue_history.append(
                revenue
            )      
            inventory_cost = (
                self.inventory[i]
                * 0.1          
            )
            shortage_cost = 0
            if self.inventory[i] < 20:
                shortage_cost = 100
            profit = (
                revenue
                - inventory_cost
                -shortage_cost
                - reorder_cost
            )    
            total_profit += profit

        self.day += 1
        done = (
            self.day >= 365
        )   
        return(
            self._get_state(),
            total_profit,
            done,
            False,
            {}
        ) 