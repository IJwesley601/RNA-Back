from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- FastAPI init ---
app = FastAPI(title="API Estimation Immo")
# --- Données d'entrée ---
class PropertyData(BaseModel):
    surface: float; rooms: int; bedrooms: int; bathrooms: int; year: int; floor: int
    condition: str; property_type: str; parking: str;latitude: float; longitude: float
# --- Estimator avec RandomForest ---
class RealEstateEstimator:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        self.is_trained = False

    def train_model(self):
        np.random.seed(42)
        n = 10000
        X = np.column_stack([
            np.random.normal(80, 30, n), np.random.randint(1, 6, n), np.random.randint(0, 4, n),  np.random.randint(1, 3, n), 
            np.random.randint(1950, 2024, n), np.random.randint(0, 10, n), np.random.randint(1, 5, n), np.random.randint(0, 5, n),  
            np.random.randint(0, 4, n), np.random.normal(48.8566, 0.1, n),  np.random.normal(2.3522, 0.1, n) 
        ])
        y = X[:, 0] * 5000 + X[:, 1] * 10000 + X[:, 5] * 3000 + np.random.normal(0, 50000, n)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, data: PropertyData):
        if not self.is_trained:
            self.train_model()

        condition_map = {'excellent': 4, 'good': 3, 'average': 2, 'renovation': 1}, type_map = {'apartment': 1, 'house': 2, 'studio': 0, 'loft': 3, 'duplex': 4}
        parking_map = {'none': 0, 'street': 1, 'covered': 2, 'garage': 3}
        features = np.array([[
            data.surface, data.rooms, data.bedrooms, data.bathrooms,data.year,data.floor, condition_map.get(data.condition, 2),
            type_map.get(data.property_type, 1), parking_map.get(data.parking, 0), data.latitude, data.longitude
        ]]), features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        return {"estimated_price": round(prediction)}
# --- API Endpoint ---
estimator = RealEstateEstimator()
@app.post("/estimate")
def estimate(data: PropertyData):
    if data.surface <= 0 or data.rooms <= 0:
        raise HTTPException(status_code=400, detail="Paramètres invalides.")
    return estimator.predict(data)
