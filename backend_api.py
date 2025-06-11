"""
Backend Python pour l'estimation immobilière
FastAPI + Machine Learning
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import uvicorn
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="EstimPro API",
    description="API d'estimation immobilière avec IA",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles de données Pydantic
class PropertyData(BaseModel):
    address: str
    property_type: str
    surface: float
    rooms: int
    bedrooms: int
    bathrooms: int
    year: int
    condition: str
    parking: str
    garden: str
    balcony: Optional[str] = None
    floor: Optional[int] = None
    elevator: Optional[bool] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class EstimationResponse(BaseModel):
    estimated_price: float
    price_min: float
    price_max: float
    price_per_sqm: float
    confidence_score: float
    market_trends: List[dict]
    comparable_properties: List[dict]
    factors_analysis: dict

# Classe pour le modèle d'estimation
class RealEstateEstimator:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'surface', 'rooms', 'bedrooms', 'bathrooms', 'year',
            'condition_encoded', 'property_type_encoded', 'parking'
            'condition_encoded', 'property_type_encoded', 'parking_encoded',
            'garden_encoded', 'latitude', 'longitude', 'district_score'
        ]
        self.is_trained = False
        
    def preprocess_data(self, data: PropertyData) -> np.ndarray:
        """Préprocessing des données d'entrée"""
        # Encodage des variables catégorielles
        condition_mapping = {'excellent': 4, 'good': 3, 'average': 2, 'renovation': 1}
        property_type_mapping = {'apartment': 1, 'house': 2, 'studio': 0, 'loft': 3, 'duplex': 4}
        parking_mapping = {'none': 0, 'street': 1, 'covered': 2, 'garage': 3}
        garden_mapping = {'none': 0, 'balcony': 1, 'terrace': 2, 'garden': 3}
        
        # Calcul du score de quartier basé sur la localisation (simulation)
        district_score = self.calculate_district_score(data.latitude, data.longitude)
        
        features = np.array([
            data.surface,
            data.rooms,
            data.bedrooms,
            data.bathrooms,
            data.year,
            condition_mapping.get(data.condition, 2),
            property_type_mapping.get(data.property_type, 1),
            parking_mapping.get(data.parking, 0),
            garden_mapping.get(data.garden, 0),
            data.latitude or 48.8566,  # Paris par défaut
            data.longitude or 2.3522,
            district_score
        ]).reshape(1, -1)
        
        return features
    
    def calculate_district_score(self, lat: Optional[float], lon: Optional[float]) -> float:
        """Calcule un score de quartier basé sur la localisation"""
        if not lat or not lon:
            return 5.0  # Score moyen par défaut
        
        # Simulation d'un score basé sur la proximité du centre de Paris
        paris_center_lat, paris_center_lon = 48.8566, 2.3522
        distance = np.sqrt((lat - paris_center_lat)**2 + (lon - paris_center_lon)**2)
        
        # Score inversement proportionnel à la distance (0-10)
        score = max(0, 10 - distance * 100)
        return min(10, score)
    
    def train_model(self):
        """Entraîne le modèle avec des données simulées"""
        logger.info("Entraînement du modèle d'estimation...")
        
        # Génération de données d'entraînement simulées
        np.random.seed(42)
        n_samples = 10000
        
        # Features simulées
        surface = np.random.normal(80, 30, n_samples)
        rooms = np.random.randint(1, 6, n_samples)
        bedrooms = np.random.randint(0, 4, n_samples)
        bathrooms = np.random.randint(1, 3, n_samples)
        year = np.random.randint(1950, 2024, n_samples)
        condition = np.random.randint(1, 5, n_samples)
        property_type = np.random.randint(0, 5, n_samples)
        parking = np.random.randint(0, 4, n_samples)
        garden = np.random.randint(0, 4, n_samples)
        latitude = np.random.normal(48.8566, 0.1, n_samples)
        longitude = np.random.normal(2.3522, 0.1, n_samples)
        district_score = np.random.uniform(3, 9, n_samples)
        
        X = np.column_stack([
            surface, rooms, bedrooms, bathrooms, year,
            condition, property_type, parking, garden,
            latitude, longitude, district_score
        ])
        
        # Prix simulé avec une formule réaliste
        base_price = (surface * 5000 +  # Prix de base par m²
                     rooms * 10000 +     # Bonus par pièce
                     condition * 15000 + # Bonus état
                     district_score * 8000 + # Bonus quartier
                     (2024 - year) * -500)   # Malus âge
        
        # Ajout de bruit réaliste
        noise = np.random.normal(0, 50000, n_samples)
        y = np.maximum(base_price + noise, 100000)  # Prix minimum 100k€
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement du modèle
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info("Modèle entraîné avec succès!")
        
    def predict(self, data: PropertyData) -> dict:
        """Effectue une prédiction d'estimation"""
        if not self.is_trained:
            self.train_model()
        
        # Préprocessing
        features = self.preprocess_data(data)
        features_scaled = self.scaler.transform(features)
        
        # Prédiction
        prediction = self.model.predict(features_scaled)[0]
        
        # Calcul de l'intervalle de confiance (±10%)
        confidence_interval = 0.10
        price_min = prediction * (1 - confidence_interval)
        price_max = prediction * (1 + confidence_interval)
        
        # Récupération des prédictions de chaque arbre
        tree_predictions = np.array([tree.predict(features_scaled)[0] for tree in self.model.estimators_])
        
        # Calcul du score de confiance amélioré
        relative_std = np.std(tree_predictions) / prediction if prediction != 0 else 0
        confidence_score = max(0, min(100, 100 * (1 - relative_std)))
        
        return {
            'estimated_price': round(prediction),
            'price_min': round(price_min),
            'price_max': round(price_max),
            'price_per_sqm': round(prediction / data.surface),
            'confidence_score': round(confidence_score, 1),
            'variance': np.var(tree_predictions)
        }
# Instance globale du modèle
estimator = RealEstateEstimator()

# Routes API
@app.get("/")
async def root():
    return {"message": "EstimPro API - Service d'estimation immobilière"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_trained": estimator.is_trained
    }

@app.post("/estimate", response_model=EstimationResponse)
async def estimate_property(property_data: PropertyData):
    """Endpoint principal d'estimation"""
    try:
        logger.info(f"Nouvelle demande d'estimation pour: {property_data.address}")
        
        # Validation des données
        if property_data.surface <= 0:
            raise HTTPException(status_code=400, detail="La surface doit être positive")
        if property_data.rooms <= 0:
            raise HTTPException(status_code=400, detail="Le nombre de pièces doit être positif")
        
        # Estimation
        estimation_result = estimator.predict(property_data)
        
        # Génération des données de marché simulées
        market_trends = generate_market_trends()
        comparable_properties = generate_comparable_properties(property_data)
        factors_analysis = analyze_price_factors(property_data, estimation_result)
        
        response = EstimationResponse(
            estimated_price=estimation_result['estimated_price'],
            price_min=estimation_result['price_min'],
            price_max=estimation_result['price_max'],
            price_per_sqm=estimation_result['price_per_sqm'],
            confidence_score=estimation_result['confidence_score'],
            market_trends=market_trends,
            comparable_properties=comparable_properties,
            factors_analysis=factors_analysis
        )
        
        logger.info(f"Estimation réalisée: {estimation_result['estimated_price']}€")
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de l'estimation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

def generate_market_trends() -> List[dict]:
    """Génère des données de tendance de marché simulées"""
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
    base_price = 4200
    trends = []
    
    for i, month in enumerate(months):
        price = base_price + (i * 150) + np.random.randint(-50, 100)
        trends.append({
            'month': month,
            'price': price,
            'change': round(((price - base_price) / base_price) * 100, 1)
        })
    
    return trends

def generate_comparable_properties(property_data: PropertyData) -> List[dict]:
    """Génère des biens comparables simulés"""
    comparables = []
    base_price = property_data.surface * 4800
    
    for i in range(4):
        surface_variation = np.random.randint(-10, 15)
        price_variation = np.random.randint(-30000, 40000)
        
        surface = max(30, property_data.surface + surface_variation)
        price = max(150000, base_price + price_variation)
        
        comparables.append({
            'address': f"Rue Example {i+1}, 75001 Paris",
            'surface': surface,
            'price': round(price),
            'price_per_sqm': round(price / surface),
            'sold_date': f"2024-0{np.random.randint(1, 4)}-{np.random.randint(10, 28)}"
        })
    
    return comparables

def analyze_price_factors(property_data: PropertyData, estimation: dict) -> dict:
    """Analyse les facteurs influençant le prix"""
    factors = {
        'location_impact': 15 if property_data.latitude and abs(property_data.latitude - 48.8566) < 0.05 else 5,
        'condition_impact': {'excellent': 10, 'good': 5, 'average': 0, 'renovation': -10}.get(property_data.condition, 0),
        'size_impact': 5 if 80 <= property_data.surface <= 120 else 0,
        'year_impact': max(-15, min(10, (property_data.year - 1990) / 10)),
        'parking_impact': {'garage': 8, 'covered': 5, 'street': 2, 'none': 0}.get(property_data.parking, 0)
    }
    
    return factors

# Endpoint pour les statistiques de marché
@app.get("/market-stats")
async def get_market_stats():
    """Retourne les statistiques générales du marché"""
    return {
        'average_price_per_sqm': 4850,
        'market_growth_6m': 15.5,
        'average_selling_time': 23,
        'total_transactions': 50000,
        'last_update': datetime.now().isoformat()
    }

# Endpoint pour la géolocalisation
@app.get("/geocode")
async def geocode_address(address: str):
    """Géocode une adresse (simulation)"""
    # En production, utiliser une vraie API de géocodage
    return {
        'address': address,
        'latitude': 48.8566 + np.random.uniform(-0.1, 0.1),
        'longitude': 2.3522 + np.random.uniform(-0.1, 0.1),
        'district': 'Paris Centre',
        'district_score': np.random.uniform(6, 9)
    }

if __name__ == "__main__":
    print("🚀 Démarrage du serveur EstimPro API...")
    print("📊 Entraînement du modèle IA...")
    estimator.train_model()
    print("✅ Serveur prêt!")

    import uvicorn
    uvicorn.run(
        "backend_api:app",    # <-- import string
        host="0.0.0.0",
        port=8000,
        reload=True,          # <-- reload utilisable
        log_level="info"
    )




