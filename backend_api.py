from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import uvicorn
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="EstimPro API - Madagascar",
    description="API d'estimation immobilière avec IA adaptée à Antananarivo, Madagascar",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Données des quartiers d'Antananarivo avec fourchettes de prix au m² (en MGA)
QUARTIERS = {
    "Analakely": {"min_price_sqm": 300000, "max_price_sqm": 1000000, "base_score": 9.0},
    "Antaninarenina": {"min_price_sqm": 300000, "max_price_sqm": 1000000, "base_score": 9.0},
    "Isoraka": {"min_price_sqm": 300000, "max_price_sqm": 1000000, "base_score": 8.5},
    "Ambatonakanga": {"min_price_sqm": 300000, "max_price_sqm": 800000, "base_score": 8.0},
    "Ankadifotsy": {"min_price_sqm": 250000, "max_price_sqm": 700000, "base_score": 7.5},
    "Tsaralalana": {"min_price_sqm": 300000, "max_price_sqm": 600000, "base_score": 7.0},
    "Ivandry": {"min_price_sqm": 200000, "max_price_sqm": 500000, "base_score": 6.5},
    "Ambohipo": {"min_price_sqm": 150000, "max_price_sqm": 300000, "base_score": 6.0},
    "Ambatoroka": {"min_price_sqm": 60000, "max_price_sqm": 150000, "base_score": 5.5},
    "Ambohijatovo": {"min_price_sqm": 150000, "max_price_sqm": 250000, "base_score": 5.5},
    "Ankadivato": {"min_price_sqm": 200000, "max_price_sqm": 350000, "base_score": 6.0},
    "Ampasampito": {"min_price_sqm": 150000, "max_price_sqm": 300000, "base_score": 5.5},
    "Ivato": {"min_price_sqm": 100000, "max_price_sqm": 250000, "base_score": 5.0},
    "Talatamaty": {"min_price_sqm": 80000, "max_price_sqm": 150000, "base_score": 4.5},
    "Tanjombato": {"min_price_sqm": 70000, "max_price_sqm": 120000, "base_score": 4.0},
    "Ambohidratrimo": {"min_price_sqm": 50000, "max_price_sqm": 120000, "base_score": 4.0},
    "Ambohimalaza": {"min_price_sqm": 30000, "max_price_sqm": 100000, "base_score": 3.5},
    "Anosizato": {"min_price_sqm": 70000, "max_price_sqm": 150000, "base_score": 4.0},
    "Andoharanofotsy": {"min_price_sqm": 60000, "max_price_sqm": 130000, "base_score": 3.5},
    "Alakamisy-Ambohidratrimo": {"min_price_sqm": 4000, "max_price_sqm": 20000, "base_score": 2.0},
    "Anjeva Gara": {"min_price_sqm": 10000, "max_price_sqm": 30000, "base_score": 2.0},
    "Moramanga": {"min_price_sqm": 3000, "max_price_sqm": 15000, "base_score": 1.5},
    "Manjakandriana": {"min_price_sqm": 5000, "max_price_sqm": 25000, "base_score": 1.5},
    "Ambatomirahavavy": {"min_price_sqm": 40000, "max_price_sqm": 60000, "base_score": 3.0},
    "Anosiala": {"min_price_sqm": 5000, "max_price_sqm": 30000, "base_score": 2.0}
}

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
    district: Optional[str] = None

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
            'surface', 'rooms', 'bedrooms', 'bathrooms', 'year', 'floor',
            'condition_encoded', 'property_type_encoded', 'parking_encoded',
            'garden_encoded', 'district_score', 'base_price_sqm'
        ]
        self.is_trained = False

    def get_district_info(self, address: str, lat: Optional[float], lon: Optional[float]) -> tuple:
        """Identifie le quartier à partir de l'adresse ou des coordonnées"""
        for district, info in QUARTIERS.items():
            if district.lower() in address.lower():
                return district, info["min_price_sqm"], info["max_price_sqm"], info["base_score"]
        # Valeur par défaut si aucun quartier n'est identifié
        return "Unknown", 50000, 100000, 5.0

    def preprocess_data(self, data: PropertyData) -> np.ndarray:
        """Préprocessing des données d'entrée"""
        condition_mapping = {'excellent': 4, 'good': 3, 'average': 2, 'renovation': 1}
        property_type_mapping = {'apartment': 1, 'house': 2, 'studio': 0, 'loft': 3, 'duplex': 4}
        parking_mapping = {'none': 0, 'street': 1, 'covered': 2, 'garage': 3}
        garden_mapping = {'none': 0, 'balcony': 1, 'terrace': 2, 'garden': 3}
        floor = data.floor if data.floor else 0

        district, min_price_sqm, max_price_sqm, district_score = self.get_district_info(data.address, data.latitude, data.longitude)
        base_price_sqm = (min_price_sqm + max_price_sqm) / 2

        features = np.array([
            data.surface,
            data.rooms,
            data.bedrooms,
            data.bathrooms,
            data.year,
            floor,
            condition_mapping.get(data.condition, 2),
            property_type_mapping.get(data.property_type, 1),
            parking_mapping.get(data.parking, 0),
            garden_mapping.get(data.garden, 0),
            district_score,
            base_price_sqm
        ]).reshape(1, -1)

        return features

    def train_model(self):
        """Entraîne le modèle avec des données simulées adaptées pour Madagascar"""
        logger.info("Entraînement du modèle d'estimation...")

        np.random.seed(42)
        n_samples = 10000

        # Générer des données simulées adaptées à Madagascar
        surface = np.random.normal(80, 20, n_samples)
        rooms = np.random.randint(1, 5, n_samples)
        bedrooms = np.random.randint(0, 3, n_samples)
        bathrooms = np.random.randint(1, 2, n_samples)
        year = np.random.randint(1980, 2024, n_samples)
        condition = np.random.randint(1, 4, n_samples)
        property_type = np.random.randint(0, 5, n_samples)
        parking = np.random.randint(0, 3, n_samples)
        garden = np.random.randint(0, 3, n_samples)
        floor = np.random.randint(0, 4, n_samples)
        district_indices = np.random.choice(list(QUARTIERS.keys()), n_samples)
        district_scores = np.array([QUARTIERS[d]["base_score"] for d in district_indices])
        base_price_sqm = np.array([(QUARTIERS[d]["min_price_sqm"] + QUARTIERS[d]["max_price_sqm"]) / 2 for d in district_indices])

        X = np.column_stack([
            surface, rooms, bedrooms, bathrooms, year, floor,
            condition, property_type, parking, garden,
            district_scores, base_price_sqm
        ])

        # Calcul du prix adapté à Madagascar
        y = surface * base_price_sqm * (
            1 + 
            (condition - 2) * 0.05 +
            (rooms - 2) * 0.03 +
            (bedrooms - 1) * 0.02 +
            (bathrooms - 1) * 0.02 +
            (year - 2000) * -0.001 +
            floor * 0.01 +
            np.random.normal(0, 0.05, n_samples)
        )

        y = np.maximum(y, base_price_sqm * surface * 0.8)

        # Séparation des données en 80% entraînement et 20% test pour l'évaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Standardisation des features (fit seulement sur les données d'entraînement)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Évaluation initiale du modèle
        eval_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        eval_model.fit(X_train_scaled, y_train)
        
        # Évaluation sur l'ensemble de test
        y_pred = eval_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Évaluation initiale - MSE: {mse:.2f}, R²: {r2:.3f}")

        # Entraînement final sur TOUTES les données
        X_scaled = self.scaler.transform(X)  # Utilise le scaler déjà ajusté sur X_train
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True

        logger.info(f"Modèle final entraîné avec succès sur toutes les données!")

    def predict(self, data: PropertyData) -> dict:
        """Effectue une prédiction d'estimation"""
        if not self.is_trained:
            self.train_model()

        features = self.preprocess_data(data)
        features_scaled = self.scaler.transform(features)

        # Prédiction avec le modèle entraîné sur toutes les données
        prediction = self.model.predict(features_scaled)[0]

        # Calcul de l'intervalle de confiance
        confidence_interval = 0.15
        price_min = prediction * (1 - confidence_interval)
        price_max = prediction * (1 + confidence_interval)

        # Calcul du score de confiance basé sur la variance des prédictions des arbres
        tree_predictions = np.array([tree.predict(features_scaled)[0] for tree in self.model.estimators_])
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
    return {"message": "EstimPro API - Service d'estimation immobilière (Antananarivo, Madagascar)"}

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

        if property_data.surface <= 0:
            raise HTTPException(status_code=400, detail="La surface doit être positive")
        if property_data.rooms <= 0:
            raise HTTPException(status_code=400, detail="Le nombre de pièces doit être positif")

        estimation_result = estimator.predict(property_data)

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

        logger.info(f"Estimation réalisée: {estimation_result['estimated_price']} MGA")
        return response

    except Exception as e:
        logger.error(f"Erreur lors de l'estimation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

def generate_market_trends() -> List[dict]:
    """Génère des données de tendance de marché simulées pour Madagascar"""
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
    base_price = 150000
    trends = []

    for i, month in enumerate(months):
        price = base_price + (i * 5000) + np.random.randint(-3000, 3000)
        trends.append({
            'month': month,
            'price': price,
            'change': round(((price - base_price) / base_price) * 100, 1)
        })

    return trends

def generate_comparable_properties(property_data: PropertyData) -> List[dict]:
    """Génère des biens comparables simulés"""
    comparables = []
    district, min_price_sqm, max_price_sqm, _ = estimator.get_district_info(property_data.address, property_data.latitude, property_data.longitude)
    base_price_sqm = (min_price_sqm + max_price_sqm) / 2
    base_price = property_data.surface * base_price_sqm

    for i in range(4):
        surface_variation = np.random.randint(-10, 10)
        price_variation = np.random.randint(-0.05 * base_price, 0.05 * base_price)

        surface = max(20, property_data.surface + surface_variation)
        price = max(base_price_sqm * surface * 0.8, base_price + price_variation)

        comparables.append({
            'address': f"{district} Exemple {i+1}, Antananarivo",
            'surface': round(surface, 1),
            'price': round(price),
            'price_per_sqm': round(price / surface),
            'sold_date': f"2024-0{np.random.randint(1, 4)}-{np.random.randint(10, 28)}"
        })

    return comparables

def analyze_price_factors(property_data: PropertyData, estimation: dict) -> dict:
    """Analyse les facteurs influençant le prix"""
    district, _, _, district_score = estimator.get_district_info(property_data.address, property_data.latitude, property_data.longitude)
    factors = {
        'location_impact': district_score * 2,
        'condition_impact': {'excellent': 5, 'good': 2, 'average': 0, 'renovation': -5}.get(property_data.condition, 0),
        'size_impact': 2 if 60 <= property_data.surface <= 100 else 0,
        'year_impact': max(-5, min(5, (property_data.year - 2000) / 10)),
        'parking_impact': {'garage': 3, 'covered': 2, 'street': 1, 'none': 0}.get(property_data.parking, 0),
        'floor_impact': property_data.floor * 0.5 if property_data.floor else 0
    }

    return factors

@app.get("/market-stats")
async def get_market_stats():
    """Retourne les statistiques générales du marché"""
    return {
        'average_price_per_sqm': 150000,
        'market_growth_6m': 5.0,
        'average_selling_time': 30,
        'total_transactions': 20000,
        'last_update': datetime.now().isoformat()
    }

@app.get("/geocode")
async def geocode_address(address: str):
    """Géocode une adresse (simulation)"""
    district, _, _, district_score = estimator.get_district_info(address, None, None)
    return {
        'address': address,
        'latitude': -18.8792 + np.random.uniform(-0.05, 0.05),
        'longitude': 47.5079 + np.random.uniform(-0.05, 0.05),
        'district': district,
        'district_score': district_score
    }

if __name__ == "__main__":
    print("🚀 Démarrage du serveur EstimPro API (Antananarivo, Madagascar)...")
    print("📊 Entraînement du modèle IA avec prix adaptés...")
    estimator.train_model()
    print("✅ Serveur prêt!")

    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )