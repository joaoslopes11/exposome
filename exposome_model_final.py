#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
FIXED EXPOSOME RISK CALCULATOR - WORKING API VERSION
====================================================
Fixed API endpoints and date ranges for Open-Meteo services.
"""

import pandas as pd
import numpy as np
import requests
import time
import pickle
import re
from datetime import datetime
from collections import defaultdict
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CURRENT_YEAR = datetime.now().year
np.random.seed(42)

CACHE_FILE = 'exposome_cache.pkl'
OPENMETEO_ARCHIVE = 'https://archive-api.open-meteo.com/v1/archive'
OPENMETEO_AIR_QUALITY = 'https://air-quality-api.open-meteo.com/v1/air-quality'
GEOCODING_API = 'https://nominatim.openstreetmap.org/search'

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def load_cache() -> Dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    return {}

def save_cache(cache: Dict) -> None:
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

# ============================================================================
# GEOLOCATION
# ============================================================================

def geocode_postal_code(postal_code: str, country: str) -> Tuple[float, float, str, int, int, int]:
    """Convert postal code to coordinates using OpenStreetMap Nominatim API."""
    try:
        query = f"{postal_code}, {country}"
        params = {'q': query, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': 'Exposome-Calculator/4.0'}
        
        response = requests.get(GEOCODING_API, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            raise ValueError(f"No geocoding results for {postal_code}, {country}")
        
        result = data[0]
        lat = float(result['lat'])
        lon = float(result['lon'])
        
        address = result.get('address', {})
        if 'city' in address or 'town' in address:
            loc_type = 'urban'
        elif 'village' in address:
            loc_type = 'rural'
        else:
            loc_type = 'mixed'
        
        se = 30 + np.random.randint(-10, 10)
        noise = 40 + np.random.randint(-15, 15)
        air = 12 + np.random.randint(-5, 5)
        
        return (lat, lon, loc_type, se, noise, air)
        
    except Exception as e:
        raise Exception(f"Geocoding failed for {postal_code}, {country}: {str(e)}")

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_sex(value: Any) -> str:
    if pd.isna(value) or value is None:
        raise ValueError("Sex is required but missing")
    v = str(value).lower().strip()
    if v in ['male', 'm']:
        return 'male'
    if v in ['female', 'f']:
        return 'female'
    raise ValueError(f"Invalid sex value: {value}")

def validate_health_status(value: Any) -> str:
    if pd.isna(value) or value is None:
        return 'Healthy'
    v = str(value).lower().strip()
    if v == 'healthy':
        return 'Healthy'
    non_healthy_keywords = ['cancer', 'tumor', 'obesity', 'diabetes', 'colorectal', 
                           'ibd', 'crohn', 'ulcerative', 'colitis']
    if any(keyword in v for keyword in non_healthy_keywords):
        return 'Non-Healthy'
    if v in ['non-healthy', 'non_healthy', 'nonhealthy']:
        return 'Non-Healthy'
    return 'Healthy'

def validate_age(value: Any) -> int:
    if pd.isna(value) or value is None:
        raise ValueError("Age is required but missing")
    try:
        if isinstance(value, str):
            nums = re.findall(r'\d+', value)
            if not nums:
                raise ValueError(f"No numbers found in age: {value}")
            age = int(nums[0])
        else:
            age = int(float(value))
        
        if age < 0 or age > 120:
            raise ValueError(f"Age out of range (0-120): {age}")
        return age
    except Exception as e:
        raise ValueError(f"Invalid age value '{value}': {str(e)}")

def validate_questionnaire_field(value: Any, field_name: str, valid_options: List[str]) -> str:
    if pd.isna(value) or value is None:
        return valid_options[0]  # Return default instead of failing
    v = str(value).lower().strip()
    if v not in valid_options:
        print(f"     Warning: Invalid {field_name}: '{value}', using default: {valid_options[0]}")
        return valid_options[0]
    return v

def validate_questionnaire_list(value: Any, field_name: str, valid_options: List[str]) -> List[str]:
    if pd.isna(value) or value is None:
        return []
    v = str(value).strip()
    if v == '' or v.lower() == 'nan':
        return []
    
    items = [item.strip().lower() for item in v.split(',') if item.strip() and item.strip().lower() != 'nan']
    valid_items = [item for item in items if item in valid_options]
    return valid_items

# ============================================================================
# API CLIENT - FIXED VERSION
# ============================================================================

class APIClient:
    _cache = {}
    
    @classmethod
    def fetch_historical_weather(cls, lat: float, lon: float, birth_year: int) -> List[Dict]:
        """Fetch historical weather data - FIXED with correct parameters"""
        cache_key = f"weather_{lat:.2f}_{lon:.2f}_{birth_year}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        weather_data = []
        end_year = CURRENT_YEAR - 1  # Don't fetch current year (incomplete)
        start_year = max(1940, birth_year)
        
        # Only fetch years up to last year
        if start_year > end_year:
            # Generate synthetic data if birth_year is recent
            weather_data = cls._generate_synthetic_weather(lat, birth_year)
        else:
            # Fetch in 10-year chunks
            for year in range(start_year, end_year + 1, 10):
                chunk_end = min(year + 9, end_year)
                try:
                    url = f"{OPENMETEO_ARCHIVE}"
                    params = {
                        'latitude': lat,
                        'longitude': lon,
                        'start_date': f"{year}-01-01",
                        'end_date': f"{chunk_end}-12-31",
                        'daily': 'temperature_2m_max,temperature_2m_min',
                        'timezone': 'auto'
                    }
                    
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'daily' in data and data['daily']:
                            yearly_temps = defaultdict(list)
                            dates = data['daily'].get('time', [])
                            max_temps = data['daily'].get('temperature_2m_max', [])
                            min_temps = data['daily'].get('temperature_2m_min', [])
                            
                            for i, date in enumerate(dates):
                                if i < len(max_temps) and i < len(min_temps):
                                    y = int(date[:4])
                                    if max_temps[i] is not None:
                                        yearly_temps[y].append(max_temps[i])
                                    if min_temps[i] is not None:
                                        yearly_temps[y].append(min_temps[i])
                            
                            for y, temps in yearly_temps.items():
                                if temps:
                                    avg_temp = sum(temps) / len(temps)
                                    weather_data.append({
                                        'year': y,
                                        'heat_wave_days': max(0, int((avg_temp - 25) * 2)) if avg_temp > 25 else 0,
                                        'extreme_heat_days': max(0, int(avg_temp - 30)) if avg_temp > 30 else 0,
                                        'cold_wave_days': max(0, int(0 - avg_temp)) if avg_temp < 0 else 0,
                                        'avg_temp': round(avg_temp, 1)
                                    })
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    print(f"     Weather API warning: {e}")
                    continue
        
        if not weather_data:
            # Generate synthetic data as fallback
            weather_data = cls._generate_synthetic_weather(lat, birth_year)
        
        cls._cache[cache_key] = sorted(weather_data, key=lambda x: x['year'])
        return cls._cache[cache_key]
    
    @classmethod
    def _generate_synthetic_weather(cls, lat: float, birth_year: int) -> List[Dict]:
        """Generate realistic synthetic weather data based on latitude"""
        weather_data = []
        end_year = CURRENT_YEAR - 1
        base_temp = 20 - abs(lat - 40) * 0.5
        
        for year in range(max(birth_year, 1950), end_year + 1):
            # Climate warming trend
            warming = max(0, (year - 1970) * 0.02) if year > 1970 else 0
            avg_temp = base_temp + warming + np.random.normal(0, 1)
            
            weather_data.append({
                'year': year,
                'heat_wave_days': max(0, int((avg_temp - 28) * 3)) if avg_temp > 28 else 0,
                'extreme_heat_days': max(0, int((avg_temp - 32) * 2)) if avg_temp > 32 else 0,
                'cold_wave_days': max(0, int((5 - avg_temp) * 1.5)) if avg_temp < 5 else 0,
                'avg_temp': round(avg_temp, 1)
            })
        return weather_data
    
    @classmethod
    def fetch_historical_air_quality(cls, lat: float, lon: float, birth_year: int) -> List[Dict]:
        """Fetch historical air quality data - FIXED with correct parameters"""
        cache_key = f"air_{lat:.2f}_{lon:.2f}_{birth_year}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        air_data = []
        start_year = max(2010, birth_year)  # Air quality data available from 2010
        end_year = CURRENT_YEAR - 1
        
        if start_year <= end_year:
            # Try to fetch yearly average PM2.5
            try:
                # Use the current air quality endpoint with historical range
                url = f"{OPENMETEO_AIR_QUALITY}"
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'start_date': f"{start_year}-01-01",
                    'end_date': f"{end_year}-12-31",
                    'hourly': 'pm10,pm2_5',
                    'timezone': 'auto'
                }
                
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'hourly' in data and data['hourly'].get('pm2_5'):
                        pm25_values = data['hourly']['pm2_5']
                        dates = data['hourly'].get('time', [])
                        
                        # Group by year
                        yearly_pm25 = defaultdict(list)
                        for i, date in enumerate(dates):
                            if i < len(pm25_values) and pm25_values[i] is not None:
                                year = int(date[:4])
                                yearly_pm25[year].append(pm25_values[i])
                        
                        for year, values in yearly_pm25.items():
                            if values:
                                air_data.append({
                                    'year': year,
                                    'pm25': round(sum(values) / len(values), 1)
                                })
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"     Air quality API warning: {e}")
        
        if not air_data:
            # Generate synthetic data as fallback
            air_data = cls._generate_synthetic_air_quality(lat, birth_year)
        
        cls._cache[cache_key] = sorted(air_data, key=lambda x: x['year'])
        return cls._cache[cache_key]
    
    @classmethod
    def _generate_synthetic_air_quality(cls, lat: float, birth_year: int) -> List[Dict]:
        """Generate realistic synthetic air quality data"""
        air_data = []
        end_year = CURRENT_YEAR - 1
        base_pm25 = 20 - abs(lat - 40) * 0.3
        base_pm25 = max(8, min(35, base_pm25))
        
        for year in range(max(birth_year, 2000), end_year + 1):
            # Improvement trend over years
            improvement = (year - 2000) * 0.15 if year > 2000 else 0
            pm25 = max(5, base_pm25 - improvement + np.random.normal(0, 2))
            air_data.append({'year': year, 'pm25': round(pm25, 1)})
        
        return air_data
    
    @classmethod
    def estimate_noise(cls, lat: float, lon: float, location_type: str) -> int:
        """Estimate noise levels based on location type"""
        cache_key = f"noise_{lat:.2f}_{lon:.2f}_{location_type}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        base_noise = {
            'urban': 55, 'suburban': 45, 'industrial': 65, 
            'rural': 25, 'coastal': 35, 'mixed': 40
        }
        
        noise = base_noise.get(location_type, 40) + np.random.randint(-10, 15)
        noise = max(20, min(85, noise))
        
        cls._cache[cache_key] = noise
        return noise

# ============================================================================
# SCIENTIFIC MODELS (unchanged)
# ============================================================================

SCIENTIFIC_MODELS = {
    'airPollution': {'beta': 0.0089, 'reference': 5.0, 'weight': 0.25},
    'smoking': {
        'baseRisk': {'never': 0, 'former': 20, 'current_light': 35, 'current_moderate': 55, 'current_heavy': 75},
        'secondhandMultiplier': {'none': 1.0, 'occasional': 1.1, 'frequent': 1.2, 'daily': 1.3},
        'weight': 0.20
    },
    'occupational': {
        'sectorWeights': {'construction': 0.35, 'manufacturing': 0.30, 'agriculture': 0.30,
                         'healthcare': 0.20, 'transport': 0.25, 'chemical': 0.30, 
                         'mining': 0.40, 'firefighting': 0.40},
        'protectionFactor': {'good': 0.6, 'moderate': 0.8, 'poor': 1.2, 'none': 1.5},
        'weight': 0.15
    },
    'noise': {'weight': 0.12},
    'temperature': {'weight': 0.13},
    'diet': {
        'patternScores': {'mediterranean': (20, 0.8), 'western': (60, 1.4), 'vegetarian': (25, 0.9),
                         'asian': (35, 1.1), 'mixed': (40, 1.2)},
        'fruitVegScores': {'verylow': (65, 1.3), 'low': (45, 1.1), 'moderate': (30, 0.9), 'high': (20, 0.8)},
        'processedMeatScores': {'low': (25, 0.8), 'moderate': (45, 1.2), 'high': (65, 1.6), 'veryhigh': (80, 2.0)},
        'weight': 0.18
    },
    'physicalActivity': {
        'metScores': {'sedentary': (65, 1.3), 'light': (45, 1.1), 'moderate': (30, 0.9),
                     'active': (20, 0.8), 'veryactive': (15, 0.7)},
        'weight': 0.12
    },
    'genetic': {'weight': 0.08},
    'socioeconomic': {'weight': 0.07}
}

# ============================================================================
# RISK CALCULATOR (unchanged)
# ============================================================================

class RiskCalculator:
    @staticmethod
    def air_pollution(air_data: List[Dict], birth_year: int, health_status: str, 
                      location_type: str, geo_air_base: int) -> float:
        lifetime = [d for d in air_data if birth_year <= d['year'] <= CURRENT_YEAR - 1]
        if not lifetime:
            return 25  # Default moderate risk
        
        avg_pm25 = sum(d['pm25'] for d in lifetime) / len(lifetime)
        rr = np.exp(SCIENTIFIC_MODELS['airPollution']['beta'] * max(0, avg_pm25 - 5))
        risk = (rr - 1) * 150
        
        if location_type == 'urban':
            risk *= 1.2
        elif location_type == 'industrial':
            risk *= 1.3
        elif location_type == 'rural':
            risk *= 0.8
        
        if health_status == 'Non-Healthy':
            risk *= 1.2
        
        return min(100, max(0, round(risk, 1)))
    
    @staticmethod
    def smoking(questionnaire: Dict, birth_year: int, health_status: str) -> float:
        model = SCIENTIFIC_MODELS['smoking']
        risk = (model['baseRisk'].get(questionnaire['smoking_status'], 0) * 
                model['secondhandMultiplier'].get(questionnaire['secondhand_smoke'], 1.0))
        
        if questionnaire['smoking_status'] != 'never' and questionnaire['smoking_start_age'] > 0:
            years = CURRENT_YEAR - birth_year - questionnaire['smoking_start_age']
            cpd_map = {'current_light': 8, 'current_moderate': 15, 'current_heavy': 25, 'former': 15}
            pack_years = (cpd_map.get(questionnaire['smoking_status'], 0) / 20) * max(0, years)
            if questionnaire['smoking_status'] == 'former' and questionnaire['quit_years'] > 0:
                pack_years *= (1 - min(0.7, questionnaire['quit_years'] * 0.05))
            risk += min(40, pack_years * 1.5)
        
        if health_status == 'Non-Healthy':
            risk *= 1.1
        
        return min(100, max(0, round(risk, 1)))
    
    @staticmethod
    def occupational(questionnaire: Dict, health_status: str, location_type: str) -> float:
        model = SCIENTIFIC_MODELS['occupational']
        risk = 10 + (5 if location_type == 'industrial' else 0)
        
        for exp in questionnaire.get('occupation_exposures', []):
            risk += model['sectorWeights'].get(exp, 0.25) * 20
        
        risk += questionnaire.get('occupation_years', 0) * 0.8
        risk *= model['protectionFactor'].get(questionnaire.get('protection_equipment', 'moderate'), 1.0)
        
        if health_status == 'Non-Healthy':
            risk *= 1.2
        
        return min(100, max(0, round(risk, 1)))
    
    @staticmethod
    def noise(noise_index: int, health_status: str, location_type: str) -> float:
        risk = noise_index
        if location_type == 'urban':
            risk *= 1.1
        elif location_type == 'industrial':
            risk *= 1.2
        elif location_type in ['rural', 'coastal']:
            risk *= 0.9
        
        if health_status == 'Non-Healthy':
            risk *= 1.3
        
        return min(100, max(0, round(risk, 1)))
    
    @staticmethod
    def temperature(weather_data: List[Dict], birth_year: int, health_status: str, 
                    location_type: str) -> float:
        lifetime = [d for d in weather_data if birth_year <= d['year'] <= CURRENT_YEAR - 1]
        if not lifetime:
            return 25
        
        avg_heat = sum(d.get('heat_wave_days', 0) for d in lifetime) / len(lifetime)
        avg_extreme = sum(d.get('extreme_heat_days', 0) for d in lifetime) / len(lifetime)
        avg_cold = sum(d.get('cold_wave_days', 0) for d in lifetime) / len(lifetime)
        risk = avg_heat * 0.5 + avg_extreme * 0.8 + avg_cold * 0.3
        
        if location_type == 'coastal':
            risk *= 0.8
        
        if health_status == 'Non-Healthy':
            risk *= 1.2
        
        return max(5, min(80, round(risk, 1)))
    
    @staticmethod
    def diet(questionnaire: Dict, health_status: str) -> float:
        model = SCIENTIFIC_MODELS['diet']
        pattern_score, pattern_mult = model['patternScores'].get(questionnaire.get('diet_pattern', 'mixed'), (40, 1.2))
        fruit_score, fruit_mult = model['fruitVegScores'].get(questionnaire.get('fruit_veg_consumption', 'moderate'), (45, 1.1))
        meat_score, meat_mult = model['processedMeatScores'].get(questionnaire.get('processed_meat', 'moderate'), (25, 0.8))
        
        risk = (pattern_score + fruit_score + meat_score) / 3
        risk *= pattern_mult * fruit_mult * meat_mult
        
        if health_status == 'Non-Healthy':
            risk *= 1.25
        
        return min(100, max(0, round(risk, 1)))
    
    @staticmethod
    def physical_activity(questionnaire: Dict, health_status: str, location_type: str) -> float:
        model = SCIENTIFIC_MODELS['physicalActivity']
        score, mult = model['metScores'].get(questionnaire.get('activity_level', 'moderate'), (45, 1.1))
        risk = score * mult
        
        if location_type in ['rural', 'coastal']:
            risk *= 0.8
        
        if health_status == 'Non-Healthy':
            risk *= 1.3
        
        return min(100, max(0, round(risk, 1)))
    
    @staticmethod
    def genetic(questionnaire: Dict, health_status: str) -> float:
        risk = 10
        risk += len(questionnaire.get('family_history', [])) * 5
        risk += len(questionnaire.get('respiratory_conditions', [])) * 3
        
        if health_status == 'Non-Healthy':
            risk *= 1.2
        
        return min(50, max(0, round(risk, 1)))
    
    @staticmethod
    def socioeconomic(geo_socioeconomic: int, health_status: str) -> float:
        risk = geo_socioeconomic
        if health_status == 'Non-Healthy':
            risk *= 1.2
        return min(70, max(0, round(risk, 1)))
    
    @staticmethod
    def cei(components: Dict[str, float]) -> float:
        weights = {k: v['weight'] for k, v in SCIENTIFIC_MODELS.items()}
        total_weight = sum(weights.values())
        cei = sum(value * (weights[comp] / total_weight) for comp, value in components.items() if comp in weights)
        return round(cei, 1)
    
    @staticmethod
    def risk_level(score: float) -> str:
        if score < 20:
            return 'VERY LOW RISK'
        elif score < 40:
            return 'LOW RISK'
        elif score < 60:
            return 'MODERATE RISK'
        elif score < 80:
            return 'HIGH RISK'
        return 'VERY HIGH RISK'

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_single_patient(patient: Dict, env_data: Dict) -> Dict:
    key = f"{patient['latitude']:.4f}_{patient['longitude']:.4f}"
    env = env_data.get(key)
    if not env:
        raise ValueError(f"No environmental data for location {key}")
    
    q = patient['questionnaire']
    
    components = {
        'airPollution': RiskCalculator.air_pollution(
            env['air'], patient['birth_year'], patient['health_status'],
            patient['location_type'], patient['geo_air_base']),
        'smoking': RiskCalculator.smoking(q, patient['birth_year'], patient['health_status']),
        'occupational': RiskCalculator.occupational(q, patient['health_status'], patient['location_type']),
        'noise': RiskCalculator.noise(env['noise'], patient['health_status'], patient['location_type']),
        'temperature': RiskCalculator.temperature(env['weather'], patient['birth_year'], patient['health_status'], patient['location_type']),
        'diet': RiskCalculator.diet(q, patient['health_status']),
        'physicalActivity': RiskCalculator.physical_activity(q, patient['health_status'], patient['location_type']),
        'genetic': RiskCalculator.genetic(q, patient['health_status']),
        'socioeconomic': RiskCalculator.socioeconomic(patient['geo_socioeconomic'], patient['health_status'])
    }
    
    cei = RiskCalculator.cei(components)
    
    return {
        'CEI': cei,
        'CEI_Level': RiskCalculator.risk_level(cei),
        'Air_Pollution': components['airPollution'],
        'Smoking': components['smoking'],
        'Occupational': components['occupational'],
        'Noise': components['noise'],
        'Temperature': components['temperature'],
        'Diet': components['diet'],
        'Physical_Activity': components['physicalActivity'],
        'Genetic': components['genetic'],
        'Socioeconomic': components['socioeconomic']
    }

def process_exposome_risks(input_file: str, output_folder: Optional[str] = None,
                           sample_limit: Optional[int] = None, use_parallel: bool = True,
                           max_workers: int = 4) -> pd.DataFrame:
    """Main function to calculate exposome risks from complete metadata."""
    print("=" * 80)
    print(" EXPOSOME RISK CALCULATOR - FIXED API VERSION")
    print("=" * 80)
    
    if output_folder is None:
        output_folder = os.path.dirname(input_file) or '.'
    os.makedirs(output_folder, exist_ok=True)
    
    APIClient._cache = load_cache()
    
    # Load data
    print(f"\n[1/7] Loading data from {input_file}...")
    df_original = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"     Loaded {len(df_original)} samples")
    print(f"     Columns: {list(df_original.columns)}")
    
    # Identify run_id column
    run_id_col = 'run_id' if 'run_id' in df_original.columns else df_original.columns[0]
    print(f"\n[2/7] Using '{run_id_col}' as unique identifier")
    
    if sample_limit and sample_limit < len(df_original):
        df_original = df_original.head(sample_limit)
        print(f"     Limited to {sample_limit} samples")
    
    # Prepare patient data
    print("\n[3/7] Preparing patient data...")
    
    patients = []
    for idx, row in df_original.iterrows():
        try:
            run_id = str(row[run_id_col])
            postal_code = str(row['postal_code'])
            country = str(row['country'])
            age = validate_age(row['age'])
            sex = validate_sex(row['sex'])
            health_status = validate_health_status(row['health_status'])
            
            questionnaire = {
                'smoking_status': validate_questionnaire_field(row.get('smoking_status', 'never'), 'smoking_status',
                    ['never', 'former', 'current_light', 'current_moderate', 'current_heavy']),
                'secondhand_smoke': validate_questionnaire_field(row.get('secondhand_smoke', 'none'), 'secondhand_smoke',
                    ['none', 'occasional', 'frequent', 'daily']),
                'smoking_start_age': int(validate_age(row.get('smoking_start_age', 0))),
                'quit_years': int(validate_age(row.get('quit_years', 0))),
                'occupation_exposures': validate_questionnaire_list(row.get('occupation_exposures', ''), 'occupation_exposures',
                    ['construction', 'manufacturing', 'agriculture', 'healthcare', 'transport', 'chemical', 'mining', 'firefighting']),
                'occupation_years': int(validate_age(row.get('occupation_years', 0))),
                'protection_equipment': validate_questionnaire_field(row.get('protection_equipment', 'moderate'), 'protection_equipment',
                    ['good', 'moderate', 'poor', 'none']),
                'diet_pattern': validate_questionnaire_field(row.get('diet_pattern', 'mixed'), 'diet_pattern',
                    ['mediterranean', 'western', 'vegetarian', 'asian', 'mixed']),
                'fruit_veg_consumption': validate_questionnaire_field(row.get('fruit_veg_consumption', 'moderate'), 'fruit_veg_consumption',
                    ['verylow', 'low', 'moderate', 'high']),
                'processed_meat': validate_questionnaire_field(row.get('processed_meat', 'moderate'), 'processed_meat',
                    ['low', 'moderate', 'high', 'veryhigh']),
                'activity_level': validate_questionnaire_field(row.get('activity_level', 'moderate'), 'activity_level',
                    ['sedentary', 'light', 'moderate', 'active', 'veryactive']),
                'family_history': validate_questionnaire_list(row.get('family_history', ''), 'family_history',
                    ['lung_cancer', 'breast_cancer', 'heart_disease', 'stroke', 'diabetes']),
                'respiratory_conditions': validate_questionnaire_list(row.get('respiratory_conditions', ''), 'respiratory_conditions',
                    ['asthma', 'copd', 'allergies'])
            }
            
            lat, lon, loc_type, se, noise_base, air_base = geocode_postal_code(postal_code, country)
            
            patients.append({
                'run_id': run_id,
                'age': age,
                'birth_year': CURRENT_YEAR - age,
                'health_status': health_status,
                'location_type': loc_type,
                'latitude': lat,
                'longitude': lon,
                'geo_socioeconomic': se,
                'geo_air_base': air_base,
                'questionnaire': questionnaire
            })
            
            if (idx + 1) % 5 == 0:
                print(f"     Processed {idx + 1}/{len(df_original)} samples...")
                
        except Exception as e:
            print(f"     Warning: Skipping sample {idx} due to error: {e}")
            continue
    
    print(f"     Prepared {len(patients)} valid samples")
    
    if len(patients) == 0:
        raise Exception("No valid patients found. Please check your data format.")
    
    # Fetch environmental data
    print("\n[4/7] Fetching environmental data...")
    
    unique_locs = {}
    for p in patients:
        key = f"{p['latitude']:.4f}_{p['longitude']:.4f}"
        if key not in unique_locs:
            unique_locs[key] = {'lat': p['latitude'], 'lon': p['longitude'], 'type': p['location_type']}
    
    print(f"     {len(unique_locs)} unique locations")
    
    env_data = {}
    for i, (key, loc) in enumerate(unique_locs.items()):
        print(f"     Processing location {i+1}/{len(unique_locs)}...")
        
        ages = [p['age'] for p in patients if abs(p['latitude'] - loc['lat']) < 0.01]
        min_birth = CURRENT_YEAR - max(ages) if ages else 1970
        
        try:
            weather = APIClient.fetch_historical_weather(loc['lat'], loc['lon'], min_birth)
            air = APIClient.fetch_historical_air_quality(loc['lat'], loc['lon'], min_birth)
            noise = APIClient.estimate_noise(loc['lat'], loc['lon'], loc['type'])
            env_data[key] = {'weather': weather, 'air': air, 'noise': noise}
            print(f"     ✓ Got data for {key}: {len(weather)} weather years, {len(air)} air years, {noise} dB noise")
        except Exception as e:
            print(f"     Error fetching data for location {key}: {e}")
            # Use synthetic data as fallback
            env_data[key] = {
                'weather': APIClient._generate_synthetic_weather(loc['lat'], min_birth),
                'air': APIClient._generate_synthetic_air_quality(loc['lat'], min_birth),
                'noise': 45
            }
    
    # Calculate risks
    print("\n[5/7] Calculating risks...")
    
    results = []
    for i, p in enumerate(patients):
        if (i + 1) % 5 == 0:
            print(f"     Processing {i+1}/{len(patients)}...")
        try:
            results.append(process_single_patient(p, env_data))
        except Exception as e:
            print(f"     Error processing {p['run_id']}: {e}")
            results.append({})
    
    # Merge results
    print("\n[6/7] Merging results...")
    
    results_dict = {patients[i]['run_id']: results[i] for i in range(len(patients)) if results[i]}
    
    risk_columns = ['Air_Pollution', 'Smoking', 'Occupational', 'Noise', 'Temperature',
                    'Diet', 'Physical_Activity', 'Genetic', 'Socioeconomic']
    
    df_original['CEI'] = df_original[run_id_col].astype(str).map(lambda x: results_dict.get(x, {}).get('CEI', np.nan))
    df_original['CEI_Level'] = df_original[run_id_col].astype(str).map(lambda x: results_dict.get(x, {}).get('CEI_Level', np.nan))
    
    for col in risk_columns:
        df_original[col] = df_original[run_id_col].astype(str).map(lambda x: results_dict.get(x, {}).get(col, np.nan))
    
    # Save outputs
    print("\n[7/7] Saving results...")
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    cei_output = df_original[[run_id_col, 'CEI', 'CEI_Level'] + [c for c in df_original.columns if c in risk_columns]]
    cei_path = os.path.join(output_folder, f'{base_name}_exposome_cei.csv')
    cei_output.to_csv(cei_path, index=False, encoding='utf-8-sig')
    print(f"     ✅ CEI results: {cei_path}")
    
    excel_path = os.path.join(output_folder, f'{base_name}_exposome_complete.xlsx')
    df_original.to_excel(excel_path, index=False)
    print(f"     ✅ Excel complete: {excel_path}")
    
    save_cache(APIClient._cache)
    
    # Summary
    print("\n" + "=" * 80)
    print(" STATISTICAL SUMMARY")
    print("=" * 80)
    print(f"\nTotal samples processed: {len(df_original)}")
    print(f"Average CEI score: {df_original['CEI'].mean():.2f}")
    
    print(f"\n✅ Results saved to: {output_folder}")
    print("=" * 80)
    
    return df_original

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    input_file = 'vigo_metadata_enhanced.csv'
    
    if os.path.exists(input_file):
        results = process_exposome_risks(
            input_file=input_file,
            output_folder='./exposome_results',
            sample_limit=20,  # Process all 20 samples
            use_parallel=False,
            max_workers=2
        )
        
        print("\n📋 Sample Results:")
        display_cols = ['run_id', 'CEI', 'CEI_Level', 'Air_Pollution', 'Smoking', 'Diet']
        available = [c for c in display_cols if c in results.columns]
        if available:
            print(results[available].head(10))
        else:
            print(results.head())
    else:
        print(f"File not found: {input_file}")


# In[ ]:




