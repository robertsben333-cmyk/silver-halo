import os
import json
import math
import requests
import statistics
from datetime import datetime, timedelta

def load_coefficients(strategy):
    """Load coefficients from JSON file based on strategy."""
    # Assume script is in root/apps/ or similar depth, try to find lm_models relative to it
    # We want to be robust about finding the project root.
    # If this file is in apps/model_utils.py, then project root is ../
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "lm_models", strategy, "coefficients.json")
    
    if not os.path.exists(path):
        print(f"Warning: Coefficients file not found at {path}. Using defaults or returning None.")
        return None

    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading coefficients: {e}")
        return None

def fetch_historic_volume(ticker, date_str, api_key):
    """
    Fetch 30-day average dollar volume ending at (and including) date_str.
    date_str should be YYYY-MM-DD.
    """
    try:
        end_dt = datetime.strptime(date_str, "%Y-%m-%d")
        # Go back ~45 days to ensure we get 30 trading days
        start_dt = end_dt - timedelta(days=45) 
        start_date_str = start_dt.strftime("%Y-%m-%d")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date_str}/{date_str}"
        params = {
            "apiKey": api_key,
            "adjusted": "true",
            "sort": "desc",
            "limit": 30
        }
        
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if not results:
                return 0
            
            total_vol = 0
            count = 0
            for r in results:
                # Dollar Volume = Close * Volume
                vol = r.get('v', 0) * r.get('c', 0)
                total_vol += vol
                count += 1
            
            if count > 0:
                return total_vol / count
    except Exception as e:
        print(f"Error fetching volume for {ticker}: {e}")
    return 0

def calculate_model_prediction(metrics, final_score, confidence, uncertainty, volume, coef_data):
    """Calculates predicted return using loaded coefficients."""
    if not coef_data:
        return 0.0
        
    intercept = coef_data.get("intercept", 0.0)
    coefs = coef_data.get("coefficients", {})
    
    pred = intercept
    
    # helper for safe get
    def get_coef(name):
        return coefs.get(name, 0.0)
    
    # Continuous Features
    pred += metrics.get('PCR', 0) * get_coef('metrics_PCR')
    pred += metrics.get('NRI', 0) * get_coef('metrics_NRI')
    pred += metrics.get('CP', 0.5) * get_coef('metrics_CP')
    pred += metrics.get('HDM', 0) * get_coef('metrics_HDM')
    pred += metrics.get('EC', 0) * get_coef('metrics_EC')
    pred += metrics.get('RD', 0) * get_coef('metrics_RD')
    pred += metrics.get('FRESH_NEG', 0) * get_coef('metrics_FRESH_NEG')
    pred += metrics.get('SD', 0) * get_coef('metrics_SD')
    pred += metrics.get('CONTR', 0) * get_coef('metrics_CONTR')
    
    pred += final_score * get_coef('finalScore')
    pred += metrics.get('oneDayReturnPct', 0) * get_coef('oneDayReturnPct')
    
    # Volume
    pred += volume * get_coef('AvgVolume_30D')

    # Categorical
    # Confidence
    conf_key = f"confidence_{confidence}"
    pred += get_coef(conf_key)
    
    # Uncertainty
    unc_key = f"uncertainty_{uncertainty}"
    pred += get_coef(unc_key)
        
    return pred
