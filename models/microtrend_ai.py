"""
AI-powered microtrend detection system with continuous learning capabilities.
Uses machine learning models to predict short-term price movements.
"""

import asyncio
import logging
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    import xgboost as xgb
    from river import tree, preprocessing, metrics
    try:
        from river import forest  # ARFClassifier for online learning
    except ImportError:
        forest = None  # Fallback to tree-based models
except ImportError as e:
    print(f"ML dependencies not installed: {e}")

from src.autopilot import PositionSide
from config.settings import *


@dataclass
class Prediction:
    """Represents a model prediction with confidence and metadata."""
    symbol: str
    side: PositionSide
    confidence: float
    tp_pct: float
    sl_pct: float
    timestamp: datetime
    features_used: Dict
    model_version: str


class FeatureEngineer:
    """Engineering features for microtrend prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    def engineer_features(self, market_data: Dict) -> Optional[np.ndarray]:
        """Engineer features from raw market data."""
        if not market_data or 'current_price' not in market_data:
            return None
            
        try:
            # Extract base features
            features = [
                market_data.get('price_change_1', 0),
                market_data.get('price_change_5', 0),
                market_data.get('price_change_10', 0),
                market_data.get('price_volatility', 0),
                market_data.get('volume_ratio', 1),
                market_data.get('spread_ratio', 1),
                market_data.get('rsi', 50),
                market_data.get('macd', 0),
                market_data.get('bb_position', 0.5),
            ]
            
            # Add derived features
            sma_5 = market_data.get('sma_5', market_data['current_price'])
            sma_10 = market_data.get('sma_10', market_data['current_price'])
            sma_20 = market_data.get('sma_20', market_data['current_price'])
            current_price = market_data['current_price']
            
            # Moving average relationships
            features.extend([
                (current_price - sma_5) / sma_5,
                (current_price - sma_10) / sma_10,
                (current_price - sma_20) / sma_20,
                (sma_5 - sma_10) / sma_10,
                (sma_10 - sma_20) / sma_20,
            ])
            
            # Volume features
            volume_sma = market_data.get('volume_sma_5', 1)
            current_volume = market_data.get('current_volume', 1)
            features.extend([
                np.log1p(current_volume),
                current_volume / max(volume_sma, 1),
            ])
            
            # Time-based features
            timestamp = market_data.get('timestamp', datetime.now())
            if isinstance(timestamp, datetime):
                features.extend([
                    timestamp.hour / 24.0,
                    timestamp.weekday() / 6.0,
                ])
            else:
                features.extend([0.5, 0.5])  # Default values
                
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logging.error(f"Error engineering features: {e}")
            return None
            
    def fit_scaler(self, features_list: List[np.ndarray]):
        """Fit the feature scaler on historical data."""
        if not features_list:
            return
            
        try:
            all_features = np.vstack(features_list)
            self.scaler.fit(all_features)
            self.is_fitted = True
            self.feature_names = [f"feature_{i}" for i in range(all_features.shape[1])]
        except Exception as e:
            logging.error(f"Error fitting scaler: {e}")
            
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using the fitted scaler."""
        if not self.is_fitted or features is None:
            return features
            
        try:
            return self.scaler.transform(features.reshape(1, -1))[0]
        except Exception as e:
            logging.error(f"Error transforming features: {e}")
            return features


class MicroTrendModel:
    """Base class for microtrend prediction models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_data = []
        self.feature_engineer = FeatureEngineer()
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
    def prepare_training_data(self, historical_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical market data."""
        features_list = []
        labels = []
        
        for i, data_point in enumerate(historical_data[:-1]):
            # Engineer features
            features = self.feature_engineer.engineer_features(data_point)
            if features is None:
                continue
                
            # Create label based on next period's price change
            current_price = data_point.get('current_price', 0)
            next_price = historical_data[i + 1].get('current_price', current_price)
            
            if current_price > 0:
                price_change = (next_price - current_price) / current_price
                
                # Multi-class classification: 0=down, 1=neutral, 2=up
                if price_change < -MIN_PRICE_CHANGE:
                    label = 0  # DOWN
                elif price_change > MIN_PRICE_CHANGE:
                    label = 2  # UP
                else:
                    label = 1  # NEUTRAL
                    
                features_list.append(features)
                labels.append(label)
                
        if not features_list:
            return np.array([]), np.array([])
            
        # Fit scaler if not already fitted
        if not self.feature_engineer.is_fitted:
            self.feature_engineer.fit_scaler(features_list)
            
        # Transform features
        X = np.vstack([self.feature_engineer.transform(f) for f in features_list])
        y = np.array(labels)
        
        return X, y
        
    def train(self, historical_data: List[Dict]):
        """Train the model on historical data."""
        raise NotImplementedError
        
    def predict(self, market_data: Dict) -> Optional[Prediction]:
        """Make a prediction on current market data."""
        raise NotImplementedError
        
    def save_model(self, path: str):
        """Save the trained model to disk."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.feature_engineer.scaler,
                'feature_names': self.feature_engineer.feature_names,
                'is_fitted': self.feature_engineer.is_fitted,
                'model_name': self.model_name
            }
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            
    def load_model(self, path: str):
        """Load a trained model from disk."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.feature_engineer.scaler = model_data['scaler']
            self.feature_engineer.feature_names = model_data['feature_names']
            self.feature_engineer.is_fitted = model_data['is_fitted']
            self.is_trained = True
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")


class XGBoostMicroTrend(MicroTrendModel):
    """XGBoost-based microtrend prediction model."""
    
    def __init__(self):
        super().__init__("XGBoost")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
    def train(self, historical_data: List[Dict]):
        """Train XGBoost model."""
        try:
            X, y = self.prepare_training_data(historical_data)
            
            if len(X) < 100:
                self.logger.warning("Insufficient training data")
                return
                
            # Train model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Log training metrics
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            self.logger.info(f"Training accuracy: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            
    def predict(self, market_data: Dict) -> Optional[Prediction]:
        """Make prediction using XGBoost."""
        if not self.is_trained:
            return None
            
        try:
            features = self.feature_engineer.engineer_features(market_data)
            if features is None:
                return None
                
            features = self.feature_engineer.transform(features)
            
            # Get prediction and probability
            prediction = self.model.predict(features.reshape(1, -1))[0]
            probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
            
            confidence = float(np.max(probabilities))
            
            # Skip neutral predictions or low confidence
            if prediction == 1 or confidence < PREDICTION_THRESHOLD:
                return None
                
            side = PositionSide.LONG if prediction == 2 else PositionSide.SHORT
            
            # Dynamic TP/SL based on confidence
            tp_pct = DEFAULT_TP_PCT * confidence
            sl_pct = DEFAULT_SL_PCT * confidence
            
            return Prediction(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                side=side,
                confidence=confidence,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                timestamp=datetime.now(),
                features_used=market_data,
                model_version=self.model_name
            )
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return None


class OnlineLearningModel(MicroTrendModel):
    """Online learning model using River for continuous adaptation."""
    
    def __init__(self):
        super().__init__("OnlineLearning")
        
        # Create online learning pipeline using HoeffdingTreeClassifier
        # River changed their API - AdaptiveRandomForest is now in forest module
        try:
            from river import forest
            self.model = forest.ARFClassifier(
                n_models=10,
                max_features="sqrt",
                lambda_value=6,
                grace_period=50,
                seed=42
            )
        except (ImportError, AttributeError):
            # Fallback to HoeffdingTree if ARF not available
            self.model = tree.HoeffdingTreeClassifier(
                grace_period=50,
                split_confidence=0.01
            )
        
        self.preprocessor = preprocessing.StandardScaler()
        self.metric = metrics.Accuracy()
        self.n_samples = 0
        
    def train(self, historical_data: List[Dict]):
        """Initialize with historical data."""
        try:
            X, y = self.prepare_training_data(historical_data)
            
            if len(X) < 50:
                self.logger.warning("Insufficient training data for online model")
                return
                
            # Train incrementally
            for i, (features, label) in enumerate(zip(X, y)):
                feature_dict = {f'f_{j}': features[j] for j in range(len(features))}
                self.model.learn_one(feature_dict, label)
                
                if i > 10:  # Start making predictions after some training
                    pred = self.model.predict_one(feature_dict)
                    if pred is not None:
                        self.metric.update(label, pred)
                        
            self.is_trained = True
            self.n_samples = len(X)
            self.logger.info(f"Online model trained on {self.n_samples} samples")
            
        except Exception as e:
            self.logger.error(f"Online training error: {e}")
            
    def predict(self, market_data: Dict) -> Optional[Prediction]:
        """Make prediction and learn from result."""
        if not self.is_trained:
            return None
            
        try:
            features = self.feature_engineer.engineer_features(market_data)
            if features is None:
                return None
                
            features = self.feature_engineer.transform(features)
            feature_dict = {f'f_{i}': features[i] for i in range(len(features))}
            
            # Get prediction
            prediction = self.model.predict_one(feature_dict)
            if prediction is None:
                return None
                
            # Get confidence (for ensemble models)
            try:
                proba_dict = self.model.predict_proba_one(feature_dict)
                confidence = max(proba_dict.values()) if proba_dict else 0.5
            except:
                confidence = 0.6  # Default confidence
                
            # Skip neutral predictions or low confidence
            if prediction == 1 or confidence < PREDICTION_THRESHOLD:
                return None
                
            side = PositionSide.LONG if prediction == 2 else PositionSide.SHORT
            
            return Prediction(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                side=side,
                confidence=float(confidence),
                tp_pct=DEFAULT_TP_PCT,
                sl_pct=DEFAULT_SL_PCT,
                timestamp=datetime.now(),
                features_used=market_data,
                model_version=self.model_name
            )
            
        except Exception as e:
            self.logger.error(f"Online prediction error: {e}")
            return None
            
    def learn_from_trade(self, features: Dict, actual_result: str):
        """Learn from actual trade results."""
        try:
            engineered_features = self.feature_engineer.engineer_features(features)
            if engineered_features is None:
                return
                
            features_transformed = self.feature_engineer.transform(engineered_features)
            feature_dict = {f'f_{i}': features_transformed[i] for i in range(len(features_transformed))}
            
            # Convert result to label
            if actual_result == "TAKE_PROFIT":
                label = 2 if features.get('side') == 'LONG' else 0
            elif actual_result == "STOP_LOSS":
                label = 0 if features.get('side') == 'LONG' else 2
            else:
                label = 1  # Neutral
                
            self.model.learn_one(feature_dict, label)
            self.n_samples += 1
            
            if self.n_samples % 100 == 0:
                self.logger.info(f"Online model updated. Total samples: {self.n_samples}")
                
        except Exception as e:
            self.logger.error(f"Error learning from trade: {e}")


class EnsemblePredictor:
    """Ensemble of multiple models for robust predictions."""
    
    def __init__(self):
        self.models = []
        self.weights = []
        self.logger = logging.getLogger(__name__)
        
    def add_model(self, model: MicroTrendModel, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        self.weights.append(weight)
        
    async def predict(self, market_data: Dict) -> Optional[Prediction]:
        """Get ensemble prediction."""
        if not self.models:
            return None
            
        predictions = []
        confidences = []
        
        for model in self.models:
            pred = model.predict(market_data)
            if pred:
                predictions.append(pred)
                confidences.append(pred.confidence)
                
        if not predictions:
            return None
            
        # Weighted voting
        long_votes = sum(w * c for p, w, c in zip(predictions, self.weights, confidences) 
                        if p.side == PositionSide.LONG)
        short_votes = sum(w * c for p, w, c in zip(predictions, self.weights, confidences) 
                         if p.side == PositionSide.SHORT)
        
        if abs(long_votes - short_votes) < 0.1:
            return None  # Too close to call
            
        final_side = PositionSide.LONG if long_votes > short_votes else PositionSide.SHORT
        final_confidence = max(long_votes, short_votes) / sum(self.weights)
        
        if final_confidence < PREDICTION_THRESHOLD:
            return None
            
        return Prediction(
            symbol=market_data.get('symbol', 'UNKNOWN'),
            side=final_side,
            confidence=final_confidence,
            tp_pct=DEFAULT_TP_PCT,
            sl_pct=DEFAULT_SL_PCT,
            timestamp=datetime.now(),
            features_used=market_data,
            model_version="Ensemble"
        )
        
    def train_all(self, historical_data: List[Dict]):
        """Train all models in the ensemble."""
        for model in self.models:
            model.train(historical_data)
            
    def save_ensemble(self, directory: str):
        """Save all models in the ensemble."""
        Path(directory).mkdir(exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = Path(directory) / f"model_{i}_{model.model_name}.pkl"
            model.save_model(str(model_path))
            
    def load_ensemble(self, directory: str):
        """Load all models in the ensemble."""
        model_files = Path(directory).glob("*.pkl")
        
        for model_file in model_files:
            # Determine model type from filename
            if "XGBoost" in model_file.name:
                model = XGBoostMicroTrend()
            elif "OnlineLearning" in model_file.name:
                model = OnlineLearningModel()
            else:
                continue
                
            model.load_model(str(model_file))
            self.models.append(model)