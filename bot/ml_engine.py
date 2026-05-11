import sqlite3
import json
import numpy as np
from datetime import datetime
from loguru import logger
import threading

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn is not installed. ML Engine will be disabled. Run: pip install scikit-learn")

class MLEngine:
    """
    Lightweight Machine Learning engine using scikit-learn.
    Trains on historical signals to predict win probability.
    """
    def __init__(self, db_path: str = "signals.db"):
        self.db_path = db_path
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        self._lock = threading.Lock()
        
        # We define consistent features to extract from the 'reasons' text
        self.reason_keywords = [
            "MACD", "RSI", "SuperTrend", "Bollinger", "CVD", "SMC", 
            "FVG", "Order Block", "Sweep", "Volume", "Trendline", 
            "Structure", "Wave 3"
        ]

    def _extract_features(self, row) -> dict:
        """Extract ML features from a raw DB row."""
        features = {}
        features['score'] = float(row['score'] or 0)
        
        # Time features
        ts_str = row['timestamp']
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            features['hour_utc'] = float(ts.hour)
            features['day_of_week'] = float(ts.weekday())
        except Exception:
            features['hour_utc'] = 12.0
            features['day_of_week'] = 3.0

        # Parse reasons
        reasons_json = row['reasons'] or "[]"
        try:
            reasons = json.loads(reasons_json)
        except Exception:
            reasons = []
            
        reasons_str = " ".join(reasons).upper()
        
        # Boolean keyword flags
        for kw in self.reason_keywords:
            features[f"has_{kw.replace(' ', '_').lower()}"] = 1.0 if kw.upper() in reasons_str else 0.0
            
        # Optional: Order Flow features (if they exist historically)
        features['funding_rate'] = float(row['funding_rate']) if row.get('funding_rate') is not None else 0.0
        
        return features

    def train(self):
        """Train the model asynchronously on closed outcomes."""
        if not SKLEARN_AVAILABLE:
            return False
            
        def _do_train():
            with self._lock:
                try:
                    logger.info("ML Engine: Starting training on historical data...")
                    with sqlite3.connect(self.db_path) as conn:
                        conn.row_factory = sqlite3.Row
                        rows = conn.execute("""
                            SELECT 
                                s.score, s.timestamp, s.reasons, s.funding_rate,
                                so.outcome
                            FROM signals s
                            JOIN signal_outcomes so ON s.id = so.signal_id
                            WHERE so.outcome NOT IN ('OPEN', 'EXITED')
                        """).fetchall()
                    
                    if len(rows) < 100:
                        logger.warning(f"ML Engine: Not enough data to train ({len(rows)} rows). Need at least 100.")
                        return

                    X_list = []
                    y_list = []
                    
                    for r in rows:
                        features = self._extract_features(dict(r))
                        
                        # Label: 1 for Win, 0 for Loss
                        # TP_HIT, TP_NEAR_HIT, TRAIL_HIT usually profit. 
                        # We'll treat SL_HIT as 0, everything else (profit) as 1.
                        outcome = r['outcome']
                        is_win = 1 if outcome != 'SL_HIT' else 0
                        
                        X_list.append(features)
                        y_list.append(is_win)
                    
                    # Ensure consistent column order
                    if len(X_list) == 0:
                        return
                        
                    self.feature_columns = sorted(list(X_list[0].keys()))
                    
                    # Convert to numpy arrays
                    X = np.array([[row[c] for c in self.feature_columns] for row in X_list])
                    y = np.array(y_list)
                    
                    # Train model
                    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                    clf.fit(X, y)
                    
                    self.model = clf
                    self.is_trained = True
                    
                    # Log accuracy on train set (just for info)
                    acc = clf.score(X, y)
                    logger.info(f"ML Engine: Trained RandomForest on {len(rows)} samples. Train accuracy: {acc:.2f}")

                except Exception as e:
                    logger.error(f"ML Engine: Training failed: {e}")

        # Run training in background to not block startup
        t = threading.Thread(target=_do_train)
        t.daemon = True
        t.start()
        return True

    def predict_probability(self, score: float, timestamp: datetime, reasons: list, funding_rate: float = 0.0) -> float:
        """
        Predict the probability of a win for a new signal.
        Returns a float between 0.0 and 1.0. Returns 0.5 if untrained or failed.
        """
        if not self.is_trained or self.model is None or not self.feature_columns:
            return 0.5
            
        try:
            row_dict = {
                'score': score,
                'timestamp': timestamp.isoformat(),
                'reasons': json.dumps([str(r) for r in reasons]),
                'funding_rate': funding_rate
            }
            
            features = self._extract_features(row_dict)
            
            # Convert to numpy array matching training columns
            x = np.array([[features.get(c, 0.0) for c in self.feature_columns]])
            
            # Predict probabilities. class[1] is the win class
            proba = self.model.predict_proba(x)[0]
            
            # If classes array only has 1 element, handle it safely
            if len(self.model.classes_) == 2:
                # Assuming classes are [0, 1]
                idx = list(self.model.classes_).index(1)
                return float(proba[idx])
            else:
                # If only one class existed in training
                if self.model.classes_[0] == 1:
                    return 1.0
                return 0.0
                
        except Exception as e:
            logger.error(f"ML Engine: Prediction failed: {e}")
            return 0.5
