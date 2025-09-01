"""
Medical Gait Classifier Module
==============================

Implements specialized classifiers for medical gait analysis and condition detection.
Includes deep learning models for Hydrocephalus and other neurological disorders.

Author: Medical Gait Analysis System
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import os
import json

logger = logging.getLogger(__name__)

class MedicalGaitClassifier(ABC):
    """
    Abstract base class for medical gait classifiers.
    """
    
    def __init__(self, model_name: str = "medical_gait_classifier"):
        """
        Initialize medical gait classifier.
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = []
        
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build the classification model architecture."""
        pass
    
    @abstractmethod
    def extract_features(self, gait_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from gait data for classification."""
        pass
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None, 
                       fit_scalers: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess features and labels for training/inference.
        
        Args:
            X: Feature matrix
            y: Labels (optional, for inference)
            fit_scalers: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            Tuple of (scaled_features, encoded_labels)
        """
        if fit_scalers:
            X_scaled = self.scaler.fit_transform(X)
            if y is not None:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = None
        else:
            X_scaled = self.scaler.transform(X)
            if y is not None:
                y_encoded = self.label_encoder.transform(y)
            else:
                y_encoded = None
        
        return X_scaled, y_encoded
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2, 
              epochs: int = 100, 
              batch_size: int = 32,
              **kwargs) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix
            y: Labels
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            **kwargs: Additional training parameters
            
        Returns:
            Training history and metrics
        """
        logger.info(f"Training {self.model_name} with {X.shape[0]} samples")
        
        # Preprocess data
        X_scaled, y_encoded = self.preprocess_data(X, y, fit_scalers=True)
        
        # Build model if not exists
        if self.model is None:
            num_classes = len(np.unique(y_encoded))
            self.model = self.build_model(X_scaled.shape[1:], num_classes)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded, test_size=validation_split, 
            stratify=y_encoded, random_state=42
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            **kwargs
        )
        
        self.is_trained = True
        
        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(val_pred_classes == y_val),
            'classification_report': classification_report(y_val, val_pred_classes),
            'confusion_matrix': confusion_matrix(y_val, val_pred_classes).tolist()
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_encoded)) == 2:
            metrics['auc'] = roc_auc_score(y_val, val_predictions[:, 1])
        
        logger.info(f"Training completed. Validation accuracy: {metrics['accuracy']:.4f}")
        
        return {
            'history': history.history,
            'metrics': metrics
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predicted_classes, prediction_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled, _ = self.preprocess_data(X, fit_scalers=False)
        
        predictions = self.model.predict(X_scaled)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        return predicted_labels, predictions
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessors."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(f"{filepath}_model.h5")
        
        # Save preprocessors
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.label_encoder, f"{filepath}_label_encoder.pkl")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessors."""
        # Load model
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        
        # Load preprocessors
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.label_encoder = joblib.load(f"{filepath}_label_encoder.pkl")
        
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata['model_name']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        logger.info(f"Model loaded from {filepath}")

class HydrocephalusClassifier(MedicalGaitClassifier):
    """
    Specialized classifier for Hydrocephalus detection from gait patterns.
    
    Uses a combination of temporal convolutional networks and traditional ML
    to detect Hydrocephalus-specific gait abnormalities.
    """
    
    def __init__(self):
        super().__init__("hydrocephalus_classifier")
        
        # Hydrocephalus-specific feature weights
        self.feature_weights = {
            'shuffling_percentage': 2.0,
            'magnetic_gait_percentage': 2.5,
            'step_height_cv': 1.5,
            'stride_time_cv': 1.3,
            'freezing_episode_count': 1.8,
            'wide_base_percentage': 1.4
        }
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """
        Build a deep neural network for Hydrocephalus classification.
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # Feature extraction layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Classification layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with appropriate loss and metrics
        if num_classes == 2:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def extract_features(self, gait_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract Hydrocephalus-specific features from gait data.
        
        Args:
            gait_data: Dictionary containing gait metrics and events
            
        Returns:
            Feature vector for classification
        """
        features = []
        feature_names = []
        
        # Extract temporal parameters
        temporal_features = [
            'stride_time_mean', 'stride_time_std', 'stride_time_cv',
            'step_time_mean', 'step_time_std', 'step_time_asymmetry',
            'stance_time_mean', 'stance_time_std',
            'swing_time_mean', 'swing_time_std'
        ]
        
        for feature in temporal_features:
            if feature in gait_data:
                features.append(gait_data[feature])
                feature_names.append(feature)
            else:
                features.append(0.0)
                feature_names.append(feature)
        
        # Extract Hydrocephalus-specific features
        hydrocephalus_features = [
            'shuffling_percentage', 'shuffling_episode_count',
            'magnetic_gait_percentage', 'magnetic_gait_episode_count',
            'freezing_episode_count', 'mean_freezing_duration', 'max_freezing_duration',
            'wide_base_percentage',
            'step_height_mean', 'step_height_std', 'step_height_cv'
        ]
        
        for feature in hydrocephalus_features:
            if feature in gait_data:
                value = gait_data[feature]
                # Apply feature weighting for important Hydrocephalus indicators
                if feature in self.feature_weights:
                    value *= self.feature_weights[feature]
                features.append(value)
                feature_names.append(feature)
            else:
                features.append(0.0)
                feature_names.append(feature)
        
        # Store feature names for interpretability
        self.feature_names = feature_names
        
        return np.array(features)
    
    def predict_hydrocephalus_severity(self, gait_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict Hydrocephalus severity and provide detailed analysis.
        
        Args:
            gait_data: Dictionary containing gait metrics and events
            
        Returns:
            Dictionary with severity prediction and analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_features(gait_data).reshape(1, -1)
        
        # Make prediction
        predicted_class, probabilities = self.predict(features)
        
        # Calculate severity score based on key indicators
        severity_indicators = {
            'shuffling_severity': gait_data.get('shuffling_percentage', 0) / 100,
            'magnetic_gait_severity': gait_data.get('magnetic_gait_percentage', 0) / 100,
            'freezing_severity': min(gait_data.get('freezing_episode_count', 0) / 10, 1.0),
            'step_variability': min(gait_data.get('step_height_cv', 0), 1.0),
            'wide_base_severity': gait_data.get('wide_base_percentage', 0) / 100
        }
        
        overall_severity = np.mean(list(severity_indicators.values()))
        
        # Classify severity level
        if overall_severity < 0.2:
            severity_level = "Mild"
        elif overall_severity < 0.5:
            severity_level = "Moderate"
        else:
            severity_level = "Severe"
        
        return {
            'predicted_class': predicted_class[0],
            'confidence': float(np.max(probabilities)),
            'class_probabilities': {
                f'class_{i}': float(prob) for i, prob in enumerate(probabilities[0])
            },
            'severity_level': severity_level,
            'severity_score': float(overall_severity),
            'severity_indicators': severity_indicators,
            'clinical_recommendations': self._generate_clinical_recommendations(
                severity_level, severity_indicators
            )
        }
    
    def _generate_clinical_recommendations(self, severity_level: str, 
                                         indicators: Dict[str, float]) -> List[str]:
        """Generate clinical recommendations based on severity analysis."""
        recommendations = []
        
        if severity_level == "Severe":
            recommendations.append("Immediate neurological consultation recommended")
            recommendations.append("Consider CSF shunt evaluation")
            
        if indicators['shuffling_severity'] > 0.5:
            recommendations.append("Physical therapy for gait training")
            recommendations.append("Fall prevention measures")
            
        if indicators['magnetic_gait_severity'] > 0.4:
            recommendations.append("Evaluate for NPH (Normal Pressure Hydrocephalus)")
            recommendations.append("Consider lumbar puncture test")
            
        if indicators['freezing_severity'] > 0.3:
            recommendations.append("Gait initiation training")
            recommendations.append("Environmental modifications")
            
        if indicators['step_variability'] > 0.6:
            recommendations.append("Balance training exercises")
            recommendations.append("Assistive device evaluation")
            
        if not recommendations:
            recommendations.append("Continue regular monitoring")
            recommendations.append("Maintain current treatment plan")
        
        return recommendations

class EnsembleMedicalClassifier:
    """
    Ensemble classifier combining multiple models for robust medical gait analysis.
    """
    
    def __init__(self, models: List[MedicalGaitClassifier]):
        """
        Initialize ensemble classifier.
        
        Args:
            models: List of trained medical gait classifiers
        """
        self.models = models
        self.weights = np.ones(len(models)) / len(models)  # Equal weights initially
    
    def predict(self, gait_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make ensemble predictions using all models.
        
        Args:
            gait_data: Dictionary containing gait metrics and events
            
        Returns:
            Ensemble prediction results
        """
        predictions = []
        probabilities = []
        
        for model in self.models:
            if model.is_trained:
                features = model.extract_features(gait_data).reshape(1, -1)
                pred_class, pred_probs = model.predict(features)
                predictions.append(pred_class[0])
                probabilities.append(pred_probs[0])
        
        if not predictions:
            raise ValueError("No trained models available for prediction")
        
        # Weighted voting
        probabilities = np.array(probabilities)
        weighted_probs = np.average(probabilities, axis=0, weights=self.weights[:len(probabilities)])
        ensemble_class = np.argmax(weighted_probs)
        
        return {
            'ensemble_prediction': ensemble_class,
            'ensemble_confidence': float(np.max(weighted_probs)),
            'individual_predictions': predictions,
            'weighted_probabilities': weighted_probs.tolist(),
            'model_agreement': len(set(predictions)) == 1
        }
    
    def update_weights(self, validation_accuracies: List[float]):
        """Update model weights based on validation performance."""
        accuracies = np.array(validation_accuracies)
        # Weight models by their relative performance
        self.weights = accuracies / np.sum(accuracies)
        logger.info(f"Updated ensemble weights: {self.weights}")
