"""
Gait Analysis Training and Evaluation Module
============================================

This module handles the complete training and evaluation pipeline for
gait analysis using TCN models, including cross-validation and custom metrics.

Author: Gait Analysis System
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from datetime import datetime
import json

from .gait_data_preprocessing import GaitDataPreprocessor
from .tcn_gait_model import create_gait_tcn_model, compile_gait_model

logger = logging.getLogger(__name__)

class GaitMetrics:
    """
    Custom metrics for gait analysis evaluation.
    """
    
    @staticmethod
    def time_deviation_mae(y_true: np.ndarray, y_pred: np.ndarray, fps: float = 30.0) -> float:
        """
        Calculate Mean Absolute Error in time deviation (milliseconds).
        
        Args:
            y_true: True event timestamps
            y_pred: Predicted event timestamps
            fps: Frames per second
            
        Returns:
            MAE in milliseconds
        """
        frame_error = np.abs(y_true - y_pred)
        time_error_ms = (frame_error / fps) * 1000
        return np.mean(time_error_ms)
    
    @staticmethod
    def gait_phase_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate overall accuracy for gait phase detection.
        
        Args:
            y_true: True phase labels
            y_pred: Predicted phase labels
            
        Returns:
            Overall accuracy
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def phase_transition_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy specifically for phase transitions.
        
        Args:
            y_true: True phase labels
            y_pred: Predicted phase labels
            
        Returns:
            Phase transition accuracy
        """
        # Find phase transitions in true labels
        transitions = np.diff(y_true) != 0
        if np.sum(transitions) == 0:
            return 1.0  # No transitions to evaluate
        
        # Calculate accuracy at transition points
        transition_accuracy = np.mean(y_true[1:][transitions] == y_pred[1:][transitions])
        return transition_accuracy

class GaitTrainer:
    """
    Comprehensive trainer for gait analysis TCN models.
    """
    
    def __init__(self,
                 data_preprocessor: GaitDataPreprocessor,
                 model_config: Dict[str, Any],
                 task_type: str = 'phase_detection',
                 output_dir: str = 'gait_analysis_results'):
        """
        Initialize the gait trainer.
        
        Args:
            data_preprocessor: Preprocessor instance
            model_config: Model configuration dictionary
            task_type: Type of task ('phase_detection' or 'event_detection')
            output_dir: Directory to save results
        """
        self.data_preprocessor = data_preprocessor
        self.model_config = model_config
        self.task_type = task_type
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training history
        self.training_history = []
        self.best_models = []
        
        logger.info(f"Initialized GaitTrainer for {task_type}")
    
    def prepare_data(self, data_paths: List[str], labels: Optional[List] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training from multiple video sequences.
        
        Args:
            data_paths: List of paths to OpenPose JSON directories
            labels: Optional labels for supervised learning
            
        Returns:
            Tuple of (features, labels)
        """
        all_windows = []
        all_labels = []
        
        for i, data_path in enumerate(data_paths):
            try:
                # Process video sequence
                result = self.data_preprocessor.process_video_sequence(data_path)
                windows = result['windows']
                
                # Use provided labels or default to 0
                if labels is not None and i < len(labels):
                    sequence_labels = [labels[i]] * len(windows)
                else:
                    sequence_labels = [0] * len(windows)
                
                all_windows.append(windows)
                all_labels.extend(sequence_labels)
                
                logger.info(f"Processed {data_path}: {len(windows)} windows")
                
            except Exception as e:
                logger.error(f"Error processing {data_path}: {e}")
                continue
        
        if not all_windows:
            raise ValueError("No valid data processed")
        
        # Concatenate all windows
        features = np.vstack(all_windows)
        labels = np.array(all_labels)
        
        logger.info(f"Total data prepared: {features.shape[0]} samples, {features.shape[2]} features")
        
        return features, labels
    
    def create_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Create and compile TCN model.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled model
        """
        model = create_gait_tcn_model(
            input_shape=input_shape,
            task_type=self.task_type,
            **self.model_config
        )
        
        model = compile_gait_model(
            model=model,
            task_type=self.task_type,
            learning_rate=self.model_config.get('learning_rate', 0.001)
        )
        
        return model
    
    def train_with_cross_validation(self,
                                   features: np.ndarray,
                                   labels: np.ndarray,
                                   n_folds: int = 5,
                                   epochs: int = 100,
                                   batch_size: int = 32,
                                   validation_split: float = 0.2,
                                   early_stopping_patience: int = 15,
                                   **kwargs) -> Dict[str, Any]:
        """
        Train model with k-fold cross-validation.
        
        Args:
            features: Input features
            labels: Target labels
            n_folds: Number of cross-validation folds
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
            early_stopping_patience: Early stopping patience
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(f"Starting {n_folds}-fold cross-validation")
        
        # Initialize cross-validation
        if self.task_type == 'phase_detection':
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'fold_scores': [],
            'fold_histories': [],
            'best_models': [],
            'predictions': [],
            'true_labels': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(features, labels)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Split data
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Create model for this fold
            model = self.create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            callbacks = self._create_callbacks(
                fold=fold,
                early_stopping_patience=early_stopping_patience,
                **kwargs
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            y_pred = model.predict(X_test, verbose=0)
            
            # Store results
            cv_results['fold_scores'].append({
                'fold': fold + 1,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            })
            cv_results['fold_histories'].append(history.history)
            cv_results['best_models'].append(model)
            cv_results['predictions'].append(y_pred)
            cv_results['true_labels'].append(y_test)
            
            logger.info(f"Fold {fold + 1} - Test Accuracy: {test_accuracy:.4f}")
        
        # Calculate overall metrics
        cv_results['overall_metrics'] = self._calculate_overall_metrics(cv_results)
        
        # Save results
        self._save_cv_results(cv_results)
        
        return cv_results
    
    def _create_callbacks(self, fold: int, early_stopping_patience: int, **kwargs) -> List:
        """Create training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'model_fold_{fold + 1}.h5')
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Learning rate reduction
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        return callbacks
    
    def _calculate_overall_metrics(self, cv_results: Dict) -> Dict[str, float]:
        """Calculate overall cross-validation metrics."""
        accuracies = [score['test_accuracy'] for score in cv_results['fold_scores']]
        losses = [score['test_loss'] for score in cv_results['fold_scores']]
        
        # Concatenate all predictions and true labels
        all_predictions = np.concatenate(cv_results['predictions'])
        all_true_labels = np.concatenate(cv_results['true_labels'])
        
        # Convert predictions to class labels
        if self.task_type == 'phase_detection':
            y_pred_classes = np.argmax(all_predictions, axis=1)
        else:
            y_pred_classes = (all_predictions > 0.5).astype(int)
        
        # Calculate metrics
        overall_metrics = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'overall_accuracy': GaitMetrics.gait_phase_accuracy(all_true_labels, y_pred_classes),
            'f1_score': f1_score(all_true_labels, y_pred_classes, average='weighted'),
            'precision': precision_score(all_true_labels, y_pred_classes, average='weighted'),
            'recall': recall_score(all_true_labels, y_pred_classes, average='weighted')
        }
        
        return overall_metrics
    
    def _save_cv_results(self, cv_results: Dict):
        """Save cross-validation results."""
        # Save metrics
        metrics_file = os.path.join(self.output_dir, 'cv_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(cv_results['overall_metrics'], f, indent=2)
        
        # Save fold scores
        fold_scores_file = os.path.join(self.output_dir, 'fold_scores.json')
        with open(fold_scores_file, 'w') as f:
            json.dump(cv_results['fold_scores'], f, indent=2)
        
        # Save training histories
        histories_file = os.path.join(self.output_dir, 'training_histories.json')
        with open(histories_file, 'w') as f:
            json.dump(cv_results['fold_histories'], f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def plot_training_curves(self, cv_results: Dict):
        """Plot training curves for all folds."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for fold, history in enumerate(cv_results['fold_histories']):
            # Loss curves
            axes[0, 0].plot(history['loss'], label=f'Fold {fold + 1} - Train')
            axes[0, 0].plot(history['val_loss'], label=f'Fold {fold + 1} - Val', linestyle='--')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy curves
            axes[0, 1].plot(history['accuracy'], label=f'Fold {fold + 1} - Train')
            axes[0, 1].plot(history['val_accuracy'], label=f'Fold {fold + 1} - Val', linestyle='--')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Overall metrics
        metrics = cv_results['overall_metrics']
        metric_names = ['mean_accuracy', 'f1_score', 'precision', 'recall']
        metric_values = [metrics[name] for name in metric_names]
        
        axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_title('Overall Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Fold accuracies
        fold_accuracies = [score['test_accuracy'] for score in cv_results['fold_scores']]
        fold_numbers = [f'Fold {i+1}' for i in range(len(fold_accuracies))]
        
        axes[1, 1].bar(fold_numbers, fold_accuracies)
        axes[1, 1].set_title('Fold Accuracies')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model_performance(self, cv_results: Dict):
        """Comprehensive model performance evaluation."""
        logger.info("Evaluating model performance...")
        
        # Overall metrics
        metrics = cv_results['overall_metrics']
        logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        
        # Per-fold results
        logger.info("\nPer-fold results:")
        for score in cv_results['fold_scores']:
            logger.info(f"Fold {score['fold']}: Accuracy = {score['test_accuracy']:.4f}, Loss = {score['test_loss']:.4f}")
        
        # Confusion matrix
        all_predictions = np.concatenate(cv_results['predictions'])
        all_true_labels = np.concatenate(cv_results['true_labels'])
        
        if self.task_type == 'phase_detection':
            y_pred_classes = np.argmax(all_predictions, axis=1)
        else:
            y_pred_classes = (all_predictions > 0.5).astype(int)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(all_true_labels, y_pred_classes)
        
        # Classification report
        if self.task_type == 'phase_detection':
            class_names = ['Stance', 'Swing', 'Double Support', 'Other']
        else:
            class_names = ['No Event', 'Event']
        
        report = classification_report(all_true_labels, y_pred_classes, target_names=class_names)
        logger.info(f"\nClassification Report:\n{report}")
        
        # Save detailed report
        report_file = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_file, 'w') as f:
            f.write(f"Gait Analysis Classification Report\n")
            f.write(f"Task Type: {self.task_type}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Overall Metrics:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write(f"\nClassification Report:\n{report}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        if self.task_type == 'phase_detection':
            class_names = ['Stance', 'Swing', 'Double Support', 'Other']
        else:
            class_names = ['No Event', 'Event']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of the gait training pipeline."""
    # Initialize preprocessor
    preprocessor = GaitDataPreprocessor(
        confidence_threshold=0.3,
        filter_cutoff=6.0,
        window_size=30
    )
    
    # Model configuration
    model_config = {
        'num_classes': 4,  # For phase detection
        'num_filters': 64,
        'kernel_size': 3,
        'num_blocks': 4,
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    }
    
    # Initialize trainer
    trainer = GaitTrainer(
        data_preprocessor=preprocessor,
        model_config=model_config,
        task_type='phase_detection',
        output_dir='gait_analysis_results'
    )
    
    # Example data paths (replace with actual paths)
    # data_paths = ['path/to/video1/json', 'path/to/video2/json', ...]
    # labels = [0, 1, 0, 1, ...]  # Corresponding labels
    
    # Prepare data
    # features, labels = trainer.prepare_data(data_paths, labels)
    
    # Train with cross-validation
    # cv_results = trainer.train_with_cross_validation(
    #     features=features,
    #     labels=labels,
    #     n_folds=5,
    #     epochs=100,
    #     batch_size=32
    # )
    
    # Evaluate performance
    # trainer.evaluate_model_performance(cv_results)
    # trainer.plot_training_curves(cv_results)
    
    print("Gait training pipeline ready!")

if __name__ == "__main__":
    main()
