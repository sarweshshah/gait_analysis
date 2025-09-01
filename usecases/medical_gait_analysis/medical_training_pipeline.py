"""
Medical Gait Training Pipeline Module
====================================

Comprehensive training pipeline for medical gait analysis models.
Handles data preparation, model training, validation, and evaluation for medical conditions.

Author: Medical Gait Analysis System
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import joblib

from .medical_gait_classifier import HydrocephalusClassifier, MedicalGaitClassifier
from .medical_data_processor import MedicalGaitDataProcessor

logger = logging.getLogger(__name__)

class MedicalGaitTrainingPipeline:
    """
    Complete training pipeline for medical gait analysis models.
    
    Features:
    - Automated data preprocessing and feature extraction
    - Cross-validation with stratified splits
    - Model training with early stopping and callbacks
    - Comprehensive evaluation and visualization
    - Model persistence and deployment preparation
    """
    
    def __init__(self, 
                 medical_condition: str = "hydrocephalus",
                 output_dir: str = "medical_gait_models",
                 random_state: int = 42):
        """
        Initialize the medical gait training pipeline.
        
        Args:
            medical_condition: Target medical condition
            output_dir: Directory to save models and results
            random_state: Random seed for reproducibility
        """
        self.medical_condition = medical_condition.lower()
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_processor = MedicalGaitDataProcessor(medical_condition=medical_condition)
        
        if medical_condition.lower() == "hydrocephalus":
            self.classifier = HydrocephalusClassifier()
        else:
            self.classifier = MedicalGaitClassifier()
        
        # Training history and results
        self.training_history = {}
        self.evaluation_results = {}
        self.best_model_path = None
        
        logger.info(f"Initialized MedicalGaitTrainingPipeline for {medical_condition}")
    
    def prepare_dataset(self, 
                       data_directory: str,
                       labels_file: str = None,
                       test_size: float = 0.2,
                       validation_size: float = 0.2) -> Dict[str, Any]:
        """
        Prepare the complete dataset for training.
        
        Args:
            data_directory: Directory containing patient video data
            labels_file: CSV file with patient labels and metadata
            test_size: Fraction of data for testing
            validation_size: Fraction of training data for validation
            
        Returns:
            Dictionary containing prepared datasets
        """
        logger.info("Preparing medical gait dataset...")
        
        # Create dataset from patient videos
        features, labels, metadata = self.data_processor.create_medical_dataset(
            data_directory, labels_file
        )
        
        logger.info(f"Dataset created: {features.shape[0]} patients, {features.shape[1]} features")
        
        # Split into train/test
        X_train_val, X_test, y_train_val, y_test, meta_train_val, meta_test = train_test_split(
            features, labels, metadata, 
            test_size=test_size, 
            stratify=labels,
            random_state=self.random_state
        )
        
        # Split train into train/validation
        X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
            X_train_val, y_train_val, meta_train_val,
            test_size=validation_size,
            stratify=y_train_val,
            random_state=self.random_state
        )
        
        # Store dataset information
        dataset_info = {
            'total_samples': len(features),
            'n_features': features.shape[1],
            'n_classes': len(np.unique(labels)),
            'class_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
        
        logger.info(f"Dataset splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Class distribution: {dataset_info['class_distribution']}")
        
        return {
            'X_train': X_train, 'y_train': y_train, 'meta_train': meta_train,
            'X_val': X_val, 'y_val': y_val, 'meta_val': meta_val,
            'X_test': X_test, 'y_test': y_test, 'meta_test': meta_test,
            'dataset_info': dataset_info
        }
    
    def train_model(self, 
                   dataset: Dict[str, Any],
                   epochs: int = 100,
                   batch_size: int = 32,
                   early_stopping_patience: int = 15,
                   reduce_lr_patience: int = 10,
                   **kwargs) -> Dict[str, Any]:
        """
        Train the medical gait classification model.
        
        Args:
            dataset: Prepared dataset dictionary
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            **kwargs: Additional training parameters
            
        Returns:
            Training results and history
        """
        logger.info("Starting model training...")
        
        X_train, y_train = dataset['X_train'], dataset['y_train']
        X_val, y_val = dataset['X_val'], dataset['y_val']
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / f'best_model_{self.medical_condition}.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        training_results = self.classifier.train(
            X_train, y_train,
            validation_split=0.0,  # We already have validation set
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )
        
        # Evaluate on validation set
        val_predictions, val_probabilities = self.classifier.predict(X_val)
        val_accuracy = np.mean(val_predictions == y_val)
        
        logger.info(f"Training completed. Validation accuracy: {val_accuracy:.4f}")
        
        # Store training history
        self.training_history = training_results['history']
        self.best_model_path = str(self.output_dir / f'best_model_{self.medical_condition}.h5')
        
        return {
            'training_history': training_results['history'],
            'validation_accuracy': val_accuracy,
            'validation_predictions': val_predictions,
            'validation_probabilities': val_probabilities,
            'best_model_path': self.best_model_path
        }
    
    def cross_validate_model(self, 
                           dataset: Dict[str, Any],
                           n_folds: int = 5,
                           epochs: int = 100,
                           batch_size: int = 32) -> Dict[str, Any]:
        """
        Perform cross-validation for robust model evaluation.
        
        Args:
            dataset: Prepared dataset dictionary
            n_folds: Number of cross-validation folds
            epochs: Number of training epochs per fold
            batch_size: Training batch size
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {n_folds}-fold cross-validation...")
        
        # Combine train and validation sets for CV
        X = np.vstack([dataset['X_train'], dataset['X_val']])
        y = np.hstack([dataset['y_train'], dataset['y_val']])
        
        cv_results = {
            'fold_scores': [],
            'fold_histories': [],
            'mean_accuracy': 0,
            'std_accuracy': 0,
            'all_predictions': [],
            'all_true_labels': []
        }
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Create new classifier for this fold
            if self.medical_condition == "hydrocephalus":
                fold_classifier = HydrocephalusClassifier()
            else:
                fold_classifier = MedicalGaitClassifier()
            
            # Train fold model
            fold_results = fold_classifier.train(
                X_fold_train, y_fold_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            # Evaluate fold
            fold_predictions, fold_probabilities = fold_classifier.predict(X_fold_val)
            fold_accuracy = np.mean(fold_predictions == y_fold_val)
            
            # Store results
            cv_results['fold_scores'].append({
                'fold': fold + 1,
                'accuracy': fold_accuracy,
                'predictions': fold_predictions,
                'probabilities': fold_probabilities,
                'true_labels': y_fold_val
            })
            cv_results['fold_histories'].append(fold_results['history'])
            cv_results['all_predictions'].extend(fold_predictions)
            cv_results['all_true_labels'].extend(y_fold_val)
            
            logger.info(f"Fold {fold + 1} accuracy: {fold_accuracy:.4f}")
        
        # Calculate overall metrics
        fold_accuracies = [score['accuracy'] for score in cv_results['fold_scores']]
        cv_results['mean_accuracy'] = np.mean(fold_accuracies)
        cv_results['std_accuracy'] = np.std(fold_accuracies)
        
        logger.info(f"Cross-validation completed. Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        
        return cv_results
    
    def evaluate_model(self, 
                      dataset: Dict[str, Any],
                      save_plots: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation on test set.
        
        Args:
            dataset: Prepared dataset dictionary
            save_plots: Whether to save evaluation plots
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Evaluating model on test set...")
        
        X_test, y_test = dataset['X_test'], dataset['y_test']
        
        # Make predictions
        test_predictions, test_probabilities = self.classifier.predict(X_test)
        
        # Calculate metrics
        test_accuracy = np.mean(test_predictions == y_test)
        
        # Classification report
        class_report = classification_report(
            y_test, test_predictions, 
            target_names=[f'Class_{i}' for i in range(len(np.unique(y_test)))],
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, test_predictions)
        
        evaluation_results = {
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': test_predictions.tolist(),
            'probabilities': test_probabilities.tolist(),
            'true_labels': y_test.tolist()
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_test)) == 2:
            auc_score = roc_auc_score(y_test, test_probabilities[:, 1])
            evaluation_results['auc_score'] = auc_score
            logger.info(f"Test AUC: {auc_score:.4f}")
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate evaluation plots
        if save_plots:
            self._generate_evaluation_plots(evaluation_results, dataset['dataset_info'])
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def _generate_evaluation_plots(self, results: Dict[str, Any], dataset_info: Dict[str, Any]):
        """Generate and save evaluation plots."""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        conf_matrix = np.array(results['confusion_matrix'])
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'Class_{i}' for i in range(len(conf_matrix))],
                   yticklabels=[f'Class_{i}' for i in range(len(conf_matrix))])
        plt.title(f'Confusion Matrix - {self.medical_condition.title()} Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{self.medical_condition}.png', dpi=300)
        plt.close()
        
        # 2. ROC Curve (for binary classification)
        if 'auc_score' in results:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(results['true_labels'], 
                                  np.array(results['probabilities'])[:, 1])
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {results["auc_score"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.medical_condition.title()} Classification')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'roc_curve_{self.medical_condition}.png', dpi=300)
            plt.close()
        
        # 3. Training History (if available)
        if hasattr(self, 'training_history') and self.training_history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Accuracy plot
            if 'accuracy' in self.training_history:
                ax1.plot(self.training_history['accuracy'], label='Training Accuracy')
                ax1.plot(self.training_history['val_accuracy'], label='Validation Accuracy')
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Loss plot
            if 'loss' in self.training_history:
                ax2.plot(self.training_history['loss'], label='Training Loss')
                ax2.plot(self.training_history['val_loss'], label='Validation Loss')
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'training_history_{self.medical_condition}.png', dpi=300)
            plt.close()
        
        logger.info(f"Evaluation plots saved to {self.output_dir}")
    
    def save_pipeline_results(self, 
                            dataset: Dict[str, Any],
                            training_results: Dict[str, Any] = None,
                            cv_results: Dict[str, Any] = None):
        """
        Save complete pipeline results and model artifacts.
        
        Args:
            dataset: Dataset information
            training_results: Training results
            cv_results: Cross-validation results
        """
        logger.info("Saving pipeline results...")
        
        # Save the trained classifier
        model_path = self.output_dir / f'{self.medical_condition}_classifier'
        self.classifier.save_model(str(model_path))
        
        # Compile comprehensive results
        pipeline_results = {
            'medical_condition': self.medical_condition,
            'dataset_info': dataset['dataset_info'],
            'model_architecture': str(self.classifier.model.summary()) if self.classifier.model else None,
            'evaluation_results': self.evaluation_results,
            'training_results': training_results,
            'cross_validation_results': cv_results,
            'feature_names': self.classifier.feature_names,
            'model_path': str(model_path)
        }
        
        # Save results as JSON
        results_file = self.output_dir / f'pipeline_results_{self.medical_condition}.json'
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        # Save dataset splits for reproducibility
        dataset_file = self.output_dir / f'dataset_splits_{self.medical_condition}.npz'
        np.savez(dataset_file,
                X_train=dataset['X_train'], y_train=dataset['y_train'],
                X_val=dataset['X_val'], y_val=dataset['y_val'],
                X_test=dataset['X_test'], y_test=dataset['y_test'])
        
        logger.info(f"Pipeline results saved to {self.output_dir}")
        
        # Generate summary report
        self._generate_summary_report(pipeline_results)
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate a human-readable summary report."""
        
        report_lines = [
            f"Medical Gait Analysis Pipeline Report",
            f"=" * 50,
            f"",
            f"Medical Condition: {self.medical_condition.title()}",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"Dataset Summary:",
            f"- Total Patients: {results['dataset_info']['total_samples']}",
            f"- Features: {results['dataset_info']['n_features']}",
            f"- Classes: {results['dataset_info']['n_classes']}",
            f"- Class Distribution: {results['dataset_info']['class_distribution']}",
            f"",
            f"Model Performance:",
        ]
        
        if self.evaluation_results:
            report_lines.extend([
                f"- Test Accuracy: {self.evaluation_results['test_accuracy']:.4f}",
                f"- Classification Report:",
            ])
            
            # Add classification metrics
            class_report = self.evaluation_results['classification_report']
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    report_lines.append(
                        f"  {class_name}: Precision={metrics['precision']:.3f}, "
                        f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}"
                    )
            
            if 'auc_score' in self.evaluation_results:
                report_lines.append(f"- AUC Score: {self.evaluation_results['auc_score']:.4f}")
        
        report_lines.extend([
            f"",
            f"Files Generated:",
            f"- Model: {self.medical_condition}_classifier_model.h5",
            f"- Scaler: {self.medical_condition}_classifier_scaler.pkl",
            f"- Label Encoder: {self.medical_condition}_classifier_label_encoder.pkl",
            f"- Results: pipeline_results_{self.medical_condition}.json",
            f"- Dataset: dataset_splits_{self.medical_condition}.npz",
            f"",
            f"Plots Generated:",
            f"- Confusion Matrix: confusion_matrix_{self.medical_condition}.png",
        ])
        
        if 'auc_score' in self.evaluation_results:
            report_lines.append(f"- ROC Curve: roc_curve_{self.medical_condition}.png")
        
        if hasattr(self, 'training_history'):
            report_lines.append(f"- Training History: training_history_{self.medical_condition}.png")
        
        # Save report
        report_file = self.output_dir / f'summary_report_{self.medical_condition}.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to {report_file}")
    
    def run_complete_pipeline(self, 
                            data_directory: str,
                            labels_file: str = None,
                            run_cross_validation: bool = True,
                            **kwargs) -> Dict[str, Any]:
        """
        Run the complete medical gait analysis training pipeline.
        
        Args:
            data_directory: Directory containing patient video data
            labels_file: CSV file with patient labels and metadata
            run_cross_validation: Whether to perform cross-validation
            **kwargs: Additional parameters for training
            
        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete medical gait training pipeline...")
        
        try:
            # 1. Prepare dataset
            dataset = self.prepare_dataset(data_directory, labels_file)
            
            # 2. Train model
            training_results = self.train_model(dataset, **kwargs)
            
            # 3. Cross-validation (optional)
            cv_results = None
            if run_cross_validation:
                cv_results = self.cross_validate_model(dataset)
            
            # 4. Evaluate model
            evaluation_results = self.evaluate_model(dataset)
            
            # 5. Save all results
            self.save_pipeline_results(dataset, training_results, cv_results)
            
            logger.info("Complete pipeline finished successfully!")
            
            return {
                'dataset': dataset,
                'training_results': training_results,
                'cross_validation_results': cv_results,
                'evaluation_results': evaluation_results,
                'output_directory': str(self.output_dir)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
