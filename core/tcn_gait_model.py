"""
Temporal Convolutional Network (TCN) for Gait Analysis
=====================================================

This module implements a TCN architecture specifically designed for gait event
and phase detection using OpenPose-extracted 2D human pose data.

Author: Gait Analysis System
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class TemporalBlock(layers.Layer):
    """
    Temporal Block for TCN architecture.
    
    Each temporal block consists of:
    - Two dilated causal convolutions
    - Batch normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """
    
    def __init__(self, 
                 n_outputs: int,
                 kernel_size: int,
                 stride: int,
                 dilation_rate: int,
                 padding: str = 'causal',
                 dropout_rate: float = 0.2,
                 name: str = None):
        """
        Initialize Temporal Block.
        
        Args:
            n_outputs: Number of output filters
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            dilation_rate: Dilation rate for the convolution
            padding: Type of padding ('causal' for TCN)
            dropout_rate: Dropout rate
            name: Layer name
        """
        super(TemporalBlock, self).__init__(name=name)
        
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.dropout_rate = dropout_rate
        
        # First convolution block
        self.conv1 = layers.Conv1D(
            filters=n_outputs,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=None,
            name=f'{name}_conv1' if name else 'conv1'
        )
        self.bn1 = layers.BatchNormalization(name=f'{name}_bn1' if name else 'bn1')
        self.dropout1 = layers.Dropout(dropout_rate, name=f'{name}_dropout1' if name else 'dropout1')
        
        # Second convolution block
        self.conv2 = layers.Conv1D(
            filters=n_outputs,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=None,
            name=f'{name}_conv2' if name else 'conv2'
        )
        self.bn2 = layers.BatchNormalization(name=f'{name}_bn2' if name else 'bn2')
        self.dropout2 = layers.Dropout(dropout_rate, name=f'{name}_dropout2' if name else 'dropout2')
        
        # Residual connection
        self.downsample = None
        
    def build(self, input_shape):
        """Build the layer with proper input shape."""
        # Create downsample layer if input and output dimensions don't match
        if input_shape[-1] != self.n_outputs:
            self.downsample = layers.Conv1D(
                filters=self.n_outputs,
                kernel_size=1,
                strides=self.stride,
                padding='same',
                name=f'{self.name}_downsample' if self.name else 'downsample'
            )
    
    def call(self, inputs, training=None):
        """Forward pass through the temporal block."""
        # First convolution block
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        out = self.dropout1(out, training=training)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.dropout2(out, training=training)
        
        # Residual connection
        if self.downsample is not None:
            inputs = self.downsample(inputs)
        
        return tf.nn.relu(out + inputs)

class GaitTCN(keras.Model):
    """
    Temporal Convolutional Network for Gait Analysis.
    
    This TCN is specifically designed for gait event and phase detection
    using OpenPose-extracted 2D human pose data.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 num_classes: int = 2,
                 num_filters: int = 64,
                 kernel_size: int = 3,
                 num_blocks: int = 4,
                 dropout_rate: float = 0.2,
                 dilation_base: int = 2,
                 name: str = 'GaitTCN'):
        """
        Initialize Gait TCN model.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            num_classes: Number of output classes (gait phases/events)
            num_filters: Number of filters in convolutional layers
            kernel_size: Size of convolutional kernels
            num_blocks: Number of temporal blocks
            dropout_rate: Dropout rate for regularization
            dilation_base: Base for exponential dilation growth
            name: Model name
        """
        super(GaitTCN, self).__init__(name=name)
        
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.dilation_base = dilation_base
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field()
        logger.info(f"TCN receptive field: {self.receptive_field} time steps")
        
        # Build model layers
        self._build_layers()
    
    def _calculate_receptive_field(self) -> int:
        """Calculate the receptive field of the TCN."""
        receptive_field = 1
        for i in range(self.num_blocks):
            dilation = self.dilation_base ** i
            receptive_field += (self.kernel_size - 1) * dilation
        return receptive_field
    
    def _build_layers(self):
        """Build the TCN layers."""
        # Input layer
        self.input_layer = layers.Input(shape=self.input_shape_)
        
        # Initial convolution to project to desired number of filters
        self.initial_conv = layers.Conv1D(
            filters=self.num_filters,
            kernel_size=1,
            padding='same',
            name='initial_conv'
        )
        
        # Temporal blocks with exponential dilation
        self.temporal_blocks = []
        for i in range(self.num_blocks):
            dilation_rate = self.dilation_base ** i
            block = TemporalBlock(
                n_outputs=self.num_filters,
                kernel_size=self.kernel_size,
                stride=1,
                dilation_rate=dilation_rate,
                dropout_rate=self.dropout_rate,
                name=f'temporal_block_{i}'
            )
            self.temporal_blocks.append(block)
        
        # Global average pooling over time dimension
        self.global_pool = layers.GlobalAveragePooling1D(name='global_pool')
        
        # Classification head
        self.classifier = keras.Sequential([
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.Dropout(self.dropout_rate, name='classifier_dropout'),
            layers.Dense(64, activation='relu', name='dense_2'),
            layers.Dropout(self.dropout_rate * 0.5, name='classifier_dropout_2'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='classifier')
    
    def call(self, inputs, training=None):
        """Forward pass through the TCN."""
        # Initial convolution
        x = self.initial_conv(inputs)
        
        # Temporal blocks
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification
        outputs = self.classifier(x, training=training)
        
        return outputs
    
    def get_config(self):
        """Get model configuration."""
        config = super(GaitTCN, self).get_config()
        config.update({
            'input_shape': self.input_shape_,
            'num_classes': self.num_classes,
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'num_blocks': self.num_blocks,
            'dropout_rate': self.dropout_rate,
            'dilation_base': self.dilation_base
        })
        return config

class GaitEventDetector(keras.Model):
    """
    Specialized TCN for gait event detection (heel strike, toe-off).
    
    This model is designed to detect specific gait events rather than
    continuous phases.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 num_events: int = 2,  # heel_strike, toe_off
                 num_filters: int = 128,
                 kernel_size: int = 5,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.3,
                 name: str = 'GaitEventDetector'):
        """
        Initialize Gait Event Detector.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            num_events: Number of gait events to detect
            num_filters: Number of filters in convolutional layers
            kernel_size: Size of convolutional kernels
            num_blocks: Number of temporal blocks
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        super(GaitEventDetector, self).__init__(name=name)
        
        self.input_shape_ = input_shape
        self.num_events = num_events
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        
        # Build model layers
        self._build_layers()
    
    def _build_layers(self):
        """Build the event detection layers."""
        # Input layer
        self.input_layer = layers.Input(shape=self.input_shape_)
        
        # Initial convolution
        self.initial_conv = layers.Conv1D(
            filters=self.num_filters,
            kernel_size=1,
            padding='same',
            name='initial_conv'
        )
        
        # Temporal blocks with larger receptive field for event detection
        self.temporal_blocks = []
        for i in range(self.num_blocks):
            dilation_rate = 2 ** i
            block = TemporalBlock(
                n_outputs=self.num_filters,
                kernel_size=self.kernel_size,
                stride=1,
                dilation_rate=dilation_rate,
                dropout_rate=self.dropout_rate,
                name=f'event_block_{i}'
            )
            self.temporal_blocks.append(block)
        
        # Event detection head (per-frame classification)
        self.event_head = keras.Sequential([
            layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate * 0.5),
            layers.Conv1D(self.num_events, kernel_size=1, activation='sigmoid')
        ], name='event_head')
    
    def call(self, inputs, training=None):
        """Forward pass through the event detector."""
        # Initial convolution
        x = self.initial_conv(inputs)
        
        # Temporal blocks
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Event detection (per-frame)
        outputs = self.event_head(x, training=training)
        
        return outputs

def create_gait_tcn_model(input_shape: Tuple[int, int],
                         task_type: str = 'phase_detection',
                         **kwargs) -> keras.Model:
    """
    Factory function to create appropriate TCN model for gait analysis.
    
    Args:
        input_shape: Shape of input data (sequence_length, n_features)
        task_type: Type of task ('phase_detection' or 'event_detection')
        **kwargs: Additional arguments for model configuration
        
    Returns:
        Configured TCN model
    """
    if task_type == 'phase_detection':
        # Default configuration for gait phase detection
        default_config = {
            'num_classes': 4,  # stance, swing, double_support, etc.
            'num_filters': 64,
            'kernel_size': 3,
            'num_blocks': 4,
            'dropout_rate': 0.2,
            'dilation_base': 2
        }
        default_config.update(kwargs)
        
        return GaitTCN(input_shape=input_shape, **default_config)
    
    elif task_type == 'event_detection':
        # Default configuration for gait event detection
        default_config = {
            'num_events': 2,  # heel_strike, toe_off
            'num_filters': 128,
            'kernel_size': 5,
            'num_blocks': 6,
            'dropout_rate': 0.3
        }
        default_config.update(kwargs)
        
        return GaitEventDetector(input_shape=input_shape, **default_config)
    
    else:
        raise ValueError(f"Unknown task type: {task_type}. Use 'phase_detection' or 'event_detection'")

def compile_gait_model(model: keras.Model,
                      task_type: str = 'phase_detection',
                      learning_rate: float = 0.001) -> keras.Model:
    """
    Compile the gait TCN model with appropriate loss and metrics.
    
    Args:
        model: TCN model to compile
        task_type: Type of task ('phase_detection' or 'event_detection')
        learning_rate: Learning rate for optimization
        
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    if task_type == 'phase_detection':
        # Multi-class classification for gait phases
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy', 'sparse_categorical_accuracy']
    elif task_type == 'event_detection':
        # Binary classification for gait events
        loss = 'binary_crossentropy'
        metrics = ['accuracy', 'binary_accuracy', 'precision', 'recall']
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

# Example usage and testing
if __name__ == "__main__":
    # Example: Create and test a gait phase detection model
    input_shape = (30, 50)  # 30 frames, 50 features
    
    # Phase detection model
    phase_model = create_gait_tcn_model(
        input_shape=input_shape,
        task_type='phase_detection',
        num_classes=4
    )
    phase_model = compile_gait_model(phase_model, task_type='phase_detection')
    
    print("Phase Detection Model Summary:")
    phase_model.summary()
    
    # Event detection model
    event_model = create_gait_tcn_model(
        input_shape=input_shape,
        task_type='event_detection',
        num_events=2
    )
    event_model = compile_gait_model(event_model, task_type='event_detection')
    
    print("\nEvent Detection Model Summary:")
    event_model.summary()
