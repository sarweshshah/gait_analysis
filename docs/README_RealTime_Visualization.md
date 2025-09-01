# Real-time Pose Visualization

A real-time pose visualization system that processes video files and displays pose keypoints as colored dots with trail effects, similar to the trail video approach but using MediaPipe for pose estimation.

## Overview

This visualization tool provides real-time pose keypoint tracking with customizable display options, making it perfect for gait analysis and pose estimation validation.

### Key Features

- **Real-time Processing**: Processes video frames in real-time
- **Trail Effect**: Shows keypoint movement trails over time
- **Customizable Display**: Multiple visualization options
- **Interactive Controls**: Keyboard controls for real-time adjustments
- **Confidence Display**: Optional confidence value visualization
- **Model Selection**: Supports multiple pose estimation models
- **Performance Modes**: Different complexity levels for speed vs accuracy

### 🎨 Visualization Options

- **Keypoint Display**: Colored dots for each body part
- **Trail Effect**: Fading trails showing movement history
- **Connections**: Lines connecting related keypoints
- **Confidence Values**: Optional confidence score display
- **Model Complexity**: Adjustable processing speed vs accuracy
- **Real-time Controls**: Interactive keyboard controls

## Quick Start

### Basic Usage

```bash
# Basic visualization with trail effect
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4
```

### Advanced Options

```bash
# Show confidence values
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --show-confidence

# No trail effect for clean visualization
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --no-trail

# Hide keypoint connections
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --no-connections

# Fast performance mode
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --model-complexity 0

# High accuracy mode
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --model-complexity 2

# Custom trail transparency
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --trail-alpha 0.5

# Adjust confidence threshold
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --min-confidence 0.7
```

## Interactive Controls

The visualization supports real-time keyboard controls:

- **'q'**: Quit visualization
- **'t'**: Toggle trail effect on/off
- **'c'**: Toggle keypoint connections on/off
- **'r'**: Reset trail (clear all trails)
- **SPACE**: Pause/resume video playback
- **'1', '2', '3'**: Change model complexity (0=fast, 1=balanced, 2=accurate)
- **'+', '-'**: Adjust trail length
- **'s'**: Save current frame as screenshot

## Color Scheme

The visualization uses different colors for each body part:

- **Head**: Blue (nose, eyes, ears)
- **Torso**: Green (shoulders, hips)
- **Arms**: Yellow (elbows, wrists)
- **Legs**: Red (knees, ankles)
- **Feet**: Purple (toes, heels)

This color scheme makes it easy to distinguish different body parts and track their movement patterns.

## Command Line Options

### Basic Options

- `--video`: Path to input video file (required)
- `--output`: Output directory for saved frames (optional)
- `--fps`: Target frame rate for processing (default: 30)

### Display Options

- `--no-trail`: Disable trail effect
- `--no-connections`: Hide keypoint connections
- `--show-confidence`: Display confidence values
- `--trail-length`: Number of frames to keep in trail (default: 30)
- `--trail-alpha`: Trail transparency (0.0-1.0, default: 0.6)

### Performance Options

- `--model-complexity`: Model complexity (0=fast, 1=balanced, 2=accurate)
- `--min-confidence`: Minimum confidence threshold (default: 0.5)
- `--min-detection-confidence`: Detection confidence threshold (default: 0.5)
- `--min-tracking-confidence`: Tracking confidence threshold (default: 0.5)

### Output Options

- `--save-frames`: Save processed frames to output directory
- `--save-video`: Save processed video to output file
- `--frame-interval`: Save every Nth frame (default: 1)

## Examples

### Basic Trail Visualization

```bash
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4
```

Shows pose keypoints with trail effect, perfect for seeing movement patterns.

### Performance Mode

```bash
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --model-complexity 0 --no-trail
```

Fast processing without trail effect for quick analysis.

### Analysis Mode

```bash
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --model-complexity 2 --show-confidence --no-connections
```

High accuracy with confidence values, no connections for clean keypoint analysis.

### Custom Configuration

```bash
python3 usecases/gait_analysis/features/realtime_pose_visualization.py \
    videos/raw/sarwesh1.mp4 \
    --model-complexity 1 \
    --trail-length 45 \
    --trail-alpha 0.4 \
    --min-confidence 0.6 \
    --show-confidence
```

Customized visualization with specific parameters.

## Troubleshooting

### Common Issues

**Video doesn't play:**

- Check video file path is correct
- Verify video format is supported (MP4, AVI, MOV)
- Ensure video file is not corrupted

**Poor pose detection:**

- Check video quality and lighting
- Lower minimum confidence: `--min-confidence 0.3`
- Try model complexity 2: `--model-complexity 2`

**Window doesn't appear:**

- Check if you have a display server running
- Try running on a machine with GUI support
- Verify OpenCV is properly installed

### Error Messages

**"Could not open video file":**

- Check file path is correct
- Verify file exists and is readable
- Check file format is supported

**"No pose detected":**

- Video may not contain clear human poses
- Try adjusting confidence threshold
- Check video quality and lighting

## Integration with Gait Analysis

This visualization system is designed to work with the main gait analysis pipeline:

1. **Unified Pose System**: Uses the same unified pose processor as the main system
2. **Model Compatibility**: Supports multiple pose estimation models
3. **Compatible Data**: Outputs same keypoint format
4. **Real-time Testing**: Perfect for testing pose detection on videos
5. **Visual Validation**: Helps verify pose estimation quality
6. **Model Comparison**: Can visualize differences between pose models

## Examples

### Basic Trail Visualization

```bash
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4
```

Shows pose keypoints with trail effect, perfect for seeing movement patterns.

### Performance Mode

```bash
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --model-complexity 0 --no-trail
```

Fast processing without trail effect for quick analysis.

### Analysis Mode

```bash
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --model-complexity 2 --show-confidence --no-connections
```

High accuracy with confidence values, no connections for clean keypoint analysis.

## File Structure

```
gait_analysis/
├── usecases/
│   └── gait_analysis/
│       └── features/
│           └── realtime_pose_visualization.py  # Main visualization script
├── videos/
│   ├── raw/
│   │   └── sarwesh1.mp4                        # Example video file
│   └── sneak/                                  # Sneak gait videos
├── docs/
│   └── README_RealTime_Visualization.md        # This file
└── .venv/                                      # Virtual environment
```

## Related Documentation

For more information about the project and its evolution:

- **Project Changelog**: [docs/README_Changelog.md](README_Changelog.md) - Complete project history and changes
- **Installation Guide**: [docs/README_Installation.md](README_Installation.md) - Comprehensive installation instruction
- **TCN System Documentation**: [docs/README_TCN_Gait_Analysis.md](README_TCN_Gait_Analysis.md) - Technical system documentation
- **Core Modules**: [core/README_CoreModules.md](../core/README_CoreModules.md) - Core system modules documentation

## Contributing

To extend the visualization system:

1. **Add new visualization modes** in the `RealTimePoseVisualizer` class
2. **Modify keypoint colors** in the `colors` list
3. **Add new controls** in the key handling section
4. **Extend command line arguments** in the argument parser

## License

This project is part of the Gait Analysis System. See the main LICENSE file for details.
