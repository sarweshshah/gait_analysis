# Real-time Pose Visualization

A real-time pose visualization system that processes video files and displays pose keypoints as colored dots with trail effects, similar to the trail video approach but using MediaPipe for pose estimation.

## Features

### 🎯 Core Functionality

- **Real-time Processing**: Video plays with pose keypoints displayed as colored dots
- **Trail Effect**: Shows keypoint movement history with fading trails
- **No Video Saving**: Displays in window and discards after playback
- **MediaPipe Integration**: Uses the same pose estimation as the gait analysis system

### 🎨 Visualization Options

- **Colored Keypoints**: Each body part has a distinct color for easy identification
- **Confidence-based Sizing**: Larger dots for higher confidence detections
- **Connecting Lines**: Shows skeletal structure between keypoints
- **Trail Effect**: Fading trail showing movement history over time
- **Real-time FPS Counter**: Shows processing performance

### 🎮 Interactive Controls

- **'q'**: Quit visualization
- **'t'**: Toggle trail effect on/off
- **'c'**: Toggle connections between keypoints on/off
- **'r'**: Reset trail history
- **SPACE**: Pause/resume video playback
- **'1', '2', '3'**: Change model complexity (fast/balanced/accurate)

## Usage

### Basic Usage

```bash
# Basic visualization with trail effect
python3 realtime_pose_visualization.py data/sarwesh1.mp4
```

### Advanced Options

```bash
# Show confidence values on keypoints
python3 realtime_pose_visualization.py data/sarwesh1.mp4 --show-confidence

# No trail effect for clean visualization
python3 realtime_pose_visualization.py data/sarwesh1.mp4 --no-trail

# No connections between keypoints
python3 realtime_pose_visualization.py data/sarwesh1.mp4 --no-connections

# Fast model for better performance
python3 realtime_pose_visualization.py data/sarwesh1.mp4 --model-complexity 0

# Accurate model for better precision
python3 realtime_pose_visualization.py data/sarwesh1.mp4 --model-complexity 2

# Custom trail alpha (0.0 to 1.0)
python3 realtime_pose_visualization.py data/sarwesh1.mp4 --trail-alpha 0.5

# Custom minimum confidence threshold
python3 realtime_pose_visualization.py data/sarwesh1.mp4 --min-confidence 0.7
```

### Command Line Arguments

| Argument             | Description                           | Default  | Options                              |
| -------------------- | ------------------------------------- | -------- | ------------------------------------ |
| `video`              | Input video file path                 | Required | Any video file                       |
| `--model-complexity` | MediaPipe model complexity            | 1        | 0 (fast), 1 (balanced), 2 (accurate) |
| `--min-confidence`   | Minimum confidence for pose detection | 0.5      | 0.0 to 1.0                           |
| `--no-trail`         | Disable trail effect                  | False    | Flag                                 |
| `--no-connections`   | Disable connections between keypoints | False    | Flag                                 |
| `--show-confidence`  | Show confidence values on keypoints   | False    | Flag                                 |
| `--trail-alpha`      | Alpha value for trail effect          | 0.3      | 0.0 to 1.0                           |

## Keypoint Colors

The visualization uses different colors for each body part:

| Body Part | Color   | Description              |
| --------- | ------- | ------------------------ |
| Nose      | Yellow  | Face center              |
| Neck      | Magenta | Upper body center        |
| Shoulders | Green   | Left and right shoulders |
| Elbows    | Blue    | Left and right elbows    |
| Wrists    | Red     | Left and right wrists    |
| Hips      | Green   | Left and right hips      |
| Knees     | Blue    | Left and right knees     |
| Ankles    | Red     | Left and right ankles    |
| Eyes      | Cyan    | Left and right eyes      |
| Ears      | Magenta | Left and right ears      |
| Feet      | Yellow  | Toes and heels           |

## Model Complexity Options

### Complexity 0 (Fast)

- **Speed**: Fastest processing
- **Accuracy**: Lower precision
- **Use Case**: Real-time applications, lower-end hardware

### Complexity 1 (Balanced) - Default

- **Speed**: Good balance
- **Accuracy**: Good precision
- **Use Case**: General purpose, most scenarios

### Complexity 2 (Accurate)

- **Speed**: Slower processing
- **Accuracy**: Highest precision
- **Use Case**: Analysis requiring maximum accuracy

## Performance Tips

### For Better Performance

1. **Use model complexity 0** for faster processing
2. **Disable trail effect** with `--no-trail`
3. **Disable connections** with `--no-connections`
4. **Use lower resolution videos** if available

### For Better Accuracy

1. **Use model complexity 2** for highest precision
2. **Increase minimum confidence** with `--min-confidence 0.7`
3. **Enable confidence display** with `--show-confidence`

## Troubleshooting

### Common Issues

**Video doesn't play:**

- Check if video file exists and is readable
- Verify video format is supported by OpenCV
- Ensure virtual environment is activated

**Low FPS:**

- Try model complexity 0: `--model-complexity 0`
- Disable trail effect: `--no-trail`
- Use lower resolution video

**No pose detected:**

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

1. **Same Pose Estimation**: Uses MediaPipe like the main system
2. **Compatible Data**: Outputs same keypoint format
3. **Real-time Testing**: Perfect for testing pose detection on videos
4. **Visual Validation**: Helps verify pose estimation quality

## Examples

### Basic Trail Visualization

```bash
python3 realtime_pose_visualization.py data/sarwesh1.mp4
```

Shows pose keypoints with trail effect, perfect for seeing movement patterns.

### Performance Mode

```bash
python3 realtime_pose_visualization.py data/sarwesh1.mp4 --model-complexity 0 --no-trail
```

Fast processing without trail effect for quick analysis.

### Analysis Mode

```bash
python3 realtime_pose_visualization.py data/sarwesh1.mp4 --model-complexity 2 --show-confidence --no-connections
```

High accuracy with confidence values, no connections for clean keypoint analysis.

## File Structure

```
gait_analysis/
├── realtime_pose_visualization.py  # Main visualization script
├── data/
│   └── sarwesh1.mp4               # Example video file
├── .venv/                         # Virtual environment
└── README_RealTime_Visualization.md  # This file
```

## Contributing

To extend the visualization system:

1. **Add new visualization modes** in the `RealTimePoseVisualizer` class
2. **Modify keypoint colors** in the `colors` list
3. **Add new controls** in the key handling section
4. **Extend command line arguments** in the argument parser

## License

This project is part of the Gait Analysis System. See the main LICENSE file for details.
