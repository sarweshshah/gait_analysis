#!/usr/bin/env python3
"""
Real-time Pose Visualization
===========================

This script processes a video file and displays pose keypoints as dots
in real-time, similar to the trail video approach but using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import List, Tuple, Optional
import argparse

class RealTimePoseVisualizer:
    """Real-time pose visualization with MediaPipe."""
    
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        """
        Initialize the real-time pose visualizer.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_detection_confidence: Minimum confidence for pose detection
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Define colors for different body parts (BGR format)
        self.colors = [
            (0, 255, 255),    # Nose - Yellow
            (255, 0, 255),    # Neck - Magenta
            (0, 255, 0),      # RShoulder - Green
            (255, 0, 0),      # RElbow - Blue
            (0, 0, 255),      # RWrist - Red
            (0, 255, 0),      # LShoulder - Green
            (255, 0, 0),      # LElbow - Blue
            (0, 0, 255),      # LWrist - Red
            (128, 128, 128),  # MidHip - Gray
            (0, 255, 0),      # RHip - Green
            (255, 0, 0),      # RKnee - Blue
            (0, 0, 255),      # RAnkle - Red
            (0, 255, 0),      # LHip - Green
            (255, 0, 0),      # LKnee - Blue
            (0, 0, 255),      # LAnkle - Red
            (255, 255, 0),    # REye - Cyan
            (255, 255, 0),    # LEye - Cyan
            (255, 0, 255),    # REar - Magenta
            (255, 0, 255),    # LEar - Magenta
            (0, 255, 255),    # LBigToe - Yellow
            (0, 255, 255),    # LSmallToe - Yellow
            (0, 255, 255),    # LHeel - Yellow
            (0, 255, 255),    # RBigToe - Yellow
            (0, 255, 255),    # RSmallToe - Yellow
            (0, 255, 255),    # RHeel - Yellow
        ]
        
        # Store keypoint history for trail effect
        self.keypoint_history = []
        self.max_history = 30  # Number of frames to keep in history
        
        # Body connections for drawing lines
        self.connections = [
            (1, 2),   # Neck to RShoulder
            (1, 5),   # Neck to LShoulder
            (2, 3),   # RShoulder to RElbow
            (3, 4),   # RElbow to RWrist
            (5, 6),   # LShoulder to LElbow
            (6, 7),   # LElbow to LWrist
            (1, 8),   # Neck to MidHip
            (8, 9),   # MidHip to RHip
            (9, 10),  # RHip to RKnee
            (10, 11), # RKnee to RAnkle
            (8, 12),  # MidHip to LHip
            (12, 13), # LHip to LKnee
            (13, 14), # LKnee to LAnkle
            (0, 1),   # Nose to Neck
            (0, 15),  # Nose to REye
            (0, 16),  # Nose to LEye
            (15, 17), # REye to REar
            (16, 18), # LEye to LEar
            (11, 19), # RAnkle to RBigToe
            (11, 20), # RAnkle to RSmallToe
            (11, 21), # RAnkle to RHeel
            (14, 22), # LAnkle to LBigToe
            (14, 23), # LAnkle to LSmallToe
            (14, 24), # LAnkle to LHeel
        ]
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
    
    def process_video(self, video_path: str, show_trail: bool = True, trail_alpha: float = 0.3, 
                     show_connections: bool = True, show_confidence: bool = False, loop_video: bool = False):
        """
        Process video and display pose keypoints in real-time.
        
        Args:
            video_path: Path to input video file
            show_trail: Whether to show keypoint trail effect
            trail_alpha: Alpha value for trail effect (0.0 to 1.0)
            show_connections: Whether to show connections between keypoints
            show_confidence: Whether to show confidence values
            loop_video: Whether to loop the video playback
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps:.2f}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.2f} seconds")
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 't' to toggle trail effect")
        print("- Press 'c' to toggle connections")
        print("- Press 'r' to reset trail")
        print("- Press SPACE to pause/resume")
        print("- Press '1', '2', '3' to change model complexity")
        
        frame_count = 0
        paused = False
        trail_enabled = show_trail
        connections_enabled = show_connections
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if loop_video:
                        print("\nEnd of video reached - restarting...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                        frame_count = 0
                        self.keypoint_history.clear()  # Clear trail history for clean restart
                        continue
                    else:
                        print("\nEnd of video reached")
                        break
                
                frame_count += 1
                
                # Process frame with MediaPipe
                keypoints = self._process_frame(frame)
                
                if keypoints is not None:
                    # Add to history for trail effect
                    if trail_enabled:
                        self.keypoint_history.append(keypoints.copy())
                        if len(self.keypoint_history) > self.max_history:
                            self.keypoint_history.pop(0)
                    
                    # Draw current frame
                    display_frame = frame.copy()
                    
                    # Draw trail effect
                    if trail_enabled and len(self.keypoint_history) > 1:
                        self._draw_trail(display_frame, trail_alpha)
                    
                    # Draw current keypoints
                    self._draw_keypoints(display_frame, keypoints, show_confidence)
                    
                    # Draw connections
                    if connections_enabled:
                        self._draw_connections(display_frame, keypoints)
                    
                    # Add frame info
                    self._draw_info(display_frame, frame_count, total_frames, fps, 
                                  trail_enabled, connections_enabled, show_confidence)
                    
                    cv2.imshow('Real-time Pose Visualization', display_frame)
                else:
                    # No pose detected
                    display_frame = frame.copy()
                    self._draw_info(display_frame, frame_count, total_frames, fps, 
                                  trail_enabled, connections_enabled, show_confidence)
                    cv2.putText(display_frame, "No pose detected", (10, frame_height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Real-time Pose Visualization', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                trail_enabled = not trail_enabled
                print(f"Trail effect: {'ON' if trail_enabled else 'OFF'}")
            elif key == ord('c'):
                connections_enabled = not connections_enabled
                print(f"Connections: {'ON' if connections_enabled else 'OFF'}")
            elif key == ord('r'):
                self.keypoint_history.clear()
                print("Trail reset")
            elif key == ord(' '):
                paused = not paused
                print(f"Video: {'PAUSED' if paused else 'RESUMED'}")
            elif key in [ord('1'), ord('2'), ord('3')]:
                new_complexity = key - ord('0')
                self._change_model_complexity(new_complexity)
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
    
    def _change_model_complexity(self, complexity: int):
        """Change MediaPipe model complexity on the fly."""
        self.pose.close()
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5
        )
        print(f"Model complexity changed to: {complexity}")
    
    def _process_frame(self, frame: np.ndarray) -> Optional[List[Tuple[int, int, float]]]:
        """
        Process a single frame and extract pose keypoints.
        
        Args:
            frame: Input frame
            
        Returns:
            List of (x, y, confidence) tuples for each keypoint, or None if no pose detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract keypoints
        keypoints = []
        landmarks = results.pose_landmarks.landmark
        
        for landmark in landmarks:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            confidence = landmark.visibility
            
            keypoints.append((x, y, confidence))
        
        return keypoints
    
    def _draw_keypoints(self, frame: np.ndarray, keypoints: List[Tuple[int, int, float]], 
                       show_confidence: bool = False):
        """Draw keypoints as dots on the frame."""
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:  # Only draw confident keypoints
                color = self.colors[i] if i < len(self.colors) else (255, 255, 255)
                
                # Adjust circle size based on confidence
                radius = int(3 + conf * 3)
                cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
                # Draw a white border
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Show confidence value if requested
                if show_confidence:
                    cv2.putText(frame, f"{conf:.2f}", (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def _draw_connections(self, frame: np.ndarray, keypoints: List[Tuple[int, int, float]]):
        """Draw connections between keypoints."""
        for connection in self.connections:
            if len(connection) == 2:
                idx1, idx2 = connection
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and 
                    keypoints[idx1][2] > 0.5 and keypoints[idx2][2] > 0.5):
                    
                    pt1 = (keypoints[idx1][0], keypoints[idx1][1])
                    pt2 = (keypoints[idx2][0], keypoints[idx2][1])
                    
                    # Draw line with thickness based on average confidence
                    avg_conf = (keypoints[idx1][2] + keypoints[idx2][2]) / 2
                    thickness = max(1, int(avg_conf * 3))
                    cv2.line(frame, pt1, pt2, (128, 128, 128), thickness, cv2.LINE_AA)
    
    def _draw_trail(self, frame: np.ndarray, alpha: float):
        """Draw keypoint trail effect."""
        if len(self.keypoint_history) < 2:
            return
        
        # Create a copy of the frame for trail
        trail_frame = frame.copy()
        
        # Draw historical keypoints with fading effect
        for i, historical_keypoints in enumerate(self.keypoint_history[:-1]):
            # Calculate alpha based on age (older = more transparent)
            trail_alpha = alpha * (i / len(self.keypoint_history))
            
            for j, (x, y, conf) in enumerate(historical_keypoints):
                if conf > 0.5:
                    color = self.colors[j] if j < len(self.colors) else (255, 255, 255)
                    # Draw smaller, more transparent dots for trail
                    radius = max(1, int(2 * (i / len(self.keypoint_history))))
                    cv2.circle(trail_frame, (x, y), radius, color, -1, cv2.LINE_AA)
        
        # Blend trail with current frame
        cv2.addWeighted(trail_frame, alpha, frame, 1 - alpha, 0, frame)
    
    def _draw_info(self, frame: np.ndarray, frame_count: int, total_frames: int, fps: float, 
                  trail_enabled: bool, connections_enabled: bool, show_confidence: bool):
        """Draw information overlay on frame."""
        height, width = frame.shape[:2]
        
        # Calculate current FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
        
        # Frame counter
        progress = frame_count / total_frames * 100
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status indicators
        y_offset = 90
        cv2.putText(frame, f"Trail: {'ON' if trail_enabled else 'OFF'}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if trail_enabled else (0, 0, 255), 2)
        
        cv2.putText(frame, f"Connections: {'ON' if connections_enabled else 'OFF'}", 
                   (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if connections_enabled else (0, 0, 255), 2)
        
        if show_confidence:
            cv2.putText(frame, "Confidence: ON", 
                       (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Controls reminder
        cv2.putText(frame, "q:quit t:trail c:connections r:reset SPACE:pause 1/2/3:complexity", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time Pose Visualization')
    parser.add_argument('video', help='Input video file path')
    parser.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2],
                       help='MediaPipe model complexity (0=fast, 1=balanced, 2=accurate)')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                       help='Minimum confidence for pose detection')
    parser.add_argument('--no-trail', action='store_true',
                       help='Disable trail effect')
    parser.add_argument('--no-connections', action='store_true',
                       help='Disable connections between keypoints')
    parser.add_argument('--show-confidence', action='store_true',
                       help='Show confidence values on keypoints')
    parser.add_argument('--trail-alpha', type=float, default=0.3,
                       help='Alpha value for trail effect (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = RealTimePoseVisualizer(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_confidence
    )
    
    # Process video
    visualizer.process_video(
        video_path=args.video,
        show_trail=not args.no_trail,
        show_connections=not args.no_connections,
        show_confidence=args.show_confidence,
        trail_alpha=args.trail_alpha
    )

if __name__ == "__main__":
    main()
