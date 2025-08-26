#!/usr/bin/env python3
"""
Real-time Pose Visualization
===========================

This script processes a video file and displays pose keypoints as dots
in real-time, similar to the trail video approach but using MediaPipe.
"""

import os
# Suppress TensorFlow Lite feedback manager warnings - must be set before importing mediapipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
            model_complexity: MediaPipe model complexity (1=fast, 2=balanced, 3=accurate)
                             Will be mapped to MediaPipe's 0-2 range internally
            min_detection_confidence: Minimum confidence for pose detection
        """
        # Map 1-3 input to 0-2 for MediaPipe
        self.model_complexity = max(0, min(2, model_complexity - 1))
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
        
        # Define colors for MediaPipe's 33 landmarks (BGR format)
        self.colors = [
            (0, 255, 255),    # 0: nose - Yellow
            (255, 0, 255),    # 1: left_eye_inner - Magenta
            (255, 255, 0),    # 2: left_eye - Cyan
            (255, 0, 255),    # 3: left_eye_outer - Magenta
            (255, 0, 255),    # 4: right_eye_inner - Magenta
            (255, 255, 0),    # 5: right_eye - Cyan
            (255, 0, 255),    # 6: right_eye_outer - Magenta
            (255, 0, 255),    # 7: left_ear - Magenta
            (255, 0, 255),    # 8: right_ear - Magenta
            (128, 255, 128),  # 9: mouth_left - Light Green
            (128, 255, 128),  # 10: mouth_right - Light Green
            (0, 255, 0),      # 11: left_shoulder - Green
            (0, 255, 0),      # 12: right_shoulder - Green
            (255, 0, 0),      # 13: left_elbow - Blue
            (255, 0, 0),      # 14: right_elbow - Blue
            (0, 0, 255),      # 15: left_wrist - Red
            (0, 0, 255),      # 16: right_wrist - Red
            (128, 0, 128),    # 17: left_pinky - Purple
            (128, 0, 128),    # 18: right_pinky - Purple
            (255, 128, 0),    # 19: left_index - Orange
            (255, 128, 0),    # 20: right_index - Orange
            (0, 128, 255),    # 21: left_thumb - Light Blue
            (0, 128, 255),    # 22: right_thumb - Light Blue
            (0, 255, 0),      # 23: left_hip - Green
            (0, 255, 0),      # 24: right_hip - Green
            (255, 0, 0),      # 25: left_knee - Blue
            (255, 0, 0),      # 26: right_knee - Blue
            (0, 0, 255),      # 27: left_ankle - Red
            (0, 0, 255),      # 28: right_ankle - Red
            (0, 255, 255),    # 29: left_heel - Yellow
            (0, 255, 255),    # 30: right_heel - Yellow
            (255, 255, 0),    # 31: left_foot_index - Cyan
            (255, 255, 0),    # 32: right_foot_index - Cyan
        ]
        
        # Store keypoint history for trail effect
        self.keypoint_history = []
        self.max_history = 30  # Number of frames to keep in history
        
        # Occlusion handling
        self.previous_keypoints = None
        self.occlusion_threshold = 0.3
        self.interpolation_frames = 5  # Max frames to interpolate missing keypoints
        
        # Body connections for drawing lines (MediaPipe landmark indices)
        self.connections = [
            # Face connections
            (0, 1),   # nose to left_eye_inner
            (0, 4),   # nose to right_eye_inner
            (1, 2),   # left_eye_inner to left_eye
            (2, 3),   # left_eye to left_eye_outer
            (4, 5),   # right_eye_inner to right_eye
            (5, 6),   # right_eye to right_eye_outer
            (3, 7),   # left_eye_outer to left_ear
            (6, 8),   # right_eye_outer to right_ear
            (9, 10),  # mouth_left to mouth_right
            
            # Upper body connections
            (11, 12), # left_shoulder to right_shoulder
            (11, 13), # left_shoulder to left_elbow
            (13, 15), # left_elbow to left_wrist
            (12, 14), # right_shoulder to right_elbow
            (14, 16), # right_elbow to right_wrist
            
            # Hand connections
            (15, 17), # left_wrist to left_pinky
            (15, 19), # left_wrist to left_index
            (15, 21), # left_wrist to left_thumb
            (16, 18), # right_wrist to right_pinky
            (16, 20), # right_wrist to right_index
            (16, 22), # right_wrist to right_thumb
            
            # Torso connections
            (11, 23), # left_shoulder to left_hip
            (12, 24), # right_shoulder to right_hip
            (23, 24), # left_hip to right_hip
            
            # Lower body connections
            (23, 25), # left_hip to left_knee
            (25, 27), # left_knee to left_ankle
            (24, 26), # right_hip to right_knee
            (26, 28), # right_knee to right_ankle
            
            # Foot connections
            (27, 29), # left_ankle to left_heel
            (27, 31), # left_ankle to left_foot_index
            (28, 30), # right_ankle to right_heel
            (28, 32), # right_ankle to right_foot_index
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
                    # Handle self-occlusion
                    keypoints = self._handle_occlusion(keypoints)
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
                    
                    # Draw connections first (so they appear behind keypoints)
                    if connections_enabled:
                        self._draw_connections(display_frame, keypoints)
                    
                    # Draw current keypoints on top
                    self._draw_keypoints(display_frame, keypoints, show_confidence)
                    
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
        """Change MediaPipe model complexity on the fly.
        
        Args:
            complexity: Input complexity level (1=fast, 2=balanced, 3=accurate)
        """
        # Map 1-3 input to 0-2 for MediaPipe
        mapped_complexity = max(0, min(2, complexity - 1))
        self.pose.close()
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=mapped_complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5
        )
        # Show the original input complexity level to user
        print(f"Model complexity changed to: {complexity} (mapped to MediaPipe level {mapped_complexity})")
    
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
    
    def _handle_occlusion(self, keypoints: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """
        Handle self-occlusion using temporal smoothing and anatomical constraints.
        
        Args:
            keypoints: Current frame keypoints
            
        Returns:
            Enhanced keypoints with occlusion handling
        """
        if self.previous_keypoints is None:
            self.previous_keypoints = keypoints.copy()
            return keypoints
        
        enhanced_keypoints = []
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf < self.occlusion_threshold:
                # Low confidence - likely occluded
                prev_x, prev_y, prev_conf = self.previous_keypoints[i]
                
                # Use temporal smoothing if previous keypoint was confident
                if prev_conf > 0.5:
                    # Interpolate position with reduced confidence
                    enhanced_keypoints.append((prev_x, prev_y, min(prev_conf * 0.7, 0.6)))
                else:
                    # Try anatomical estimation
                    estimated = self._estimate_from_anatomy(i, keypoints)
                    if estimated:
                        enhanced_keypoints.append(estimated)
                    else:
                        enhanced_keypoints.append((x, y, conf))
            else:
                enhanced_keypoints.append((x, y, conf))
        
        # Update previous keypoints for next frame
        self.previous_keypoints = enhanced_keypoints.copy()
        return enhanced_keypoints
    
    def _estimate_from_anatomy(self, joint_idx: int, keypoints: List[Tuple[int, int, float]]) -> Optional[Tuple[int, int, float]]:
        """
        Estimate occluded joint position using anatomical constraints.
        
        Args:
            joint_idx: Index of the occluded joint
            keypoints: Current frame keypoints
            
        Returns:
            Estimated (x, y, confidence) or None if estimation not possible
        """
        # Define anatomical relationships for key joints
        estimations = {
            # Shoulders - estimate from hips and head
            11: [(0, 23), 0.3],  # left_shoulder from nose and left_hip
            12: [(0, 24), 0.3],  # right_shoulder from nose and right_hip
            
            # Elbows - estimate from shoulders and wrists
            13: [(11, 15), 0.5], # left_elbow from left_shoulder and left_wrist
            14: [(12, 16), 0.5], # right_elbow from right_shoulder and right_wrist
            
            # Knees - estimate from hips and ankles
            25: [(23, 27), 0.5], # left_knee from left_hip and left_ankle
            26: [(24, 28), 0.5], # right_knee from right_hip and right_ankle
        }
        
        if joint_idx not in estimations:
            return None
        
        (ref1_idx, ref2_idx), ratio = estimations[joint_idx]
        
        # Check if reference points are confident
        if (ref1_idx < len(keypoints) and ref2_idx < len(keypoints) and
            keypoints[ref1_idx][2] > 0.5 and keypoints[ref2_idx][2] > 0.5):
            
            x1, y1, _ = keypoints[ref1_idx]
            x2, y2, _ = keypoints[ref2_idx]
            
            # Interpolate position
            est_x = int(x1 + ratio * (x2 - x1))
            est_y = int(y1 + ratio * (y2 - y1))
            
            return (est_x, est_y, 0.4)  # Lower confidence for estimated points
        
        return None
    
    def _draw_keypoints(self, frame: np.ndarray, keypoints: List[Tuple[int, int, float]], 
                       show_confidence: bool = False):
        """Draw keypoints as dots on the frame."""
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:  # Draw keypoints with some confidence (lowered for estimated points)
                color = self.colors[i] if i < len(self.colors) else (255, 255, 255)
                
                # Adjust circle size based on confidence
                radius = int(1 + conf * 3)
                
                # Different visualization for estimated vs detected keypoints
                if conf <= 0.5:  # Likely estimated/interpolated
                    # Draw dashed circle for estimated points
                    cv2.circle(frame, (x, y), radius, color, 1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
                    # Add small cross to indicate estimation
                    cv2.line(frame, (x-3, y-3), (x+3, y+3), (255, 255, 255), 1)
                    cv2.line(frame, (x-3, y+3), (x+3, y-3), (255, 255, 255), 1)
                else:
                    # Solid circle for detected points
                    cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
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
        
        # Create stats panel - calculate height based on content
        panel_width = 280
        line_height = 18  # Reduced for 12px text
        num_lines = 3  # Frame, Time, FPS
        num_status_lines = 2  # Trail and Connections (Confidence only shows when enabled)
        if show_confidence:
            num_status_lines = 3
        panel_height = 20 + (num_lines * line_height) + (num_status_lines * line_height)
        panel_x = width - panel_width - 15
        panel_y = 15
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (60, 60, 60), 1)
        
        # --- Text Styling ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Adjusted for 14px text
        text_color = (255, 255, 255)
        line_height = 18  # Use the same line_height as defined above
        status_dot_radius = 4
        status_on_color = (0, 255, 0)  # Green
        status_off_color = (0, 0, 255) # Red

        # --- Draw Stats ---
        y_pos = panel_y + 25

        # Progress and timing
        progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
        duration_sec = (total_frames / fps) if fps > 0 else 0
        current_time_sec = (frame_count / fps) if fps > 0 else 0

        # Non-status stats
        stats_text = [
            f"Frame: {frame_count} / {total_frames} ({progress:.1f}%)",
            f"Time: {current_time_sec:.1f}s / {duration_sec:.1f}s",
            f"FPS: {self.current_fps:.1f}"
        ]
        for line in stats_text:
            cv2.putText(frame, line, (panel_x + 10, y_pos), font, font_scale, text_color, 1, cv2.LINE_AA)
            y_pos += line_height

        # Status indicators with colored dots
        statuses = {
            "Trail": trail_enabled,
            "Connections": connections_enabled,
            "Confidence": show_confidence
        }
        for label, is_enabled in statuses.items():
            if label == "Confidence" and not is_enabled:
                continue # Don't show confidence if it's off

            color = status_on_color if is_enabled else status_off_color
            dot_x = panel_x + 15
            text_x = panel_x + 25

            cv2.circle(frame, (dot_x, y_pos - 5), status_dot_radius, color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (text_x, y_pos), font, font_scale, text_color, 1, cv2.LINE_AA)
            y_pos += line_height
        
        # Minimal controls at bottom
        controls_text = "q:quit | SPACE:pause | t:trail | c:connections"
        text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(frame, controls_text, 
                   (width - text_size[0] - 10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

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
