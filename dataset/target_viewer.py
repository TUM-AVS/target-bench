#!/usr/bin/env python3
"""Combined Target dataset visualizer and interactive viewer.

Default usage launches the interactive Tk viewer. Use ``--summary`` to print a
lightweight dataset summary.
"""

import argparse
import base64
import io
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


class TargetVisualizer:
    def __init__(self, dataset_path: str = "Target_trajectories"):
        """
        Initialize the target dataset visualizer.
        
        Args:
            dataset_path: Path to the Target_trajectories folder
        """
        self.dataset_path = Path(dataset_path)
        self.data_path = self.dataset_path / "data"
        self.video_path = self.dataset_path / "video"
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)
    
    @staticmethod
    def quaternion_to_heading(qx, qy, qz, qw):
        """
        Convert quaternion to heading angle in radians.
        
        Args:
            qx, qy, qz, qw: Quaternion components
            
        Returns:
            Heading angle in radians (0 to 2π)
        """
        # Convert quaternion to yaw angle (rotation around z-axis)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 
                        1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Normalize to [0, 2π]
        if yaw < 0:
            yaw += 2 * math.pi
            
        return yaw
    
    @staticmethod
    def get_heading_vector(qx, qy, qz, qw, length=1.0):
        """
        Get heading direction vector from quaternion.
        
        Args:
            qx, qy, qz, qw: Quaternion components
            length: Length of the direction vector
            
        Returns:
            Tuple of (dx, dy) representing the direction vector
        """
        heading = TargetVisualizer.quaternion_to_heading(qx, qy, qz, qw)
        dx = length * math.cos(heading)
        dy = length * math.sin(heading)
        return dx, dy
    
    @staticmethod
    def rotate_points(x_coords, y_coords, angle_rad):
        """
        Rotate a set of 2D points by a given angle.
        
        Args:
            x_coords: Array of x coordinates
            y_coords: Array of y coordinates  
            angle_rad: Rotation angle in radians
            
        Returns:
            Tuple of (rotated_x, rotated_y) arrays
        """
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        rotated_x = x_coords * cos_angle - y_coords * sin_angle
        rotated_y = x_coords * sin_angle + y_coords * cos_angle
        
        return rotated_x, rotated_y
    
    @staticmethod
    def rotate_quaternion_heading(qx, qy, qz, qw, rotation_angle):
        """
        Rotate a quaternion's heading by a given angle around the z-axis.
        
        Args:
            qx, qy, qz, qw: Original quaternion components
            rotation_angle: Additional rotation angle in radians
            
        Returns:
            Tuple of (new_qx, new_qy, new_qz, new_qw) representing rotated quaternion
        """
        # Create a quaternion for the additional rotation around z-axis
        half_angle = rotation_angle / 2
        rot_qx = 0
        rot_qy = 0
        rot_qz = math.sin(half_angle)
        rot_qw = math.cos(half_angle)
        
        # Multiply quaternions: q_new = q_rotation * q_original
        new_qw = rot_qw * qw - rot_qx * qx - rot_qy * qy - rot_qz * qz
        new_qx = rot_qw * qx + rot_qx * qw + rot_qy * qz - rot_qz * qy
        new_qy = rot_qw * qy - rot_qx * qz + rot_qy * qw + rot_qz * qx
        new_qz = rot_qw * qz + rot_qx * qy - rot_qy * qx + rot_qz * qw
        
        return new_qx, new_qy, new_qz, new_qw
    
    def load_scene_data(self, scene_name: str) -> Tuple[None, pd.DataFrame]:
        """
        Load trajectory data for a specific scene (no annotations).
        
        Args:
            scene_name: Name of the scene (e.g., "sample_096")
        
        Returns:
            Tuple of (None, trajectory_df) - annotations not used
        """
        # Load trajectory data only
        trajectory_file = self.data_path / f"{scene_name}.parquet"
        trajectory_df = pd.read_parquet(trajectory_file)
        
        return None, trajectory_df
    
    def extract_video_clip_frames(self, scene_name: str, start_frame: int, end_frame: int, 
                                 output_path: str = "temp_clip.mp4", output_fps: int = 5) -> str:
        """
        Extract a video clip using the same frame mapping logic as the GUI display.
        This ensures the saved video has the exact same frames as shown in the viewer.
        
        Args:
            scene_name: Name of the scene
            start_frame: Starting frame number (trajectory frame)
            end_frame: Ending frame number (trajectory frame)
            output_path: Output video file path
            output_fps: Frames per second for output video
        
        Returns:
            Path to the extracted video clip
        """
        video_file = self.video_path / f"{scene_name}_rgb_camera.mp4"
        
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        # Load trajectory data to get the segment
        trajectory_file = self.data_path / f"{scene_name}.parquet"
        if not trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
        
        trajectory_df = pd.read_parquet(trajectory_file)
        
        # Get trajectory segment (same as GUI)
        trajectory_segment = trajectory_df[
            (trajectory_df['frame'] >= start_frame) & 
            (trajectory_df['frame'] <= end_frame)
        ].copy()
        
        if len(trajectory_segment) == 0:
            raise ValueError("No trajectory data for the specified frame range")
        
        trajectory_frames = trajectory_segment['frame'].values
        
        # Open video and get properties
        cap = cv2.VideoCapture(str(video_file))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps <= 0:
            video_fps = 30  # Fallback FPS
        
        # Use the same frame mapping logic as the GUI
        traj_start = trajectory_frames[0]
        traj_end = trajectory_frames[-1]
        max_traj_frame = trajectory_df['frame'].max()
        
        # Map to video frames proportionally
        video_start_frame = int((traj_start / max_traj_frame) * total_video_frames)
        video_end_frame = int((traj_end / max_traj_frame) * total_video_frames)
        
        # Ensure bounds
        video_start_frame = max(0, min(video_start_frame, total_video_frames - 1))
        video_end_frame = max(video_start_frame + 1, min(video_end_frame, total_video_frames - 1))
        
        # Calculate frame step for uniform sampling
        if len(trajectory_frames) > 1:
            video_frame_span = video_end_frame - video_start_frame
            frame_step = video_frame_span / (len(trajectory_frames) - 1)
        else:
            frame_step = 0
        
        # Extract frames exactly as done in GUI
        extracted_frames = []
        for i, traj_frame in enumerate(trajectory_frames):
            if len(trajectory_frames) > 1:
                video_frame_pos = int(video_start_frame + i * frame_step)
            else:
                video_frame_pos = video_start_frame
            
            video_frame_pos = max(0, min(video_frame_pos, total_video_frames - 1))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_pos)
            ret, frame = cap.read()
            
            if ret and frame is not None and frame.size > 0:
                extracted_frames.append(frame)
            elif extracted_frames:
                # Duplicate last frame if current frame fails
                extracted_frames.append(extracted_frames[-1])
            else:
                # Create black placeholder if no frames yet
                placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                extracted_frames.append(placeholder_frame)
        
        cap.release()
        
        if not extracted_frames:
            raise ValueError("No frames could be extracted")
        
        # Write video using cv2.VideoWriter
        height, width = extracted_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        
        for frame in extracted_frames:
            out.write(frame)
        
        out.release()
        
        print(f"Extracted {len(extracted_frames)} frames to {output_path}")
        return output_path
    
    def transform_trajectory(self, trajectory_df: pd.DataFrame, start_frame: int, 
                           end_frame: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Transform trajectory coordinates to start from (0,0) with initial heading aligned to Y-axis.
        
        Args:
            trajectory_df: DataFrame containing trajectory data
            start_frame: Starting frame number
            end_frame: Ending frame number
        
        Returns:
            Tuple of (x_coords, y_coords, segment_df) where segment_df contains transformed quaternion data
        """
        # Filter trajectory data for the specified frame range
        segment_df = trajectory_df[
            (trajectory_df['frame'] >= start_frame) & 
            (trajectory_df['frame'] <= end_frame)
        ].copy()
        
        if len(segment_df) == 0:
            return np.array([]), np.array([]), pd.DataFrame()
        
        # Extract x, y coordinates - using pos_x, pos_y instead of cart_x, cart_y
        x_coords = segment_df['pos_x'].values
        y_coords = segment_df['pos_y'].values
        
        # Transform to start from (0, 0)
        x_coords = x_coords - x_coords[0]
        y_coords = y_coords - y_coords[0]
        
        # Get initial heading from the first quaternion
        first_row = segment_df.iloc[0]
        initial_qx, initial_qy, initial_qz, initial_qw = (
            first_row['quat_x'], first_row['quat_y'], first_row['quat_z'], first_row['quat_w'])
        
        # Calculate the rotation needed to align initial heading with Y-axis
        initial_heading = self.quaternion_to_heading(initial_qx, initial_qy, initial_qz, initial_qw)
        # We want to rotate so that initial heading points to Y-axis (90 degrees or π/2 radians)
        rotation_angle = (math.pi / 2) - initial_heading
        
        # Apply rotation to all trajectory points
        rotated_x, rotated_y = self.rotate_points(x_coords, y_coords, rotation_angle)
        
        # Update the segment_df with transformed coordinates and rotated quaternions
        segment_df = segment_df.copy()
        segment_df['transformed_x'] = rotated_x
        segment_df['transformed_y'] = rotated_y
        
        # Apply the same rotation to all quaternions in the segment
        rotated_quaternions = []
        for _, row in segment_df.iterrows():
            qx, qy, qz, qw = row['quat_x'], row['quat_y'], row['quat_z'], row['quat_w']
            new_qx, new_qy, new_qz, new_qw = self.rotate_quaternion_heading(qx, qy, qz, qw, rotation_angle)
            rotated_quaternions.append((new_qx, new_qy, new_qz, new_qw))
        
        # Update quaternion columns with rotated values
        rotated_quats = np.array(rotated_quaternions)
        segment_df['transformed_quat_x'] = rotated_quats[:, 0]
        segment_df['transformed_quat_y'] = rotated_quats[:, 1]
        segment_df['transformed_quat_z'] = rotated_quats[:, 2]
        segment_df['transformed_quat_w'] = rotated_quats[:, 3]
        
        # Store the rotation angle used for this segment
        segment_df['rotation_angle'] = rotation_angle
        
        return rotated_x, rotated_y, segment_df
    
    def transform_trajectory_with_reference_frame(self, trajectory_df: pd.DataFrame, 
                                                start_frame: int, end_frame: int, 
                                                reference_frame: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Transform trajectory segment with a specific reference frame as the origin and heading alignment point.
        
        Args:
            trajectory_df: Full trajectory DataFrame
            start_frame: Start frame of the segment to extract
            end_frame: End frame of the segment to extract  
            reference_frame: Frame to use as the transformation reference (origin and heading alignment)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, pd.DataFrame]: x_coords, y_coords, segment_df
        """
        # Extract the segment
        segment_df = trajectory_df[
            (trajectory_df['frame'] >= start_frame) & 
            (trajectory_df['frame'] <= end_frame)
        ].copy()
        
        if len(segment_df) == 0:
            return np.array([]), np.array([]), pd.DataFrame()
        
        # Find the reference point within the segment
        reference_rows = segment_df[segment_df['frame'] == reference_frame]
        if len(reference_rows) == 0:
            # If reference frame is not in segment, use the closest frame
            closest_idx = np.argmin(np.abs(segment_df['frame'] - reference_frame))
            reference_row = segment_df.iloc[closest_idx]
            print(f"Reference frame {reference_frame} not found in segment, using closest frame {reference_row['frame']}")
        else:
            reference_row = reference_rows.iloc[0]
        
        # Get reference position and heading - using pos_x, pos_y instead of cart_x, cart_y
        ref_x = reference_row['pos_x'] 
        ref_y = reference_row['pos_y']
        ref_qx = reference_row['quat_x']
        ref_qy = reference_row['quat_y'] 
        ref_qz = reference_row['quat_z']
        ref_qw = reference_row['quat_w']
        
        # Calculate reference heading
        reference_heading = self.quaternion_to_heading(ref_qx, ref_qy, ref_qz, ref_qw)
        
        # Extract x, y coordinates for all points in segment - using pos_x, pos_y
        x_coords = segment_df['pos_x'].values
        y_coords = segment_df['pos_y'].values
        
        # Translate all points so reference point becomes origin
        x_coords = x_coords - ref_x
        y_coords = y_coords - ref_y
        
        # Calculate rotation needed to align reference heading with Y-axis (90 degrees)
        rotation_angle = math.pi / 2 - reference_heading
        
        # Apply rotation to all points
        rotated_x, rotated_y = self.rotate_points(x_coords, y_coords, rotation_angle)
        
        # Update the segment_df with transformed coordinates
        segment_df = segment_df.copy()
        segment_df['transformed_x'] = rotated_x
        segment_df['transformed_y'] = rotated_y
        
        # Apply same rotation to all quaternions in the segment
        rotated_quaternions = []
        for _, row in segment_df.iterrows():
            qx, qy, qz, qw = row['quat_x'], row['quat_y'], row['quat_z'], row['quat_w']
            new_qx, new_qy, new_qz, new_qw = self.rotate_quaternion_heading(qx, qy, qz, qw, rotation_angle)
            rotated_quaternions.append((new_qx, new_qy, new_qz, new_qw))
        
        # Update quaternion columns with rotated values
        rotated_quats = np.array(rotated_quaternions)
        segment_df['transformed_quat_x'] = rotated_quats[:, 0]
        segment_df['transformed_quat_y'] = rotated_quats[:, 1]
        segment_df['transformed_quat_z'] = rotated_quats[:, 2]
        segment_df['transformed_quat_w'] = rotated_quats[:, 3]
        
        # Store the rotation angle and reference frame used for this segment
        segment_df['rotation_angle'] = rotation_angle
        segment_df['reference_frame'] = reference_frame
        
        return rotated_x, rotated_y, segment_df
    
    def list_available_scenes(self) -> List[str]:
        """
        List all available scenes in the dataset by scanning data directory.
        
        Returns:
            List of scene names
        """
        scenes = []
        for file in self.data_path.glob("*.parquet"):
            scene_name = file.stem  # e.g., "sample_096.parquet" -> "sample_096"
            scenes.append(scene_name)
        return sorted(scenes)


class TargetInteractiveViewer:
    def __init__(self, root, dataset_path="Target_trajectories"):
        self.root = root
        self.root.title("Target Dataset Interactive Viewer")
        self.root.geometry("1400x900")
        
        # Initialize the visualizer
        try:
            self.visualizer = TargetVisualizer(dataset_path)
        except Exception as e:
            messagebox.showerror("Error", "Failed to initialize visualizer: {}".format(str(e)))
            self.visualizer = None
        
        # Load available scenes
        self.scenes = self.visualizer.list_available_scenes() if self.visualizer else []
        self.current_scene_idx = 0
        
        # Current data
        self.current_trajectory_df = None
        self.current_video_cap = None
        self.current_video_frames = []
        self.current_trajectory_segment = None
        self.video_playing = False
        self.video_thread = None
        self.current_frame_idx = 0
        self.ego_position_dot = None
        
        # Caption for current selected segment (will be saved with each sample)
        # No persistent storage needed - caption is part of each saved sample
        
        # Create samples directory
        self.samples_dir = Path("Target_samples")
        self.samples_dir.mkdir(exist_ok=True)
        
        # Setup GUI
        self.setup_gui()
        
        # Load first scene
        if self.scenes and self.visualizer:
            self.load_scene(0)
        elif not self.scenes:
            messagebox.showwarning("Warning", "No scenes found in the dataset")
        elif not self.visualizer:
            messagebox.showerror("Error", "Visualizer not initialized properly")
    
    def setup_gui(self):
        """Set up the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Scene selection
        scene_frame = ttk.LabelFrame(control_frame, text="Scene Selection")
        scene_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(scene_frame, text="Scene:").pack(side=tk.LEFT, padx=5)
        self.scene_var = tk.StringVar()
        self.scene_combo = ttk.Combobox(scene_frame, textvariable=self.scene_var, 
                                       values=self.scenes, state="readonly", width=25)
        self.scene_combo.pack(side=tk.LEFT, padx=5)
        self.scene_combo.bind("<<ComboboxSelected>>", self.on_scene_selected)
        
        ttk.Button(scene_frame, text="Previous Scene", 
                  command=self.previous_scene).pack(side=tk.LEFT, padx=5)
        ttk.Button(scene_frame, text="Next Scene", 
                  command=self.next_scene).pack(side=tk.LEFT, padx=5)
        
        # Content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video
        left_panel = ttk.LabelFrame(content_frame, text="Video Clip")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video display
        self.video_frame = ttk.Frame(left_panel)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Use Canvas instead of Label for more reliable image display
        self.video_canvas = tk.Canvas(self.video_frame, bg='black')
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Fallback label for text messages
        self.video_label = ttk.Label(self.video_frame, text="Loading video...")
        self.video_label.pack_forget()  # Initially hidden
        
        # Video controls
        video_controls = ttk.Frame(left_panel)
        video_controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.play_button = ttk.Button(video_controls, text="Play", command=self.toggle_video)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Frame info
        self.frame_info = ttk.Label(video_controls, text="Frame: 0/0")
        self.frame_info.pack(side=tk.LEFT, padx=5)
        
        # Progress bar container with integrated segment selection
        progress_frame = ttk.Frame(video_controls)
        progress_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Mode selector for setting Start/Current/End
        mode_frame = ttk.Frame(progress_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(mode_frame, text="Click mode:").pack(side=tk.LEFT, padx=(0, 5))
        self.segment_mode = tk.StringVar(value="Current")
        ttk.Radiobutton(mode_frame, text="Start", variable=self.segment_mode, value="Start", 
                       command=self.on_mode_changed).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="Current", variable=self.segment_mode, value="Current", 
                       command=self.on_mode_changed).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="End", variable=self.segment_mode, value="End", 
                       command=self.on_mode_changed).pack(side=tk.LEFT, padx=2)
        
        # Single integrated progress bar with markers
        self.progress_canvas_frame = ttk.Frame(progress_frame)
        self.progress_canvas_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Canvas for custom progress bar with markers
        self.progress_canvas = tk.Canvas(self.progress_canvas_frame, height=60, bg='white', highlightthickness=1)
        self.progress_canvas.pack(fill=tk.X)
        self.progress_canvas.bind("<Button-1>", self.on_progress_canvas_click)
        self.progress_canvas.bind("<B1-Motion>", self.on_progress_canvas_drag)
        # Update canvas when window resizes
        self.progress_canvas.bind("<Configure>", lambda e: self.update_progress_canvas() if hasattr(self, 'max_video_frames') else None)
        
        # Initialize segment selection frames
        self.segment_start_frame = 0
        self.segment_end_frame = 0
        self.segment_current_frame = 0  # Current frame within segment
        self.max_video_frames = 0

        # Playback progress indicator will be a moving yellow marker on the canvas
        
        # Right panel - Trajectory and Info
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Trajectory plot
        trajectory_frame = ttk.LabelFrame(right_panel, text="Trajectory")
        # Make trajectory panel a bit smaller (do not expand)
        trajectory_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 6))
        
        # Create matplotlib figure for trajectory (slightly smaller)
        self.traj_fig = Figure(figsize=(5, 5), dpi=100)
        self.traj_ax = self.traj_fig.add_subplot(111)
        self.traj_canvas = FigureCanvasTkAgg(self.traj_fig, trajectory_frame)
        self.traj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        # Caption and info
        info_frame = ttk.LabelFrame(right_panel, text="Information")
        # Make information panel larger (expand to use remaining space)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Caption (editable)
        caption_label_frame = ttk.Frame(info_frame)
        caption_label_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(caption_label_frame, text="Segment Caption (editable):").pack(side=tk.LEFT)
        
        # Add button to reset caption to empty
        self.reset_caption_button = ttk.Button(caption_label_frame, text="Reset", 
                                             command=self.reset_caption, style="Toolbutton")
        self.reset_caption_button.pack(side=tk.RIGHT)
        
        self.caption_text = tk.Text(info_frame, height=5, wrap=tk.WORD, 
                                   bg="white", fg="black", insertbackground="black")
        self.caption_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Store original caption for reset functionality
        self.original_caption = ""
        
        # Statistics (adjusted height)
        self.stats_text = tk.Text(info_frame, height=8, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Save button
        save_frame = ttk.Frame(right_panel)
        save_frame.pack(fill=tk.X)
        
        self.save_button = ttk.Button(save_frame, text="Save Selected Segment", 
                                     command=self.save_current_sample, style='Accent.TButton')
        self.save_button.pack(expand=True, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def load_scene(self, scene_idx):
        """Load a scene by index"""
        if not self.visualizer:
            messagebox.showerror("Error", "Visualizer not available")
            return
            
        if 0 <= scene_idx < len(self.scenes):
            self.current_scene_idx = scene_idx
            scene_name = self.scenes[scene_idx]
            
            self.status_var.set("Loading scene: {}...".format(scene_name))
            self.root.update()
            
            try:
                # Load scene data (annotations will be None)
                _, self.current_trajectory_df = \
                    self.visualizer.load_scene_data(scene_name)
                
                # Update GUI
                self.scene_var.set(scene_name)
                
                # Clear caption box - user will add caption for each selected segment
                self.caption_text.delete(1.0, tk.END)
                self.original_caption = ""
                
                # Load full trajectory (no annotation-based segmentation)
                if self.current_trajectory_df is not None:
                    start_frame = self.current_trajectory_df['frame'].min()
                    end_frame = self.current_trajectory_df['frame'].max()
                    self.load_video_segment(start_frame, end_frame)
                    self.update_trajectory_plot(start_frame, end_frame)
                    self.update_statistics(start_frame, end_frame)
                    
                    # Reset segment bars to full range
                    self.reset_segment_bars()
                
                self.status_var.set("Loaded scene: {} ({} trajectory points)".format(
                    scene_name, len(self.current_trajectory_df) if self.current_trajectory_df is not None else 0))
                
            except Exception as e:
                messagebox.showerror("Error", "Failed to load scene: {}".format(str(e)))
                self.status_var.set("Error loading scene")
    
    def load_video_segment(self, start_frame, end_frame):
        """Load video frames for the segment matching trajectory timeframe exactly"""
        scene_name = self.scenes[self.current_scene_idx]
        video_path = self.visualizer.video_path / "{}_rgb_camera.mp4".format(scene_name)
        
        print("Loading video from: {}".format(video_path))
        print("Frame range: {} - {}".format(start_frame, end_frame))
        
        if not video_path.exists():
            print("Video file not found: {}".format(video_path))
            self.show_video_message("Video file not found")
            return
        
        try:
            # Get trajectory segment for synchronization
            trajectory_segment = self.current_trajectory_df[
                (self.current_trajectory_df['frame'] >= start_frame) & 
                (self.current_trajectory_df['frame'] <= end_frame)
            ].copy()
            
            if len(trajectory_segment) == 0:
                self.show_video_message("No trajectory data for this segment")
                return
            
            self.current_trajectory_segment = trajectory_segment
            trajectory_frames = trajectory_segment['frame'].values
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps if video_fps > 0 else 0
            
            print("Video FPS: {}, Total frames: {}, Duration: {:.2f}s".format(video_fps, total_video_frames, video_duration))
            
            # Handle invalid FPS
            if video_fps <= 0:
                print("Warning: Invalid video FPS detected, using fallback value")
                video_fps = 30  # Fallback FPS
            
            # Calculate the actual data sampling rate from trajectory
            trajectory_frame_diffs = np.diff(trajectory_frames)
            if len(trajectory_frame_diffs) > 0:
                avg_frame_diff = np.mean(trajectory_frame_diffs)
                print("Average trajectory frame difference: {:.2f}".format(avg_frame_diff))
                data_fps = video_fps / avg_frame_diff if avg_frame_diff > 0 else 5
                print("Estimated trajectory data rate: {:.2f} FPS".format(data_fps))
            else:
                data_fps = 5  # Default fallback
            
            self.current_video_frames = []
            valid_trajectory_indices = []
            
            print("Loading {} trajectory frames".format(len(trajectory_frames)))
            
            # Direct frame mapping approach: 
            # Map trajectory frame range proportionally to video frame range
            traj_start = trajectory_frames[0]
            traj_end = trajectory_frames[-1]
            traj_range = traj_end - traj_start
            
            # Map to video frames proportionally
            video_start_frame = int((traj_start / self.current_trajectory_df['frame'].max()) * total_video_frames)
            video_end_frame = int((traj_end / self.current_trajectory_df['frame'].max()) * total_video_frames)
            
            # Ensure we don't exceed video bounds
            video_start_frame = max(0, min(video_start_frame, total_video_frames - 1))
            video_end_frame = max(video_start_frame + 1, min(video_end_frame, total_video_frames - 1))
            
            print("Trajectory frame range: {} - {} (span: {})".format(traj_start, traj_end, traj_range))
            print("Mapped video frame range: {} - {}".format(video_start_frame, video_end_frame))
            
            # Calculate frame step for uniform sampling across the video segment
            if len(trajectory_frames) > 1:
                video_frame_span = video_end_frame - video_start_frame
                frame_step = video_frame_span / (len(trajectory_frames) - 1) if len(trajectory_frames) > 1 else 0
            else:
                frame_step = 0
            
            for i, traj_frame in enumerate(trajectory_frames):
                # Calculate video frame position using linear interpolation
                if len(trajectory_frames) > 1:
                    video_frame_pos = int(video_start_frame + i * frame_step)
                else:
                    video_frame_pos = video_start_frame
                
                # Ensure we don't go beyond video bounds
                video_frame_pos = max(0, min(video_frame_pos, total_video_frames - 1))
                
                # Set video position and read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_pos)
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Validate frame data
                    if frame_rgb.shape[0] > 0 and frame_rgb.shape[1] > 0:
                        self.current_video_frames.append(frame_rgb)
                        valid_trajectory_indices.append(i)
                        if i % 10 == 0:  # Print progress every 10 frames
                            video_time = video_frame_pos / video_fps
                            print("Loaded frame {}/{} (traj frame {}, video frame {}, video time {:.2f}s)".format(
                                i+1, len(trajectory_frames), traj_frame, video_frame_pos, video_time))
                else:
                    print("Failed to load frame {} (trajectory frame {}, video frame {})".format(i, traj_frame, video_frame_pos))
                    # If frame not available, duplicate last frame or create placeholder
                    if self.current_video_frames:
                        self.current_video_frames.append(self.current_video_frames[-1])
                        valid_trajectory_indices.append(i)
                    else:
                        # Create a black placeholder frame if no frames loaded yet
                        placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        self.current_video_frames.append(placeholder_frame)
                        valid_trajectory_indices.append(i)
            
            cap.release()
            
            # Update trajectory segment to match video frames
            if valid_trajectory_indices:
                self.current_trajectory_segment = self.current_trajectory_segment.iloc[valid_trajectory_indices].reset_index(drop=True)
                print("=== SYNCHRONIZATION SUMMARY ===")
                print("Video frames loaded: {}".format(len(self.current_video_frames)))
                print("Trajectory points: {}".format(len(self.current_trajectory_segment)))
                print("Video-Trajectory frame mapping:")
                print("  Video FPS: {:.1f}".format(video_fps))
                print("  Video frame range: {} - {}".format(video_start_frame, video_end_frame))
                print("  Trajectory frame range: {} - {}".format(traj_start, traj_end))
                print("  Frame step: {:.2f}".format(frame_step))
                print("=== END SUMMARY ===")
            else:
                print("Warning: No valid trajectory indices found")
            
            # Reset playback state
            self.current_frame_idx = 0
            
            # Display first frame and update controls
            if self.current_video_frames:
                print("Loaded {} video frames".format(len(self.current_video_frames)))
                max_frame = len(self.current_video_frames) - 1
                
                self.display_video_frame(0)
                # Initialize segment selection
                self.max_video_frames = max_frame
                self.reset_segment_bars()
                
                # Update frame info and canvas after a short delay to ensure canvas is sized
                self.root.after(100, lambda: (self.update_frame_info(), self.update_progress_canvas()))
            else:
                print("No video frames loaded")
                self.show_video_message("No video frames found")
                
        except Exception as e:
            self.show_video_message("Error loading video: {}".format(str(e)))
    
    def display_video_frame(self, frame_idx):
        """Display a specific video frame and update ego position"""
        # Basic validation first
        if not self.current_video_frames or frame_idx >= len(self.current_video_frames):
            return
        
        try:
            # Get frame safely
            frame = self.current_video_frames[frame_idx]
            self.current_frame_idx = frame_idx
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Get canvas dimensions
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            # Use default dimensions if canvas not yet sized
            if canvas_width <= 1:
                canvas_width = 640
            if canvas_height <= 1:
                canvas_height = 480
            
            # Resize image to fit canvas while maintaining aspect ratio
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for Tkinter
            photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.video_canvas.delete("all")
            
            # Center the image on canvas
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            self.video_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
            
            # Keep a reference to prevent garbage collection
            self.video_canvas.image = photo
            
            # Hide error message if visible
            self.hide_video_message()
            
        except Exception as e:
            print("Frame display error: {}".format(e))
            # Fallback to simple text display
            try:
                self.show_video_message("Frame {} - Error displaying".format(frame_idx))
            except:
                pass  # Ultimate fallback - do nothing
        
        # Update progress canvas (with simplified error handling)
        try:
            self.update_progress_canvas()
        except Exception:
            pass  # Ignore canvas update errors
        
        # Update frame info safely
        try:
            self.update_frame_info()
        except:
            pass
        
        # Update ego position on trajectory
        self.update_ego_position(frame_idx)
    
    def show_video_message(self, message):
        """Show a text message in the video area"""
        try:
            self.video_canvas.delete("all")
            self.video_label.config(text=str(message))
            self.video_label.pack(expand=True)
        except:
            pass  # Ultimate fallback
    
    def hide_video_message(self):
        """Hide the text message in the video area"""
        try:
            self.video_label.pack_forget()
        except:
            pass  # Ultimate fallback
    
    def update_trajectory_plot(self, start_frame, end_frame):
        """Update the trajectory plot with static elements"""
        # Clear previous plot
        self.traj_ax.clear()
        self.ego_position_dot = None
        
        # Get trajectory segment
        x_coords, y_coords, segment_df = self.visualizer.transform_trajectory(
            self.current_trajectory_df, start_frame, end_frame)
        
        if len(x_coords) == 0:
            self.traj_ax.text(0.5, 0.5, 'No trajectory data', 
                             ha='center', va='center', transform=self.traj_ax.transAxes)
            self.traj_canvas.draw()
            return
        
        # Store trajectory coordinates and quaternion data for ego position updates
        self.trajectory_x_coords = x_coords
        self.trajectory_y_coords = y_coords
        self.trajectory_segment_df = segment_df
        
        # Plot trajectory path
        self.traj_ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Trajectory Path')
        
        # Plot trajectory points as light dots
        self.traj_ax.scatter(x_coords, y_coords, c='lightblue', s=20, alpha=0.6, zorder=2)
        
        # Start and end points
        self.traj_ax.scatter(x_coords[0], y_coords[0], color='green', s=150, 
                            marker='o', label='Start', zorder=4, edgecolors='darkgreen', linewidth=2)
        self.traj_ax.scatter(x_coords[-1], y_coords[-1], color='blue', s=150, 
                            marker='s', label='End', zorder=4, edgecolors='darkblue', linewidth=2)
        
        # Add a legend entry for heading arrows (invisible arrow for legend only)
        if len(x_coords) > 5 and len(segment_df) > 0:
            # Create an invisible arrow just for the legend
            legend_arrow = self.traj_ax.arrow(x_coords[0], y_coords[0], 0, 0,
                                            head_width=0, head_length=0, 
                                            fc='green', ec='green', alpha=0, 
                                            label='Heading Direction')
        
        # Add heading arrows using quaternion data (fewer arrows for cleaner look)
        if len(x_coords) > 5 and len(segment_df) > 0:
            arrow_step = max(1, len(x_coords) // 8)
            # Calculate appropriate arrow length based on trajectory scale
            if len(x_coords) > 1:
                avg_step_size = np.mean(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
                arrow_length = max(0.5, avg_step_size * 2)
            else:
                arrow_length = 1.0
                
            for i in range(0, len(x_coords), arrow_step):
                if i < len(segment_df):
                    row = segment_df.iloc[i]
                    # Use transformed quaternions for correct heading after rotation
                    qx, qy, qz, qw = (row['transformed_quat_x'], row['transformed_quat_y'], 
                                     row['transformed_quat_z'], row['transformed_quat_w'])
                    
                    # Get heading vector from transformed quaternion
                    dx, dy = self.visualizer.get_heading_vector(qx, qy, qz, qw, arrow_length)
                    
                    # Draw heading arrow
                    self.traj_ax.arrow(x_coords[i], y_coords[i], dx, dy,
                                      head_width=arrow_length*0.25, head_length=arrow_length*0.15, 
                                      fc='green', ec='darkgreen', alpha=0.7, zorder=3, linewidth=1.5)
        
        # Initialize ego position dot (will be updated during playback)
        self.ego_position_dot = self.traj_ax.scatter(x_coords[0], y_coords[0], 
                                                    color='red', s=200, marker='o', 
                                                    label='Current Position', zorder=5,
                                                    edgecolors='darkred', linewidth=3)
        
        # Initialize ego heading arrow placeholder
        self.ego_heading_arrow = None
        
        # Set equal aspect ratio
        self.traj_ax.set_aspect('equal', adjustable='box')
        self.traj_ax.set_xlabel('X (meters)')
        self.traj_ax.set_ylabel('Y (meters) - Initial Heading Direction')
        title = f'Heading-Aligned Trajectory: Frames {start_frame}→{end_frame}'
        self.traj_ax.set_title(title)
        self.traj_ax.grid(True, alpha=0.3)
        
        # Position legend outside the plot area to avoid blocking the trajectory
        self.traj_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        # Refresh canvas
        self.traj_fig.tight_layout()
        self.traj_canvas.draw()
    
    def update_statistics(self, start_frame, end_frame):
        """Update statistics display with relative times and current frame info"""
        # Get trajectory segment
        x_coords, y_coords, _ = self.visualizer.transform_trajectory(
            self.current_trajectory_df, start_frame, end_frame)
        
        stats_text = ""
        
        # Get current frame within selected segment (absolute index)
        current_frame_abs = self.segment_current_frame
        start_idx = self.segment_start_frame
        end_idx = self.segment_end_frame
        
        # Calculate relative times using image_timestamp (Unix) from parquet
        durations_shown = False
        try:
            if (hasattr(self, 'current_trajectory_segment') and 
                self.current_trajectory_segment is not None and
                len(self.current_trajectory_segment) > 0):
                
                # Local helpers to normalize Unix timestamps to seconds
                def _normalize_unix_seconds(val):
                    try:
                        t = float(val)
                    except Exception:
                        return None
                    # Heuristics: downscale if stored in ns/us/ms
                    if t > 1e14:  # nanoseconds
                        return t / 1e9
                    if t > 1e12:  # microseconds
                        return t / 1e6
                    if t > 1e10:  # milliseconds
                        return t / 1e3
                    return t  # seconds

                def _ts_s(row):
                    v = row.get('image_timestamp', None)
                    if v is None:
                        return None
                    return _normalize_unix_seconds(v)

                # Get Start frame data
                if start_idx < len(self.current_trajectory_segment):
                    start_row = self.current_trajectory_segment.iloc[start_idx]
                    start_timestamp = _ts_s(start_row)
                    
                    # Get Current frame data (ensure it's within bounds)
                    current_idx = max(start_idx, min(current_frame_abs, end_idx))
                    if current_idx < len(self.current_trajectory_segment):
                        current_row = self.current_trajectory_segment.iloc[current_idx]
                        current_timestamp = _ts_s(current_row)
                        
                        # Get End frame data
                        if end_idx < len(self.current_trajectory_segment):
                            end_row = self.current_trajectory_segment.iloc[end_idx]
                            end_timestamp = _ts_s(end_row)
                            
                            # Calculate relative times
                            if (start_timestamp is not None and 
                                current_timestamp is not None and 
                                end_timestamp is not None):
                                relative_time_current = current_timestamp - start_timestamp
                                relative_time_end = end_timestamp - start_timestamp
                                duration_current_to_end = end_timestamp - current_timestamp
                                
                                # Explicit durations requested (with frame counts)
                                start_current_frames = max(0, current_idx - start_idx)
                                current_end_frames = max(0, end_idx - current_idx)
                                stats_text += f"Durations:\n"
                                stats_text += f"  Start->Current: {relative_time_current:.3f} s ({start_current_frames} frames)\n"
                                stats_text += f"  Current->End: {duration_current_to_end:.3f} s ({current_end_frames} frames)\n"
                                durations_shown = True
        except Exception as e:
            # If timestamp calculation fails, still show frame info (we won't add frame-in-segment)
            print(f"Warning: Could not calculate relative times: {e}")
            pass
        
        # Ensure durations are always shown (fallback to N/A if timestamps missing)
        if not durations_shown:
            # Compute frame counts even if timestamps missing
            try:
                current_idx_fallback = max(start_idx, min(current_frame_abs, end_idx))
            except Exception:
                current_idx_fallback = start_idx
            start_current_frames = max(0, current_idx_fallback - start_idx)
            current_end_frames = max(0, end_idx - current_idx_fallback)
            stats_text += f"Durations:\n"
            stats_text += f"  Start->Current: N/A ({start_current_frames} frames)\n"
            stats_text += f"  Current->End: N/A ({current_end_frames} frames)\n"
        
        if len(x_coords) > 1:
            # Calculate total distance
            distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
            total_distance = np.sum(distances)
            stats_text += f"\nTotal Distance: {total_distance:.2f} meters\n"
            stats_text += f"Trajectory Points: {len(x_coords)}"
        else:
            stats_text += "\nNo trajectory data available"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def update_ego_position(self, frame_idx):
        """Update the ego position dot and heading arrow on the trajectory plot"""
        if (self.ego_position_dot is None or 
            not hasattr(self, 'trajectory_x_coords') or 
            not hasattr(self, 'trajectory_y_coords')):
            return
        
        # Ensure frame_idx is within bounds
        if frame_idx >= len(self.trajectory_x_coords):
            frame_idx = len(self.trajectory_x_coords) - 1
        
        if frame_idx < 0:
            frame_idx = 0
        
        # Update the position of the ego dot and heading arrow
        try:
            x_pos = self.trajectory_x_coords[frame_idx]
            y_pos = self.trajectory_y_coords[frame_idx]
            
            # Update the scatter plot data
            self.ego_position_dot.set_offsets([[x_pos, y_pos]])
            
            # Remove previous ego heading arrow if it exists
            if hasattr(self, 'ego_heading_arrow') and self.ego_heading_arrow:
                self.ego_heading_arrow.remove()
                self.ego_heading_arrow = None
            
            # Add current heading arrow if quaternion data is available
            if (hasattr(self, 'trajectory_segment_df') and 
                len(self.trajectory_segment_df) > frame_idx):
                
                row = self.trajectory_segment_df.iloc[frame_idx]
                # Use transformed quaternions for correct heading after rotation
                qx, qy, qz, qw = (row['transformed_quat_x'], row['transformed_quat_y'], 
                                 row['transformed_quat_z'], row['transformed_quat_w'])
                
                # Calculate arrow length based on trajectory scale
                if len(self.trajectory_x_coords) > 1:
                    avg_step_size = np.mean(np.sqrt(np.diff(self.trajectory_x_coords)**2 + 
                                                   np.diff(self.trajectory_y_coords)**2))
                    arrow_length = max(1.0, avg_step_size * 3)
                else:
                    arrow_length = 1.5
                
                # Get heading vector from transformed quaternion
                dx, dy = self.visualizer.get_heading_vector(qx, qy, qz, qw, arrow_length)
                
                # Draw current heading arrow (larger and more prominent)
                self.ego_heading_arrow = self.traj_ax.arrow(
                    x_pos, y_pos, dx, dy,
                    head_width=arrow_length*0.3, head_length=arrow_length*0.2, 
                    fc='red', ec='darkred', alpha=0.9, zorder=6, linewidth=2)
            
            # Use a safer redraw method to avoid recursion
            try:
                self.traj_canvas.draw_idle()
            except:
                # Fallback to a simple update if draw_idle fails
                pass
            
        except (IndexError, AttributeError) as e:
            print("Error updating ego position: {}".format(e))
            pass  # Ignore errors during position updates
    
    def update_frame_info(self):
        """Update the frame information display"""
        if self.current_video_frames:
            total_frames = len(self.current_video_frames)
            current_frame = self.current_frame_idx + 1
            frame_text = "Frame: {}/{}".format(current_frame, total_frames)
            self.frame_info.config(text=frame_text)
        else:
            self.frame_info.config(text="Frame: 0/0")
    
    def on_mode_changed(self):
        """Handle mode change (Start/Current/End)"""
        # Mode change doesn't need to do anything, just visual feedback
        pass
    
    def on_progress_canvas_click(self, event):
        """Handle click on progress canvas to set Start/Current/End"""
        if not self.current_video_frames or self.max_video_frames == 0:
            return
            
        canvas_width = self.progress_canvas.winfo_width()
        if canvas_width <= 1:
            return
        
        # Calculate frame index from click position
        click_x = event.x
        frame_idx = int((click_x / canvas_width) * self.max_video_frames)
        frame_idx = max(0, min(frame_idx, self.max_video_frames))
        
        # Stop playback if playing
        if self.video_playing:
            self.stop_video()
                
        # Set frame based on current mode
        mode = self.segment_mode.get()
        if mode == "Start":
            # Ensure start is before end
            if frame_idx >= self.segment_end_frame and self.segment_end_frame < self.max_video_frames:
                frame_idx = max(0, self.segment_end_frame - 1)
            self.segment_start_frame = frame_idx
            # Constrain current if needed
            if self.segment_current_frame < frame_idx:
                self.segment_current_frame = frame_idx
            self.current_frame_idx = frame_idx
        elif mode == "End":
            # Ensure end is after start
            if frame_idx <= self.segment_start_frame and self.segment_start_frame > 0:
                frame_idx = min(self.max_video_frames, self.segment_start_frame + 1)
            self.segment_end_frame = frame_idx
            # Constrain current if needed
            if self.segment_current_frame > frame_idx:
                self.segment_current_frame = frame_idx
        else:  # Current
            # Constrain current between start and end
            frame_idx = max(self.segment_start_frame, min(frame_idx, self.segment_end_frame))
            self.segment_current_frame = frame_idx
            self.current_frame_idx = frame_idx
        
        # Update display
        self.display_video_frame(self.segment_current_frame)
        self.update_segment_info()
        
        # Update statistics
        if self.current_trajectory_df is not None and len(self.current_trajectory_segment) > 0:
            start_traj_frame = self.current_trajectory_segment.iloc[self.segment_start_frame]['frame'] if self.segment_start_frame < len(self.current_trajectory_segment) else 0
            end_traj_frame = self.current_trajectory_segment.iloc[self.segment_end_frame]['frame'] if self.segment_end_frame < len(self.current_trajectory_segment) else 0
            self.update_statistics(start_traj_frame, end_traj_frame)
    
    def on_progress_canvas_drag(self, event):
        """Handle drag on progress canvas"""
        # Same as click handler
        self.on_progress_canvas_click(event)
    
    def update_progress_canvas(self):
        """Update the progress canvas visualization with Start, Current, End markers"""
        if not hasattr(self, 'progress_canvas') or not self.current_video_frames:
            return
        
        canvas_width = self.progress_canvas.winfo_width()
        canvas_height = self.progress_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Clear canvas
        self.progress_canvas.delete("all")
        
        if self.max_video_frames == 0:
            return
        
        # Draw background bar
        bar_height = 20
        bar_y = (canvas_height - bar_height) // 2
        self.progress_canvas.create_rectangle(0, bar_y, canvas_width, bar_y + bar_height, 
                                            fill='lightgray', outline='black', width=1)
        
        # Calculate positions
        start_x = (self.segment_start_frame / self.max_video_frames) * canvas_width
        current_x = (self.segment_current_frame / self.max_video_frames) * canvas_width
        end_x = (self.segment_end_frame / self.max_video_frames) * canvas_width
        
        # Draw selected segment highlight (neutral base color)
        self.progress_canvas.create_rectangle(start_x, bar_y, end_x, bar_y + bar_height, 
                                            fill='lightblue', outline='', tags='segment')
                
        # Draw Start marker (green triangle pointing up)
        marker_size = 8
        start_y_top = bar_y - marker_size
        # Margins to keep labels inside canvas bounds
        margin_x = 24
        margin_y = 8
        self.progress_canvas.create_polygon(
            start_x, bar_y + bar_height,
            start_x - marker_size, start_y_top,
            start_x + marker_size, start_y_top,
            fill='green', outline='darkgreen', width=2, tags='start_marker'
        )
        # Clamp start label within bounds
        start_label_x = max(margin_x, min(start_x, canvas_width - margin_x))
        start_label_y = max(margin_y, min(start_y_top - 10, canvas_height - margin_y))
        self.progress_canvas.create_text(start_label_x, start_label_y, text=f"S:{self.segment_start_frame}", 
                                       font=('Arial', 8), fill='green', tags='start_label')
        
        # Draw Current marker (blue triangle pointing down)
        current_y_bottom = bar_y + bar_height + marker_size
        self.progress_canvas.create_polygon(
            current_x, bar_y,
            current_x - marker_size, current_y_bottom,
            current_x + marker_size, current_y_bottom,
            fill='blue', outline='darkblue', width=2, tags='current_marker'
        )
        # Clamp current label within bounds
        current_label_x = max(margin_x, min(current_x, canvas_width - margin_x))
        current_label_y = max(margin_y, min(current_y_bottom + 15, canvas_height - margin_y))
        self.progress_canvas.create_text(current_label_x, current_label_y, text=f"C:{self.segment_current_frame}", 
                                       font=('Arial', 8), fill='blue', tags='current_label')
        
        # Draw End marker (black triangle pointing up)
        end_y_top = bar_y - marker_size
        self.progress_canvas.create_polygon(
            end_x, bar_y + bar_height,
            end_x - marker_size, end_y_top,
            end_x + marker_size, end_y_top,
            fill='black', outline='darkgray', width=2, tags='end_marker'
        )
        # Clamp end label within bounds
        end_label_x = max(margin_x, min(end_x, canvas_width - margin_x))
        end_label_y = max(margin_y, min(end_y_top - 10, canvas_height - margin_y))
        self.progress_canvas.create_text(end_label_x, end_label_y, text=f"E:{self.segment_end_frame}", 
                                       font=('Arial', 8), fill='black', tags='end_label')

        # Draw playback progress marker (yellow) as a thin vertical line
        # Use current playback frame if playing, otherwise reflect selected Current
        playback_frame = self.current_frame_idx if self.video_playing else self.segment_current_frame
        playback_frame = max(self.segment_start_frame, min(playback_frame, self.segment_end_frame))
        playback_x = (playback_frame / self.max_video_frames) * canvas_width
        self.progress_canvas.create_line(playback_x, bar_y - 10, playback_x, bar_y + bar_height + 10,
                                         fill='gold', width=2, tags='playback_marker')
    
    def update_segment_info(self):
        """Update the display to show selected segment information"""
        if self.current_video_frames:
            duration = self.segment_end_frame - self.segment_start_frame + 1
            # Update status with format: Selected Segment (start-current-end)
            self.status_var.set(f"Selected Segment ({self.segment_start_frame}-{self.segment_current_frame}-{self.segment_end_frame})")
            
            # Update trajectory plot to highlight selected segment
            self.highlight_selected_segment()
            
            # Update the progress canvas visualization
            self.update_progress_canvas()
            # Playback marker is drawn on canvas; no separate slider updates needed

            # Update statistics (durations) when selection changes
            try:
                if (self.current_trajectory_df is not None and 
                    hasattr(self, 'current_trajectory_segment') and 
                    self.current_trajectory_segment is not None and
                    len(self.current_trajectory_segment) > 0):
                    start_traj_frame = self.current_trajectory_segment.iloc[self.segment_start_frame]['frame'] if self.segment_start_frame < len(self.current_trajectory_segment) else 0
                    end_traj_frame = self.current_trajectory_segment.iloc[self.segment_end_frame]['frame'] if self.segment_end_frame < len(self.current_trajectory_segment) else 0
                    self.update_statistics(start_traj_frame, end_traj_frame)
            except:
                pass
    
    def reset_segment_bars(self):
        """Reset segment selection to full range"""
        if self.current_video_frames:
            max_frame = len(self.current_video_frames) - 1
            self.max_video_frames = max_frame
            self.segment_start_frame = 0
            self.segment_end_frame = max_frame
            self.segment_current_frame = 0
            self.current_frame_idx = 0
            self.update_segment_info()
            self.update_progress_canvas()
    
    def reset_caption(self):
        """Reset caption to empty (no default caption for segments)"""
        self.caption_text.delete(1.0, tk.END)
        self.original_caption = ""
    
    def get_current_caption(self):
        """Get the current (possibly modified) caption text"""
        return self.caption_text.get(1.0, tk.END).strip()
    
    def get_selected_segment_data(self):
        """Get trajectory data for the selected segment with progress bar based transformation"""
        if (self.current_trajectory_segment is None or 
            not hasattr(self, 'segment_start_frame') or 
            not hasattr(self, 'segment_end_frame')):
            return None, None, None
        
        # Map video frame indices to trajectory frame indices
        start_video_idx = self.segment_start_frame
        end_video_idx = self.segment_end_frame
        
        # Ensure indices are within bounds
        max_video_idx = len(self.current_video_frames) - 1 if self.current_video_frames else 0
        start_video_idx = max(0, min(start_video_idx, max_video_idx))
        end_video_idx = max(start_video_idx, min(end_video_idx, max_video_idx))
        
        # Extract the corresponding trajectory segment
        if (hasattr(self, 'current_trajectory_segment') and 
            len(self.current_trajectory_segment) > end_video_idx):
            
            selected_traj_segment = self.current_trajectory_segment.iloc[start_video_idx:end_video_idx+1].copy()
            
            if len(selected_traj_segment) == 0:
                return None, None, None
            
            # Get the actual frame range from trajectory data
            traj_start_frame = selected_traj_segment.iloc[0]['frame']
            traj_end_frame = selected_traj_segment.iloc[-1]['frame']
            
            # Get the current frame position (use segment_current_frame if available, otherwise progress bar)
            if hasattr(self, 'segment_current_frame'):
                current_progress_idx = self.segment_current_frame
            else:
                current_progress_idx = self.current_frame_idx
            
            # Ensure current is within selected segment bounds
            current_progress_idx = max(start_video_idx, min(current_progress_idx, end_video_idx))
            
            # Convert progress bar video frame index to trajectory frame 
            if (hasattr(self, 'current_trajectory_segment') and 
                current_progress_idx < len(self.current_trajectory_segment)):
                reference_traj_frame = self.current_trajectory_segment.iloc[current_progress_idx]['frame']
            else:
                # Fallback to start frame if progress bar is out of range
                print(f"Warning: Current frame position (video frame {current_progress_idx}) is outside trajectory segment")
                reference_traj_frame = traj_start_frame
            
            # Use the new transformation method that centers on the progress bar's current frame
            x_coords, y_coords, segment_df = self.visualizer.transform_trajectory_with_reference_frame(
                self.current_trajectory_df, traj_start_frame, traj_end_frame, reference_traj_frame)
            
            print(f"Transformed segment based on progress bar frame {reference_traj_frame} (video frame {current_progress_idx})")
            
            return x_coords, y_coords, segment_df
        
        return None, None, None
    
    def save_selected_video_frames(self, output_path):
        """Save the video frames for the selected segment"""
        if not self.current_video_frames:
            raise ValueError("No video frames available")
        
        # Get selected frame range
        start_idx = self.segment_start_frame
        end_idx = self.segment_end_frame
        max_idx = len(self.current_video_frames) - 1
        
        start_idx = max(0, min(start_idx, max_idx))
        end_idx = max(start_idx, min(end_idx, max_idx))
        
        # Extract selected frames
        selected_frames = self.current_video_frames[start_idx:end_idx+1]
        
        if not selected_frames:
            raise ValueError("No frames in selected range")
        
        # Write video using cv2.VideoWriter
        height, width = selected_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))  # 25 FPS
        
        for frame in selected_frames:
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Saved {len(selected_frames)} frames to {output_path}")

    def save_current_to_end_video(self, output_path, fps: float = 25.0):
        """Save a video from the selected Current frame to the End frame at the given FPS."""
        if not self.current_video_frames:
            raise ValueError("No video frames available")
        
        # Determine indices
        start_idx = getattr(self, 'segment_current_frame', None)
        if start_idx is None:
            start_idx = getattr(self, 'current_frame_idx', 0)
        end_idx = getattr(self, 'segment_end_frame', None)
        if end_idx is None:
            end_idx = len(self.current_video_frames) - 1
        
        max_idx = len(self.current_video_frames) - 1
        start_idx = max(0, min(start_idx, max_idx))
        end_idx = max(start_idx, min(end_idx, max_idx))
        
        selected_frames = self.current_video_frames[start_idx:end_idx+1]
        if not selected_frames:
            raise ValueError("No frames in current-to-end range")
        
        # Write video
        height, width = selected_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        for frame in selected_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"Saved {len(selected_frames)} frames (current->end) to {output_path} @ {fps} FPS")
    
    def save_first_frame_image(self, output_path):
        """Save the first frame of the selected segment as an image"""
        if not self.current_video_frames:
            raise ValueError("No video frames available")
        
        # Get the first frame of the selected segment
        start_idx = self.segment_start_frame
        max_idx = len(self.current_video_frames) - 1
        start_idx = max(0, min(start_idx, max_idx))
        
        if start_idx >= len(self.current_video_frames):
            raise ValueError("Selected start frame is out of range")
        
        # Get the first frame
        first_frame = self.current_video_frames[start_idx]
        
        # Convert RGB to PIL Image
        pil_image = Image.fromarray(first_frame)
        
        # Save as PNG for best quality
        pil_image.save(output_path, 'PNG')
        print(f"Saved first frame image to {output_path}")
    
    def save_current_frame_image(self, output_path):
        """Save the current frame (where Current slider is positioned) as an image"""
        if not self.current_video_frames:
            raise ValueError("No video frames available")
        
        # Get the current frame index from the Current slider position
        if hasattr(self, 'segment_current_frame'):
            current_idx = self.segment_current_frame
        else:
            current_idx = self.current_frame_idx
        max_idx = len(self.current_video_frames) - 1
        current_idx = max(0, min(current_idx, max_idx))
        
        if current_idx >= len(self.current_video_frames):
            raise ValueError("Current frame index is out of range")
        
        # Get the current frame
        current_frame = self.current_video_frames[current_idx]
        
        # Convert RGB to PIL Image
        pil_image = Image.fromarray(current_frame)
        
        # Save as PNG for best quality
        pil_image.save(output_path, 'PNG')
        print(f"Saved current frame image (frame {current_idx}) to {output_path}")
    
    def get_progress_bar_state(self):
        """Get the current state of the progress bar and playback"""
        if not self.current_video_frames:
            return None
        
        total_frames = len(self.current_video_frames)
        current_frame = self.current_frame_idx
        progress_percentage = (current_frame / max(1, total_frames - 1)) * 100
        
        # Get corresponding trajectory information if available
        trajectory_info = None
        if (hasattr(self, 'current_trajectory_segment') and 
            self.current_trajectory_segment is not None and
            current_frame < len(self.current_trajectory_segment)):
            
            traj_row = self.current_trajectory_segment.iloc[current_frame]
            trajectory_info = {
                'trajectory_frame': int(traj_row['frame']),
                'position_x': float(traj_row.get('pos_x', 0)),
                'position_y': float(traj_row.get('pos_y', 0)),
                'position_z': float(traj_row.get('pos_z', 0)),
                'quaternion_x': float(traj_row.get('quat_x', 0)),
                'quaternion_y': float(traj_row.get('quat_y', 0)),
                'quaternion_z': float(traj_row.get('quat_z', 0)),
                'quaternion_w': float(traj_row.get('quat_w', 1))
            }
            
            # Add transformed position if available
            if hasattr(self, 'trajectory_x_coords') and hasattr(self, 'trajectory_y_coords'):
                if current_frame < len(self.trajectory_x_coords):
                    trajectory_info['transformed_x'] = float(self.trajectory_x_coords[current_frame])
                    trajectory_info['transformed_y'] = float(self.trajectory_y_coords[current_frame])
        
        return {
            'current_video_frame_index': int(current_frame),
            'total_video_frames': int(total_frames),
            'progress_percentage': float(progress_percentage),
            'is_playing': bool(self.video_playing),
            'trajectory_info': trajectory_info
        }
    
    def calculate_distance_first_to_current(self):
        """Calculate distance between first frame and current progress bar frame (x,y only)"""
        if (not hasattr(self, 'current_trajectory_segment') or 
            self.current_trajectory_segment is None or
            not hasattr(self, 'segment_start_frame') or
            not hasattr(self, 'segment_end_frame')):
            return None
        
        # Get the selected segment bounds
        start_idx = self.segment_start_frame
        end_idx = self.segment_end_frame
        # Use segment_current_frame if available, otherwise fall back to current_frame_idx
        if hasattr(self, 'segment_current_frame'):
            current_progress_idx = self.segment_current_frame
        else:
            current_progress_idx = self.current_frame_idx
        
        # Ensure indices are within bounds
        max_video_idx = len(self.current_video_frames) - 1 if self.current_video_frames else 0
        start_idx = max(0, min(start_idx, max_video_idx))
        end_idx = max(start_idx, min(end_idx, max_video_idx))
        current_progress_idx = max(0, min(current_progress_idx, max_video_idx))
        
        # Check if we have trajectory data for these indices
        if (start_idx >= len(self.current_trajectory_segment) or 
            current_progress_idx >= len(self.current_trajectory_segment)):
            return None
        
        # Helper to normalize Unix timestamp to seconds
        def _norm_ts(v):
            try:
                t = float(v)
            except Exception:
                return None
            if t > 1e14:
                return t / 1e9
            if t > 1e12:
                return t / 1e6
            if t > 1e10:
                return t / 1e3
            return t
        
        # Get the first frame of the selected segment - using pos_x, pos_y
        first_frame_data = self.current_trajectory_segment.iloc[start_idx]
        first_x = first_frame_data['pos_x']
        first_y = first_frame_data['pos_y']
        first_frame_num = first_frame_data['frame']
        first_z = first_frame_data.get('pos_z', None)
        first_qx = first_frame_data.get('quat_x', None)
        first_qy = first_frame_data.get('quat_y', None)
        first_qz = first_frame_data.get('quat_z', None)
        first_qw = first_frame_data.get('quat_w', None)
        first_ts = _norm_ts(first_frame_data.get('image_timestamp', None))
        # Transformed coordinates, if available
        first_tx = None
        first_ty = None
        if hasattr(self, 'trajectory_x_coords') and hasattr(self, 'trajectory_y_coords'):
            if start_idx < len(self.trajectory_x_coords):
                first_tx = float(self.trajectory_x_coords[start_idx])
                first_ty = float(self.trajectory_y_coords[start_idx])
        
        # Get the current progress bar frame data - using pos_x, pos_y
        current_frame_data = self.current_trajectory_segment.iloc[current_progress_idx]
        current_x = current_frame_data['pos_x']
        current_y = current_frame_data['pos_y']
        current_frame_num = current_frame_data['frame']
        current_z = current_frame_data.get('pos_z', None)
        current_qx = current_frame_data.get('quat_x', None)
        current_qy = current_frame_data.get('quat_y', None)
        current_qz = current_frame_data.get('quat_z', None)
        current_qw = current_frame_data.get('quat_w', None)
        current_ts = _norm_ts(current_frame_data.get('image_timestamp', None))
        current_tx = None
        current_ty = None
        if hasattr(self, 'trajectory_x_coords') and hasattr(self, 'trajectory_y_coords'):
            if current_progress_idx < len(self.trajectory_x_coords):
                current_tx = float(self.trajectory_x_coords[current_progress_idx])
                current_ty = float(self.trajectory_y_coords[current_progress_idx])
        
        # Calculate Euclidean distance using only x, y coordinates
        distance = ((current_x - first_x)**2 + (current_y - first_y)**2)**0.5
        # Compute time duration in seconds if timestamps available
        duration_seconds = None
        if first_ts is not None and current_ts is not None:
            duration_seconds = float(current_ts - first_ts)
        
        return {
            'distance_meters': float(distance),
            'time_duration_seconds': float(duration_seconds) if duration_seconds is not None else None,
            'first_frame': {
                'trajectory_frame': int(first_frame_num),
                'video_index': int(start_idx),
                'position_x': float(first_x),
                'position_y': float(first_y),
                'position_z': float(first_z) if first_z is not None else None,
                'quaternion_x': float(first_qx) if first_qx is not None else None,
                'quaternion_y': float(first_qy) if first_qy is not None else None,
                'quaternion_z': float(first_qz) if first_qz is not None else None,
                'quaternion_w': float(first_qw) if first_qw is not None else None,
                'transformed_x': first_tx,
                'transformed_y': first_ty
            },
            'current_frame': {
                'trajectory_frame': int(current_frame_num),
                'video_index': int(current_progress_idx),
                'position_x': float(current_x),
                'position_y': float(current_y),
                'position_z': float(current_z) if current_z is not None else None,
                'quaternion_x': float(current_qx) if current_qx is not None else None,
                'quaternion_y': float(current_qy) if current_qy is not None else None,
                'quaternion_z': float(current_qz) if current_qz is not None else None,
                'quaternion_w': float(current_qw) if current_qw is not None else None,
                'transformed_x': current_tx,
                'transformed_y': current_ty
            },
            'calculation_method': 'euclidean_distance_xy_only'
        }

    def calculate_distance_current_to_end(self):
        """Calculate distance between current progress frame and end frame (x,y only)."""
        if (not hasattr(self, 'current_trajectory_segment') or 
            self.current_trajectory_segment is None or
            not hasattr(self, 'segment_start_frame') or
            not hasattr(self, 'segment_end_frame')):
            return None
        
        # Get the selected segment bounds
        start_idx = self.segment_start_frame
        end_idx = self.segment_end_frame
        # Use segment_current_frame if available, otherwise fall back to current_frame_idx
        if hasattr(self, 'segment_current_frame'):
            current_progress_idx = self.segment_current_frame
        else:
            current_progress_idx = self.current_frame_idx
        
        # Ensure indices are within bounds
        max_video_idx = len(self.current_video_frames) - 1 if self.current_video_frames else 0
        start_idx = max(0, min(start_idx, max_video_idx))
        end_idx = max(start_idx, min(end_idx, max_video_idx))
        current_progress_idx = max(0, min(current_progress_idx, max_video_idx))
        
        # Check if we have trajectory data for these indices
        if (end_idx >= len(self.current_trajectory_segment) or 
            current_progress_idx >= len(self.current_trajectory_segment)):
            return None
        
        # Helper to normalize Unix timestamp to seconds
        def _norm_ts(v):
            try:
                t = float(v)
            except Exception:
                return None
            if t > 1e14:
                return t / 1e9
            if t > 1e12:
                return t / 1e6
            if t > 1e10:
                return t / 1e3
            return t

        # Get the current progress frame data
        current_frame_data = self.current_trajectory_segment.iloc[current_progress_idx]
        current_x = current_frame_data['pos_x']
        current_y = current_frame_data['pos_y']
        current_frame_num = current_frame_data['frame']
        current_z = current_frame_data.get('pos_z', None)
        current_qx = current_frame_data.get('quat_x', None)
        current_qy = current_frame_data.get('quat_y', None)
        current_qz = current_frame_data.get('quat_z', None)
        current_qw = current_frame_data.get('quat_w', None)
        current_ts = _norm_ts(current_frame_data.get('image_timestamp', None))
        current_tx = None
        current_ty = None
        if hasattr(self, 'trajectory_x_coords') and hasattr(self, 'trajectory_y_coords'):
            if current_progress_idx < len(self.trajectory_x_coords):
                current_tx = float(self.trajectory_x_coords[current_progress_idx])
                current_ty = float(self.trajectory_y_coords[current_progress_idx])
        
        # Get the end frame data
        end_frame_data = self.current_trajectory_segment.iloc[end_idx]
        end_x = end_frame_data['pos_x']
        end_y = end_frame_data['pos_y']
        end_frame_num = end_frame_data['frame']
        end_z = end_frame_data.get('pos_z', None)
        end_qx = end_frame_data.get('quat_x', None)
        end_qy = end_frame_data.get('quat_y', None)
        end_qz = end_frame_data.get('quat_z', None)
        end_qw = end_frame_data.get('quat_w', None)
        end_ts = _norm_ts(end_frame_data.get('image_timestamp', None))
        end_tx = None
        end_ty = None
        if hasattr(self, 'trajectory_x_coords') and hasattr(self, 'trajectory_y_coords'):
            if end_idx < len(self.trajectory_x_coords):
                end_tx = float(self.trajectory_x_coords[end_idx])
                end_ty = float(self.trajectory_y_coords[end_idx])
        
        # Calculate Euclidean distance using only x, y coordinates
        distance = ((end_x - current_x)**2 + (end_y - current_y)**2)**0.5
        # Duration seconds if timestamps available
        duration_seconds = None
        if current_ts is not None and end_ts is not None:
            duration_seconds = float(end_ts - current_ts)
        
        return {
            'distance_meters': float(distance),
            'time_duration_seconds': float(duration_seconds) if duration_seconds is not None else None,
            'current_frame': {
                'trajectory_frame': int(current_frame_num),
                'video_index': int(current_progress_idx),
                'position_x': float(current_x),
                'position_y': float(current_y),
                'position_z': float(current_z) if current_z is not None else None,
                'quaternion_x': float(current_qx) if current_qx is not None else None,
                'quaternion_y': float(current_qy) if current_qy is not None else None,
                'quaternion_z': float(current_qz) if current_qz is not None else None,
                'quaternion_w': float(current_qw) if current_qw is not None else None,
                'transformed_x': current_tx,
                'transformed_y': current_ty
            },
            'end_frame': {
                'trajectory_frame': int(end_frame_num),
                'video_index': int(end_idx),
                'position_x': float(end_x),
                'position_y': float(end_y),
                'position_z': float(end_z) if end_z is not None else None,
                'quaternion_x': float(end_qx) if end_qx is not None else None,
                'quaternion_y': float(end_qy) if end_qy is not None else None,
                'quaternion_z': float(end_qz) if end_qz is not None else None,
                'quaternion_w': float(end_qw) if end_qw is not None else None,
                'transformed_x': end_tx,
                'transformed_y': end_ty
            },
            'calculation_method': 'euclidean_distance_xy_only'
        }
    
    def highlight_selected_segment(self):
        """Highlight the selected segment on the trajectory plot"""
        if (not hasattr(self, 'trajectory_x_coords') or 
            not hasattr(self, 'trajectory_y_coords') or
            not hasattr(self, 'segment_start_frame') or
            not hasattr(self, 'segment_end_frame')):
            return
            
        # Remove previous highlight if it exists
        if hasattr(self, 'segment_highlight') and self.segment_highlight:
            try:
                self.segment_highlight.remove()
            except:
                pass
            self.segment_highlight = None
        
        try:
            # Get the coordinates for the selected segment
            start_idx = max(0, min(self.segment_start_frame, len(self.trajectory_x_coords) - 1))
            end_idx = max(start_idx, min(self.segment_end_frame, len(self.trajectory_x_coords) - 1))
            
            if start_idx < len(self.trajectory_x_coords) and end_idx < len(self.trajectory_x_coords):
                # Highlight the selected portion of the trajectory
                x_segment = self.trajectory_x_coords[start_idx:end_idx+1]
                y_segment = self.trajectory_y_coords[start_idx:end_idx+1]
                
                if len(x_segment) > 1:
                    # Draw highlighted trajectory segment
                    self.segment_highlight = self.traj_ax.plot(
                        x_segment, y_segment, 'r-', linewidth=4, alpha=0.8, 
                        label=f'Selected Segment ({start_idx}-{self.segment_current_frame}-{end_idx})', zorder=10)[0]
                    
                    # Update legend
                    self.traj_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    
                    # Refresh canvas
                    self.traj_canvas.draw_idle()
        except Exception as e:
            print(f"Error highlighting segment: {e}")
    
    def toggle_video(self):
        """Toggle video playback"""
        print("Toggle video called")
        print("Current video frames: {}".format(len(self.current_video_frames) if self.current_video_frames else 0))
        print("Video playing: {}".format(self.video_playing))
        
        if not self.current_video_frames:
            print("No video frames available")
            return
        
        if self.video_playing:
            print("Stopping video")
            self.stop_video()
        else:
            print("Starting video")
            self.play_video()
    
    def play_video(self):
        """Start video playback with synchronized trajectory updates - plays from Start to End"""
        print("Play video called")
        if not self.current_video_frames:
            print("No video frames in play_video")
            return
        
        # Always start playback from Start frame
        if hasattr(self, 'segment_start_frame'):
            self.current_frame_idx = self.segment_start_frame
        
        print("Starting playback from frame {} (segment: {}-{})".format(
            self.current_frame_idx, self.segment_start_frame, self.segment_end_frame))
        self.video_playing = True
        self.play_button.config(text="Pause")
        
        # Use Tkinter's after method for non-blocking playback
        self.play_next_frame()
    
    def play_next_frame(self):
        """Play the next frame in the sequence - only plays from Start to End"""
        if not self.video_playing:
            return
        
        # Check if we're at or past the End frame
        if (not hasattr(self, 'segment_end_frame') or 
            self.current_frame_idx >= self.segment_end_frame + 1 or
            self.current_frame_idx >= len(self.current_video_frames)):
            self.stop_video()
            return
        
        # Check if we're before Start frame - jump to Start
        if hasattr(self, 'segment_start_frame') and self.current_frame_idx < self.segment_start_frame:
            self.current_frame_idx = self.segment_start_frame
        
        # Display current frame
        self.display_video_frame(self.current_frame_idx)
        
        # Move to next frame
        self.current_frame_idx += 1
        
        # Schedule next frame (40ms delay = 25 FPS)
        if self.video_playing:
            self.root.after(40, self.play_next_frame)
    
    def stop_video(self):
        """Stop video playback"""
        self.video_playing = False
        self.play_button.config(text="Play")
    
    def on_scene_selected(self, event):
        """Handle scene selection from combo box"""
        scene_name = self.scene_var.get()
        if scene_name in self.scenes:
            scene_idx = self.scenes.index(scene_name)
            self.load_scene(scene_idx)
    
    def previous_scene(self):
        """Go to previous scene"""
        if self.current_scene_idx > 0:
            self.load_scene(self.current_scene_idx - 1)
    
    def next_scene(self):
        """Go to next scene"""
        if self.current_scene_idx < len(self.scenes) - 1:
            self.load_scene(self.current_scene_idx + 1)
    
    def save_current_sample(self):
        """Save the selected segment to Target_samples folder"""
        if self.current_trajectory_df is None:
            messagebox.showwarning("Warning", "No data to save")
            return
        
        if not hasattr(self, 'segment_start_frame') or not hasattr(self, 'segment_end_frame'):
            messagebox.showwarning("Warning", "Segment selection not initialized")
            return
        
        # Get selected segment data
        x_coords, y_coords, segment_df = self.get_selected_segment_data()
        
        if x_coords is None or len(x_coords) == 0:
            messagebox.showwarning("Warning", "No trajectory data in selected segment")
            return
        
        # Get current scene info
        scene_name = self.scenes[self.current_scene_idx]
        
        # Create sample folder with custom segment info
        start_frame_idx = self.segment_start_frame
        end_frame_idx = self.segment_end_frame
        sample_name = "{}_custom_segment_{:03d}_{:03d}".format(scene_name, start_frame_idx, end_frame_idx)
        sample_dir = self.samples_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        
        self.status_var.set("Saving custom segment...")
        self.root.update()
        
        try:
            # Get the actual trajectory frame range for the selected segment
            traj_start_frame = segment_df.iloc[0]['frame'] if len(segment_df) > 0 else 0
            traj_end_frame = segment_df.iloc[-1]['frame'] if len(segment_df) > 0 else 0
            
            # Get the current caption from the text widget (user-provided for this segment)
            current_caption_text = self.get_current_caption()
            
            # Use the caption provided by user, or default if empty
            segment_caption = current_caption_text if current_caption_text.strip() else f"Custom selected segment (frames {start_frame_idx}-{end_frame_idx})"
            
            # 1. Save video clip using frame-exact extraction for the selected segment
            video_output = sample_dir / "{}_video.mp4".format(sample_name)
            self.save_selected_video_frames(str(video_output))
            
            # # 1b. Save current->end video at 25 FPS into all_videos_current_to_end folder
            # data_dir = Path("all_videos_current_to_end")
            # try:
            #     data_dir.mkdir(exist_ok=True)
            # except Exception:
            #     pass
            # current_to_end_output = data_dir / "{}_current_to_end_video.mp4".format(sample_name)
            # try:
            #     self.save_current_to_end_video(str(current_to_end_output), fps=25.0)
            # except Exception as e:
            #     print(f"Warning: Failed to save current->end video: {e}")
            
            # 2. Save first frame image
            first_frame_output = sample_dir / "{}_first_frame.png".format(sample_name)
            self.save_first_frame_image(str(first_frame_output))
            
            # 3. Save current frame image (where progress bar is positioned)
            current_frame_output = sample_dir / "{}_current_frame.png".format(sample_name)
            self.save_current_frame_image(str(current_frame_output))
            
            # 4. Get progress bar state
            progress_state = self.get_progress_bar_state()
            
            # 5. Calculate distances
            distance_info = self.calculate_distance_first_to_current()
            distance_current_to_end = self.calculate_distance_current_to_end()
            
            # 6. Use the already transformed trajectory coordinates from selected segment
            if len(x_coords) > 0:
                trajectory_data = {
                    'x_coordinates': x_coords.tolist(),
                    'y_coordinates': y_coords.tolist(),
                    'start_frame': int(traj_start_frame),
                    'end_frame': int(traj_end_frame),
                    'video_start_idx': int(start_frame_idx),
                    'video_end_idx': int(end_frame_idx),
                    'total_points': int(len(x_coords)),
                    'transformation_applied': True,
                    'heading_aligned_to_y_axis': True
                }
                
                trajectory_file = sample_dir / "{}_trajectory.json".format(sample_name)
                with open(trajectory_file, 'w') as f:
                    json.dump(trajectory_data, f, indent=2)
                
                # Also save as CSV for easy analysis
                trajectory_csv = sample_dir / "{}_trajectory.csv".format(sample_name)
                trajectory_df_save = pd.DataFrame({
                    'x': x_coords,
                    'y': y_coords
                })
                trajectory_df_save.to_csv(trajectory_csv, index=False)
            
            # 7. Save caption and metadata (without annotation-related fields)
            selected_frame_count = end_frame_idx - start_frame_idx + 1
            
            metadata = {
                'scene_name': scene_name,
                'segment_type': 'custom_selected_segment',
                'segment_description': f"Custom segment from video frame {start_frame_idx} to {end_frame_idx}",
                'user_caption': segment_caption,  # User-provided caption for this segment
                'trajectory_start_frame': int(traj_start_frame),
                'trajectory_end_frame': int(traj_end_frame),
                'video_start_idx': int(start_frame_idx),
                'video_end_idx': int(end_frame_idx),
                'frame_duration': int(traj_end_frame - traj_start_frame),
                'video_frame_duration': int(selected_frame_count),
                'total_distance_meters': float(np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))) if len(x_coords) > 1 else 0.0,
                'trajectory_points': int(len(x_coords)),
                'sample_name': sample_name,
                'save_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'extraction_method': 'custom_segment_selection_with_progress_based_transformation',
                'transformation_applied': True,
                'transformation_reference': 'progress_bar_current_frame',
                'heading_aligned_to_y_axis': True,
                'progress_frame_heading_rotated_to_90_degrees': True,
                'includes_first_frame_image': True,
                'includes_current_frame_image': True,
                'progress_bar_state': progress_state,
                'distance_first_to_current': distance_info,
                'distance_current_to_end': distance_current_to_end
            }
            
            # Save caption as text file
            caption_file = sample_dir / "{}_caption.txt".format(sample_name)
            with open(caption_file, 'w') as f:
                f.write(segment_caption)
            
            # Save metadata as JSON
            metadata_file = sample_dir / "{}_metadata.json".format(sample_name)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save trajectory plot
            plot_file = sample_dir / "{}_trajectory_plot.png".format(sample_name)
            self.traj_fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            self.status_var.set("Sample saved: {}".format(sample_name))
            messagebox.showinfo("Success", "Sample saved successfully to:\n{}".format(sample_dir))
            
        except Exception as e:
            messagebox.showerror("Error", "Failed to save sample: {}".format(str(e)))
            self.status_var.set("Error saving sample")


def run_summary(dataset_path: str) -> None:
    """Run the lightweight data summary from the old data-vis script."""
    visualizer = TargetVisualizer(dataset_path)
    scenes = visualizer.list_available_scenes()
    print(f"Available scenes: {len(scenes)}")
    print("First 5 scenes:", scenes[:5])

    if scenes:
        scene_name = scenes[0]
        print(f"\nVisualizing scene: {scene_name}")
        _, trajectory_df = visualizer.load_scene_data(scene_name)
        print(f"Scene has {len(trajectory_df)} trajectory points")


def run_gui(dataset_path: str) -> None:
    """Launch the interactive Tk viewer."""
    root = tk.Tk()

    style = ttk.Style()
    style.theme_use('clam')

    TargetInteractiveViewer(root, dataset_path=dataset_path)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Target dataset visualizer and interactive viewer")
    parser.add_argument(
        "--dataset-path",
        default="Target_trajectories",
        help="Path to the Target_trajectories-style dataset directory",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print dataset summary instead of launching the GUI",
    )
    args = parser.parse_args()

    if args.summary:
        run_summary(args.dataset_path)
    else:
        run_gui(args.dataset_path)


if __name__ == "__main__":
    main()
