 # Final Configuration

## Overview

This phase focuses on finalizing your VSAT application configuration for optimal performance and usability. As the sole user of this specialized audio analysis tool, proper configuration is essential to ensure the application meets your specific needs, operates efficiently on your hardware, and provides a seamless experience.

This guide will help you establish optimal default parameters, configure startup behavior, and set resource usage limits tailored to your workflow and system capabilities. By implementing these configurations, you'll ensure that VSAT operates reliably and efficiently for your specific use cases.

## Prerequisites

Before finalizing your VSAT configuration, ensure you have:

- [ ] Completed personal user acceptance testing
- [ ] Implemented critical fixes identified during testing
- [ ] Optimized code and performance for your hardware
- [ ] Implemented error recovery mechanisms
- [ ] Set up your local backup system
- [ ] 2-3 hours of implementation time
- [ ] Detailed knowledge of your system specifications and resource availability

## Implementing Your Final Configuration

### 1. Create a Configuration Manager

First, enhance the existing configuration system with a dedicated manager class:

```python
# src/utils/config_manager.py

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

class ConfigManager:
    """Manages VSAT application configuration with environment-aware settings."""
    
    def __init__(self, config_path: str = "./config/app_config.yaml"):
        """Initialize the configuration manager with path to config file."""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        self.user_overrides = {}
        self._load_user_overrides()
        
    def _load_config(self) -> Dict:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        return yaml.safe_load(f)
                    else:
                        return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading config from {self.config_path}: {str(e)}")
                return self._create_default_config()
        else:
            return self._create_default_config()
            
    def _create_default_config(self) -> Dict:
        """Create default configuration with optimal settings."""
        default_config = {
            "application": {
                "name": "Voice Separation & Analysis Tool",
                "version": "1.0.0",
                "startup": {
                    "load_last_project": True,
                    "check_for_updates": True,
                    "show_welcome_screen": False,
                    "verify_models": True,
                    "preload_models": True
                },
                "shutdown": {
                    "save_session": True,
                    "backup_on_exit": True,
                    "cleanup_temp_files": True
                },
                "logging": {
                    "level": "INFO",
                    "file_rotation": "daily",
                    "max_log_files": 30,
                    "log_to_console": False
                }
            },
            "resources": {
                "memory": {
                    "max_usage_percent": 70,
                    "warning_threshold_percent": 85,
                    "model_unload_threshold_percent": 90
                },
                "cpu": {
                    "max_threads": "auto",  # "auto" or specific number
                    "priority": "normal",   # low, normal, high
                    "affinity": "all"       # "all" or specific cores e.g. "0,1,2,3"
                },
                "gpu": {
                    "enabled": True,
                    "device_id": 0,
                    "memory_limit_mb": 0,   # 0 means no limit
                    "precision": "mixed"    # float32, float16, mixed
                },
                "disk": {
                    "temp_dir": "./temp",
                    "min_free_space_gb": 5,
                    "cleanup_threshold_gb": 2
                }
            },
            "audio": {
                "processing": {
                    "sample_rate": 16000,
                    "chunk_size_seconds": 30,
                    "overlap_seconds": 1.5,
                    "default_format": "wav"
                },
                "playback": {
                    "device": "default",
                    "buffer_size": 1024,
                    "volume": 1.0
                }
            },
            "ml_models": {
                "diarization": {
                    "model": "pyannote/speaker-diarization-3.0",
                    "threshold": 0.7,
                    "min_speakers": 1,
                    "max_speakers": 6
                },
                "transcription": {
                    "model": "large-v3",
                    "language": "en",
                    "beam_size": 5,
                    "word_timestamps": True
                },
                "separation": {
                    "model": "Conv-TasNet",
                    "iterations": 1,
                    "quality_preset": "high"
                }
            },
            "ui": {
                "theme": "system",  # system, light, dark
                "font_size": 10,
                "waveform_colors": [
                    "#1f77b4", "#ff7f0e", "#2ca02c", 
                    "#d62728", "#9467bd", "#8c564b"
                ],
                "timeline_zoom_level": 5,
                "show_confidence_scores": True,
                "auto_scroll": True
            },
            "paths": {
                "projects_dir": "./projects",
                "exports_dir": "./exports",
                "models_dir": "./models",
                "plugins_dir": "./plugins"
            }
        }
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Save default configuration
        self._save_config(default_config)
            
        return default_config
    
    def _save_config(self, config: Dict) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=2)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config to {self.config_path}: {str(e)}")
    
    def _load_user_overrides(self) -> None:
        """Load user-specific configuration overrides."""
        user_config_path = os.path.join(
            os.path.dirname(self.config_path),
            "user_config.yaml"
        )
        if os.path.exists(user_config_path):
            try:
                with open(user_config_path, 'r') as f:
                    if user_config_path.endswith('.yaml') or user_config_path.endswith('.yml'):
                        self.user_overrides = yaml.safe_load(f) or {}
                    else:
                        self.user_overrides = json.load(f)
                self.logger.info(f"Loaded user configuration overrides from {user_config_path}")
            except Exception as e:
                self.logger.error(f"Error loading user config from {user_config_path}: {str(e)}")
                self.user_overrides = {}
```

### 2. Implement Configuration Access and Override Methods

Add methods to access and override configuration values:

```python
def get(self, key_path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation path.
    
    Args:
        key_path: Dot notation path to configuration value (e.g., "audio.processing.sample_rate")
        default: Default value to return if key doesn't exist
        
    Returns:
        Configuration value or default if not found
    """
    # First check user overrides
    value = self._get_nested_value(self.user_overrides, key_path.split('.'))
    if value is not None:
        return value
        
    # Then check main config
    value = self._get_nested_value(self.config, key_path.split('.'))
    if value is not None:
        return value
        
    return default
    
def _get_nested_value(self, config_dict: Dict, key_parts: List[str]) -> Any:
    """Retrieve a nested value from a dictionary using a list of keys."""
    if not config_dict or not key_parts:
        return None
        
    current = config_dict
    for part in key_parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
            
    return current
    
def set(self, key_path: str, value: Any, save_override: bool = True) -> bool:
    """Set a configuration value using dot notation path.
    
    Args:
        key_path: Dot notation path to configuration value
        value: Value to set
        save_override: Whether to save as a user override
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Update in-memory configuration
        self._set_nested_value(self.config, key_path.split('.'), value)
        
        # If requested, save as user override
        if save_override:
            self._set_nested_value(self.user_overrides, key_path.split('.'), value)
            self._save_user_overrides()
            
        return True
    except Exception as e:
        self.logger.error(f"Error setting config value {key_path}: {str(e)}")
        return False
        
def _set_nested_value(self, config_dict: Dict, key_parts: List[str], value: Any) -> None:
    """Set a nested value in a dictionary using a list of keys."""
    if not config_dict or not key_parts:
        return
        
    current = config_dict
    for i, part in enumerate(key_parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]
        
    current[key_parts[-1]] = value
    
def _save_user_overrides(self) -> None:
    """Save user configuration overrides to file."""
    user_config_path = os.path.join(
        os.path.dirname(self.config_path),
        "user_config.yaml"
    )
    try:
        os.makedirs(os.path.dirname(user_config_path), exist_ok=True)
        with open(user_config_path, 'w') as f:
            yaml.dump(self.user_overrides, f, default_flow_style=False)
        self.logger.info(f"User configuration saved to {user_config_path}")
    except Exception as e:
        self.logger.error(f"Error saving user config to {user_config_path}: {str(e)}")
```

### 3. Add System-Aware Configuration Methods

Implement methods to detect system capabilities and optimize configuration accordingly:

```python
def optimize_for_system(self) -> None:
    """Detect system capabilities and optimize configuration accordingly."""
    self.logger.info("Optimizing configuration for current system...")
    
    # Optimize CPU settings
    self._optimize_cpu_settings()
    
    # Optimize GPU settings
    self._optimize_gpu_settings()
    
    # Optimize memory settings
    self._optimize_memory_settings()
    
    # Optimize disk settings
    self._optimize_disk_settings()
    
    # Save optimized configuration
    self._save_user_overrides()
    self.logger.info("Configuration optimized for current system")
    
def _optimize_cpu_settings(self) -> None:
    """Optimize CPU-related settings based on system capabilities."""
    import multiprocessing
    import psutil
    
    # Get number of CPU cores
    cpu_count = multiprocessing.cpu_count()
    
    # Set optimal thread count (leave 1-2 cores for system)
    optimal_threads = max(1, cpu_count - 1)
    self.set("resources.cpu.max_threads", optimal_threads, save_override=False)
    
    # Set CPU affinity based on available cores
    # For simplicity, we'll use all cores but could be more sophisticated
    self.set("resources.cpu.affinity", "all", save_override=False)
    
    self.logger.info(f"CPU settings optimized for {cpu_count} cores")
    
def _optimize_gpu_settings(self) -> None:
    """Optimize GPU-related settings based on system capabilities."""
    try:
        import torch
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_enabled = True
            device_id = 0  # Default to first GPU
            
            # Get GPU memory (in MB)
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 * 1024)
            
            # Set memory limit to 90% of available memory
            memory_limit = int(gpu_memory * 0.9)
            
            # Determine optimal precision based on GPU capabilities
            precision = "mixed"  # Default to mixed precision
            if torch.cuda.get_device_capability(device_id)[0] >= 7:
                # Volta or newer architecture supports efficient mixed precision
                precision = "mixed"
            else:
                # Older architectures may perform better with fp32
                precision = "float32"
                
            self.set("resources.gpu.enabled", gpu_enabled, save_override=False)
            self.set("resources.gpu.device_id", device_id, save_override=False)
            self.set("resources.gpu.memory_limit_mb", memory_limit, save_override=False)
            self.set("resources.gpu.precision", precision, save_override=False)
            
            self.logger.info(f"GPU settings optimized for {gpu_count} GPUs with {memory_limit}MB memory limit")
        else:
            self.set("resources.gpu.enabled", False, save_override=False)
            self.logger.info("No GPU detected, disabled GPU acceleration")
    except ImportError:
        self.set("resources.gpu.enabled", False, save_override=False)
        self.logger.info("PyTorch not available, disabled GPU acceleration")
    except Exception as e:
        self.logger.error(f"Error optimizing GPU settings: {str(e)}")
        self.set("resources.gpu.enabled", False, save_override=False)
```

### 4. Implement Resource Monitoring and Management

Add methods to monitor and manage system resources:

```python
def _optimize_memory_settings(self) -> None:
    """Optimize memory-related settings based on system capabilities."""
    import psutil
    
    # Get total system memory in GB
    total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    
    # Set memory thresholds based on available memory
    if total_memory_gb >= 32:
        # High memory system
        max_usage = 80
        warning = 90
        unload = 95
    elif total_memory_gb >= 16:
        # Medium memory system
        max_usage = 70
        warning = 85
        unload = 90
    else:
        # Low memory system
        max_usage = 60
        warning = 75
        unload = 85
        
    self.set("resources.memory.max_usage_percent", max_usage, save_override=False)
    self.set("resources.memory.warning_threshold_percent", warning, save_override=False)
    self.set("resources.memory.model_unload_threshold_percent", unload, save_override=False)
    
    self.logger.info(f"Memory settings optimized for system with {total_memory_gb:.1f}GB RAM")
    
def _optimize_disk_settings(self) -> None:
    """Optimize disk-related settings based on system capabilities."""
    import psutil
    
    # Get disk information for the drive containing the application
    app_path = os.path.abspath(os.path.dirname(self.config_path))
    drive = psutil.disk_usage(app_path)
    
    # Total disk space in GB
    total_space_gb = drive.total / (1024 * 1024 * 1024)
    
    # Set minimum free space requirement based on disk size
    if total_space_gb >= 1000:
        # Large disk (1TB+)
        min_free = 20
        cleanup = 10
    elif total_space_gb >= 500:
        # Medium disk
        min_free = 10
        cleanup = 5
    else:
        # Small disk
        min_free = 5
        cleanup = 2
        
    self.set("resources.disk.min_free_space_gb", min_free, save_override=False)
    self.set("resources.disk.cleanup_threshold_gb", cleanup, save_override=False)
    
    # Set optimal temp directory
    temp_dir = os.path.join(app_path, "temp")
    self.set("resources.disk.temp_dir", temp_dir, save_override=False)
    
    self.logger.info(f"Disk settings optimized for system with {total_space_gb:.1f}GB storage")
```

### 5. Create a Configuration UI

Implement a dedicated configuration UI to manage all settings:

```python
# src/ui/config_dialog.py

from PyQt6.QtWidgets import (QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, 
                            QComboBox, QLineEdit, QPushButton, QFileDialog,
                            QGroupBox, QFormLayout, QDialogButtonBox)
from PyQt6.QtCore import Qt, pyqtSignal

class ConfigDialog(QDialog):
    """Dialog for configuring VSAT application settings."""
    
    config_changed = pyqtSignal()
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("VSAT Configuration")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.create_general_tab()
        self.create_resources_tab()
        self.create_audio_tab()
        self.create_ml_models_tab()
        self.create_ui_tab()
        self.create_paths_tab()
        
        layout.addWidget(self.tab_widget)
        
        # Add optimize button
        optimize_button = QPushButton("Optimize for Current System")
        optimize_button.clicked.connect(self.optimize_for_system)
        layout.addWidget(optimize_button)
        
        # Add dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                     QDialogButtonBox.StandardButton.Cancel | 
                                     QDialogButtonBox.StandardButton.Apply | 
                                     QDialogButtonBox.StandardButton.Reset)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_settings)
        button_box.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self.reset_settings)
        layout.addWidget(button_box)
        
    def create_general_tab(self):
        """Create the general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Startup group
        startup_group = QGroupBox("Startup")
        startup_layout = QFormLayout(startup_group)
        
        # Load last project
        self.load_last_project = QCheckBox()
        self.load_last_project.setChecked(self.config_manager.get("application.startup.load_last_project", True))
        startup_layout.addRow("Load last project:", self.load_last_project)
        
        # Check for updates
        self.check_for_updates = QCheckBox()
        self.check_for_updates.setChecked(self.config_manager.get("application.startup.check_for_updates", True))
        startup_layout.addRow("Check for updates:", self.check_for_updates)
        
        # Show welcome screen
        self.show_welcome = QCheckBox()
        self.show_welcome.setChecked(self.config_manager.get("application.startup.show_welcome_screen", False))
        startup_layout.addRow("Show welcome screen:", self.show_welcome)
        
        # Verify models
        self.verify_models = QCheckBox()
        self.verify_models.setChecked(self.config_manager.get("application.startup.verify_models", True))
        startup_layout.addRow("Verify ML models:", self.verify_models)
        
        # Preload models
        self.preload_models = QCheckBox()
        self.preload_models.setChecked(self.config_manager.get("application.startup.preload_models", True))
        startup_layout.addRow("Preload ML models:", self.preload_models)
        
        layout.addWidget(startup_group)
        
        # Shutdown group
        shutdown_group = QGroupBox("Shutdown")
        shutdown_layout = QFormLayout(shutdown_group)
        
        # Save session
        self.save_session = QCheckBox()
        self.save_session.setChecked(self.config_manager.get("application.shutdown.save_session", True))
        shutdown_layout.addRow("Save session:", self.save_session)
        
        # Backup on exit
        self.backup_on_exit = QCheckBox()
        self.backup_on_exit.setChecked(self.config_manager.get("application.shutdown.backup_on_exit", True))
        shutdown_layout.addRow("Backup on exit:", self.backup_on_exit)
        
        # Cleanup temp files
        self.cleanup_temp = QCheckBox()
        self.cleanup_temp.setChecked(self.config_manager.get("application.shutdown.cleanup_temp_files", True))
        shutdown_layout.addRow("Cleanup temporary files:", self.cleanup_temp)
        
        layout.addWidget(shutdown_group)
        
        # Logging group
        logging_group = QGroupBox("Logging")
        logging_layout = QFormLayout(logging_group)
        
        # Log level
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        current_level = self.config_manager.get("application.logging.level", "INFO")
        self.log_level.setCurrentText(current_level)
        logging_layout.addRow("Log level:", self.log_level)
        
        # Log rotation
        self.log_rotation = QComboBox()
        self.log_rotation.addItems(["daily", "weekly", "monthly", "never"])
        current_rotation = self.config_manager.get("application.logging.file_rotation", "daily")
        self.log_rotation.setCurrentText(current_rotation)
        logging_layout.addRow("Log rotation:", self.log_rotation)
        
        # Max log files
        self.max_log_files = QSpinBox()
        self.max_log_files.setRange(1, 365)
        self.max_log_files.setValue(self.config_manager.get("application.logging.max_log_files", 30))
        logging_layout.addRow("Maximum log files:", self.max_log_files)
        
        # Log to console
        self.log_to_console = QCheckBox()
        self.log_to_console.setChecked(self.config_manager.get("application.logging.log_to_console", False))
        logging_layout.addRow("Log to console:", self.log_to_console)
        
        layout.addWidget(logging_group)
        
        # Add tab to widget
        self.tab_widget.addTab(tab, "General")
```

### 6. Integrate Configuration Manager with Application

Update the main application to use the configuration manager:

```python
# src/app.py

from src.utils.config_manager import ConfigManager

class VSATApplication:
    """Main VSAT application class."""
    
    def __init__(self):
        """Initialize the VSAT application."""
        # Initialize configuration manager
        self.config_manager = ConfigManager()
        
        # Apply system-specific optimizations if this is the first run
        if self.is_first_run():
            self.config_manager.optimize_for_system()
        
        # Initialize other components with configuration
        self.initialize_components()
        
        # Set up resource monitoring
        self.setup_resource_monitoring()
        
    def is_first_run(self) -> bool:
        """Check if this is the first application run."""
        user_config_path = os.path.join(
            os.path.dirname(self.config_manager.config_path),
            "user_config.yaml"
        )
        return not os.path.exists(user_config_path)
        
    def initialize_components(self):
        """Initialize application components with configuration."""
        # Set up logging
        self.setup_logging()
        
        # Initialize UI with configuration
        self.setup_ui()
        
        # Initialize audio processing with configuration
        self.setup_audio_processing()
        
        # Initialize ML models with configuration
        self.setup_ml_models()
        
        # Initialize database with configuration
        self.setup_database()
        
    def setup_logging(self):
        """Set up application logging based on configuration."""
        log_level = self.config_manager.get("application.logging.level", "INFO")
        log_to_console = self.config_manager.get("application.logging.log_to_console", False)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("vsat.log"),
                logging.StreamHandler() if log_to_console else logging.NullHandler()
            ]
        )
        
    def setup_resource_monitoring(self):
        """Set up system resource monitoring."""
        # Get configuration values
        memory_warning = self.config_manager.get("resources.memory.warning_threshold_percent", 85)
        memory_unload = self.config_manager.get("resources.memory.model_unload_threshold_percent", 90)
        min_disk_space = self.config_manager.get("resources.disk.min_free_space_gb", 5)
        
        # Set up periodic resource checks
        # Implementation depends on your application architecture
        # This could be a timer, a background thread, etc.
```

## Finalizing Your Configuration

### 1. Determine Optimal Default Parameters

- [ ] Run the application with different parameter combinations to find the optimal settings for your workflow
- [ ] Test different ML model configurations to balance accuracy and performance
- [ ] Determine the ideal UI settings for your display and preferences
- [ ] Measure resource usage under various conditions to set appropriate limits

### 2. Configure Startup Behavior

- [ ] Decide which components should be preloaded at startup
- [ ] Configure automatic model verification and updates
- [ ] Set up session restoration options
- [ ] Configure startup checks for system resources

### 3. Establish Resource Usage Limits

- [ ] Set memory usage thresholds based on your system capabilities
- [ ] Configure CPU and GPU utilization limits
- [ ] Establish disk space requirements and cleanup policies
- [ ] Set up resource monitoring and alerts

### 4. Create User-Specific Configuration

- [ ] Create a personal configuration file with your preferred settings
- [ ] Document any custom configuration in your personal documentation
- [ ] Test the application with your custom configuration
- [ ] Create configuration backups

## Testing Your Configuration

- [ ] Verify that all configuration settings are properly applied
- [ ] Test application behavior under different resource conditions
- [ ] Verify startup and shutdown behavior
- [ ] Test configuration persistence across application restarts
- [ ] Verify that resource limits are properly enforced

## Next Steps

After finalizing your VSAT configuration, you should:

1. Proceed to [Desktop Integration](14_desktop_integration.md) to integrate VSAT with your desktop environment
2. Consider creating configuration presets for different use cases
3. Document your configuration decisions in your personal documentation

By completing this phase, you've ensured that VSAT is optimally configured for your specific needs and system capabilities, providing a solid foundation for efficient and reliable operation.

 