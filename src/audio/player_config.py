"""
Player configuration module for VSAT.

This module provides configuration management for the audio player.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PlayerConfig:
    """Manages configuration settings for the audio player."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "volume": 0.7,
        "muted": False,
        "word_padding": 0.05,  # 50ms padding for word playback
        "auto_play_on_load": False,
        "remember_position": True,
        "remember_volume": True,
        "segment_crossfade": 0.01,  # 10ms crossfade for segment transitions
        "playback_rate": 1.0,
        "use_hardware_acceleration": True,
        "buffer_size": 16384,  # Audio buffer size
        "max_recent_files": 10,
        "recent_files": []
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the player configuration.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load configuration if path is provided
        if config_path:
            self.load_config(config_path)
        
        logger.debug("Player configuration initialized")
    
    def load_config(self, config_path: str) -> bool:
        """Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return False
            
            with open(path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update configuration with loaded values
            self.config.update(loaded_config)
            self.config_path = config_path
            
            logger.info(f"Loaded configuration from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save configuration to a file.
        
        Args:
            config_path: Path to the configuration file (optional, uses current path if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            path = Path(config_path or self.config_path)
            if not path:
                logger.warning("No configuration path specified")
                return False
            
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logger.info(f"Saved configuration to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Any: Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        logger.debug(f"Set configuration {key} = {value}")
    
    def add_recent_file(self, file_path: str) -> None:
        """Add a file to the recent files list.
        
        Args:
            file_path: Path to the file
        """
        # Get current recent files
        recent_files = self.get("recent_files", [])
        
        # Remove file if it already exists
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # Add file to the beginning of the list
        recent_files.insert(0, file_path)
        
        # Limit the number of recent files
        max_recent = self.get("max_recent_files", 10)
        recent_files = recent_files[:max_recent]
        
        # Update configuration
        self.set("recent_files", recent_files)
    
    def get_recent_files(self) -> list[str]:
        """Get the list of recent files.
        
        Returns:
            list: List of recent file paths
        """
        return self.get("recent_files", [])
    
    def clear_recent_files(self) -> None:
        """Clear the recent files list."""
        self.set("recent_files", [])
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()
        logger.info("Reset configuration to defaults")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.
        
        Returns:
            dict: All configuration values
        """
        return self.config.copy() 