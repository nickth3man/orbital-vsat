 # Local Backup System

## Overview

This phase focuses on implementing a reliable local backup system for your VSAT application data. As the sole user of this specialized audio analysis tool, having a robust backup strategy is essential to protect your valuable audio recordings, transcriptions, speaker profiles, and analysis results from accidental loss or corruption.

This guide will help you set up automated, versioned backups tailored to your specific workflow and data importance. By implementing proper backup procedures, you'll ensure that your work is protected while maintaining efficient storage usage.

## Prerequisites

Before implementing your local backup system, ensure you have:

- [ ] Completed data management strategy implementation
- [ ] Identified critical data that requires backup
- [ ] 3-4 hours of implementation time
- [ ] External storage device or network location for off-site backups
- [ ] At least 2-3x the storage space of your current VSAT data

## Implementing Your Backup System

### 1. Create a Backup Manager

First, create a dedicated backup manager class to handle all backup operations:

```python
# src/utils/backup_manager.py

import os
import json
import shutil
import datetime
import zipfile
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class BackupManager:
    """Manages automated backups of VSAT data."""
    
    def __init__(self, config_path: str = "./config/backup_config.json"):
        """Initialize the backup manager with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict:
        """Load backup configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON in {self.config_path}, using defaults")
                return self._create_default_config()
        else:
            return self._create_default_config()
            
    def _create_default_config(self) -> Dict:
        """Create default backup configuration."""
        default_config = {
            "backup_enabled": True,
            "backup_locations": {
                "local": os.path.join(os.path.expanduser("~"), "VSAT_Backups"),
                "external": "",  # To be filled by user
                "network": ""    # To be filled by user
            },
            "backup_schedule": {
                "frequency": "daily",  # Options: hourly, daily, weekly
                "retention": {
                    "daily": 7,    # Keep 7 daily backups
                    "weekly": 4,   # Keep 4 weekly backups
                    "monthly": 3   # Keep 3 monthly backups
                },
                "time": "02:00"    # 2 AM
            },
            "backup_items": {
                "database": True,
                "audio_files": True,
                "transcripts": True,
                "speaker_profiles": True,
                "user_settings": True,
                "analysis_results": True
            },
            "compression_level": 6,  # 0-9, where 9 is maximum compression
            "verify_backups": True,
            "notification_email": ""  # To be filled by user
        }
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Save default configuration
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        return default_config
        
    def create_backup(self, backup_type: str = "full") -> Optional[str]:
        """Create a backup of the specified type.
        
        Args:
            backup_type: Type of backup to create ("full", "incremental", or "differential")
            
        Returns:
            Path to created backup file or None if backup failed
        """
        if not self.config["backup_enabled"]:
            self.logger.warning("Backup is disabled in configuration")
            return None
            
        # Create timestamp for backup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"vsat_backup_{backup_type}_{timestamp}"
        
        # Determine backup location
        backup_dir = self._get_backup_location()
        if not backup_dir:
            self.logger.error("No valid backup location available")
            return None
            
        # Ensure backup directory exists
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup file path
        backup_file = os.path.join(backup_dir, f"{backup_name}.zip")
        
        try:
            # Collect files to backup
            files_to_backup = self._collect_backup_files(backup_type)
            
            # Create backup archive
            self._create_backup_archive(backup_file, files_to_backup)
            
            # Verify backup integrity
            if self.config["verify_backups"] and not self._verify_backup(backup_file, files_to_backup):
                self.logger.error(f"Backup verification failed for {backup_file}")
                return None
                
            self.logger.info(f"Successfully created {backup_type} backup at {backup_file}")
            
            # Apply retention policy
            self._apply_retention_policy()
            
            return backup_file
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {str(e)}")
            return None
            
    def _get_backup_location(self) -> Optional[str]:
        """Get the first available backup location from configuration."""
        locations = self.config["backup_locations"]
        
        # Try locations in order of preference: external, network, local
        for location_type in ["external", "network", "local"]:
            location = locations.get(location_type)
            if location and os.path.exists(location):
                return location
            elif location and location_type == "local":
                # Create local directory if it doesn't exist
                os.makedirs(location, exist_ok=True)
                return location
                
        return None
```

### 2. Implement File Collection and Archiving

Next, add methods to collect files for backup and create the archive:

```python
def _collect_backup_files(self, backup_type: str) -> Dict[str, List[str]]:
    """Collect files to be included in the backup.
    
    Args:
        backup_type: Type of backup to create
    
    Returns:
        Dictionary mapping categories to lists of file paths
    """
    backup_items = self.config["backup_items"]
    files_by_category = {}
    
    # Database files
    if backup_items["database"]:
        db_files = self._get_database_files()
        if db_files:
            files_by_category["database"] = db_files
            
    # Audio files
    if backup_items["audio_files"]:
        audio_files = self._get_audio_files(backup_type)
        if audio_files:
            files_by_category["audio"] = audio_files
            
    # Transcripts
    if backup_items["transcripts"]:
        transcript_files = self._get_transcript_files()
        if transcript_files:
            files_by_category["transcripts"] = transcript_files
            
    # Speaker profiles
    if backup_items["speaker_profiles"]:
        profile_files = self._get_speaker_profile_files()
        if profile_files:
            files_by_category["profiles"] = profile_files
            
    # User settings
    if backup_items["user_settings"]:
        setting_files = self._get_setting_files()
        if setting_files:
            files_by_category["settings"] = setting_files
            
    # Analysis results
    if backup_items["analysis_results"]:
        analysis_files = self._get_analysis_files()
        if analysis_files:
            files_by_category["analysis"] = analysis_files
            
    return files_by_category
    
def _create_backup_archive(self, backup_file: str, files_by_category: Dict[str, List[str]]) -> None:
    """Create a ZIP archive containing all backup files.
    
    Args:
        backup_file: Path to the output ZIP file
        files_by_category: Dictionary mapping categories to lists of file paths
    """
    # Create a manifest of all files
    manifest = {
        "backup_date": datetime.datetime.now().isoformat(),
        "backup_version": "1.0",
        "files": {}
    }
    
    # Create ZIP file
    with zipfile.ZipFile(backup_file, 'w', compression=zipfile.ZIP_DEFLATED, 
                         compresslevel=self.config["compression_level"]) as zipf:
        
        # Add files by category
        for category, file_list in files_by_category.items():
            manifest["files"][category] = []
            
            for file_path in file_list:
                if os.path.exists(file_path):
                    # Calculate file hash for verification
                    file_hash = self._calculate_file_hash(file_path)
                    
                    # Add file to ZIP
                    arcname = os.path.join(category, os.path.basename(file_path))
                    zipf.write(file_path, arcname=arcname)
                    
                    # Add to manifest
                    manifest["files"][category].append({
                        "original_path": file_path,
                        "archive_path": arcname,
                        "hash": file_hash,
                        "size": os.path.getsize(file_path)
                    })
        
        # Add manifest to ZIP
        zipf.writestr("manifest.json", json.dumps(manifest, indent=2))
```

### 3. Set Up Automated Backup Scheduling

Implement a scheduler to run backups automatically:

```python
# src/utils/backup_scheduler.py

import os
import time
import schedule
import threading
import logging
from datetime import datetime
from typing import Optional

from src.utils.backup_manager import BackupManager

class BackupScheduler:
    """Schedules and manages automated backups."""
    
    def __init__(self):
        """Initialize the backup scheduler."""
        self.backup_manager = BackupManager()
        self.logger = logging.getLogger(__name__)
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
    def start(self) -> None:
        """Start the backup scheduler in a background thread."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.logger.warning("Backup scheduler is already running")
            return
            
        # Clear any existing scheduled jobs
        schedule.clear()
        
        # Set up scheduled backups based on configuration
        self._setup_scheduled_backups()
        
        # Create and start the scheduler thread
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Backup scheduler started")
        
    def stop(self) -> None:
        """Stop the backup scheduler."""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            self.logger.warning("Backup scheduler is not running")
            return
            
        # Signal the scheduler thread to stop
        self.stop_event.set()
        
        # Wait for the thread to terminate
        self.scheduler_thread.join(timeout=5.0)
        
        # Clear all scheduled jobs
        schedule.clear()
        
        self.logger.info("Backup scheduler stopped")
        
    def _setup_scheduled_backups(self) -> None:
        """Set up scheduled backups based on configuration."""
        config = self.backup_manager.config
        
        if not config["backup_enabled"]:
            self.logger.info("Automated backups are disabled in configuration")
            return
            
        # Get backup schedule configuration
        backup_schedule = config["backup_schedule"]
        frequency = backup_schedule["frequency"]
        backup_time = backup_schedule.get("time", "02:00")
        
        # Schedule backups based on frequency
        if frequency == "hourly":
            schedule.every().hour.at(":00").do(self._run_backup, "incremental")
            schedule.every().day.at(backup_time).do(self._run_backup, "full")
            
        elif frequency == "daily":
            schedule.every().day.at(backup_time).do(self._run_backup, "full")
            
        elif frequency == "weekly":
            schedule.every().monday.at(backup_time).do(self._run_backup, "full")
            
        self.logger.info(f"Scheduled {frequency} backups")
        
    def _run_scheduler(self) -> None:
        """Run the scheduler loop in a background thread."""
        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    def _run_backup(self, backup_type: str) -> None:
        """Run a backup job of the specified type."""
        try:
            self.logger.info(f"Starting scheduled {backup_type} backup")
            backup_file = self.backup_manager.create_backup(backup_type)
            
            if backup_file:
                self.logger.info(f"Scheduled {backup_type} backup completed successfully: {backup_file}")
            else:
                self.logger.error(f"Scheduled {backup_type} backup failed")
                
        except Exception as e:
            self.logger.error(f"Error during scheduled backup: {str(e)}")
```

### 4. Implement Backup Restoration

Create functionality to restore from backups:

```python
# Add to BackupManager class

def restore_from_backup(self, backup_file: str, restore_path: Optional[str] = None) -> bool:
    """Restore data from a backup file.
    
    Args:
        backup_file: Path to the backup file to restore from
        restore_path: Optional path to restore to (if None, restore to original locations)
        
    Returns:
        True if restoration was successful, False otherwise
    """
    if not os.path.exists(backup_file):
        self.logger.error(f"Backup file not found: {backup_file}")
        return False
        
    try:
        # Create temporary directory for extraction
        temp_dir = os.path.join(os.path.dirname(backup_file), "temp_restore")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract backup to temporary directory
        with zipfile.ZipFile(backup_file, 'r') as zipf:
            zipf.extractall(temp_dir)
            
        # Load manifest
        manifest_path = os.path.join(temp_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            self.logger.error(f"Manifest not found in backup: {backup_file}")
            shutil.rmtree(temp_dir)
            return False
            
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        # Restore files
        restored_files = 0
        failed_files = 0
        
        for category, file_list in manifest["files"].items():
            for file_info in file_list:
                source_path = os.path.join(temp_dir, file_info["archive_path"])
                
                if restore_path:
                    # Restore to custom location
                    target_dir = os.path.join(restore_path, category)
                    os.makedirs(target_dir, exist_ok=True)
                    target_path = os.path.join(target_dir, os.path.basename(file_info["original_path"]))
                else:
                    # Restore to original location
                    target_path = file_info["original_path"]
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                try:
                    shutil.copy2(source_path, target_path)
                    restored_files += 1
                except Exception as e:
                    self.logger.error(f"Failed to restore {source_path}: {str(e)}")
                    failed_files += 1
                    
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        self.logger.info(f"Restore completed: {restored_files} files restored, {failed_files} files failed")
        
        return failed_files == 0
        
    except Exception as e:
        self.logger.error(f"Restoration failed: {str(e)}")
        return False
```

## Testing Your Backup System

### 1. Create Unit Tests

Create comprehensive tests for your backup system:

```python
# tests/utils/test_backup_manager.py

import os
import json
import shutil
import unittest
import tempfile
from unittest.mock import patch, MagicMock

from src.utils.backup_manager import BackupManager

class TestBackupManager(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test configuration file
        self.config_path = os.path.join(self.test_dir, "test_backup_config.json")
        
        # Create test data
        self.test_data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create some test files
        self.test_files = []
        for category in ["database", "audio", "transcripts"]:
            category_dir = os.path.join(self.test_data_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            for i in range(3):
                file_path = os.path.join(category_dir, f"test_{category}_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"Test content for {category} file {i}")
                self.test_files.append(file_path)
        
        # Initialize backup manager with test configuration
        self.backup_manager = BackupManager(config_path=self.config_path)
        
        # Update configuration for testing
        self.backup_manager.config["backup_locations"]["local"] = os.path.join(self.test_dir, "backups")
        self.backup_manager.config["verify_backups"] = True
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_create_backup(self):
        # Mock the file collection methods
        with patch.object(self.backup_manager, '_collect_backup_files') as mock_collect:
            # Set up mock to return our test files
            files_by_category = {
                "database": [f for f in self.test_files if "database" in f],
                "audio": [f for f in self.test_files if "audio" in f],
                "transcripts": [f for f in self.test_files if "transcripts" in f]
            }
            mock_collect.return_value = files_by_category
            
            # Create a backup
            backup_file = self.backup_manager.create_backup("full")
            
            # Verify backup was created
            self.assertIsNotNone(backup_file)
            self.assertTrue(os.path.exists(backup_file))
            
            # Verify backup contains expected files
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                # Check manifest exists
                self.assertIn("manifest.json", zipf.namelist())
                
                # Load manifest
                with zipf.open("manifest.json") as f:
                    manifest = json.loads(f.read().decode('utf-8'))
                
                # Check all categories are present
                self.assertIn("database", manifest["files"])
                self.assertIn("audio", manifest["files"])
                self.assertIn("transcripts", manifest["files"])
                
                # Check file count
                total_files = sum(len(files) for files in manifest["files"].values())
                self.assertEqual(total_files, len(self.test_files))
    
    def test_restore_from_backup(self):
        # First create a backup
        with patch.object(self.backup_manager, '_collect_backup_files') as mock_collect:
            files_by_category = {
                "database": [f for f in self.test_files if "database" in f],
                "audio": [f for f in self.test_files if "audio" in f],
                "transcripts": [f for f in self.test_files if "transcripts" in f]
            }
            mock_collect.return_value = files_by_category
            
            backup_file = self.backup_manager.create_backup("full")
            
        # Create a restore directory
        restore_dir = os.path.join(self.test_dir, "restore")
        
        # Delete original files to ensure restoration works
        for file_path in self.test_files:
            os.remove(file_path)
            
        # Restore from backup
        result = self.backup_manager.restore_from_backup(backup_file, restore_dir)
        
        # Verify restoration was successful
        self.assertTrue(result)
        
        # Check that all files were restored
        for category in ["database", "audio", "transcripts"]:
            category_dir = os.path.join(restore_dir, category)
            self.assertTrue(os.path.exists(category_dir))
            
            for i in range(3):
                file_path = os.path.join(category_dir, f"test_{category}_{i}.txt")
                self.assertTrue(os.path.exists(file_path))
                
                with open(file_path, 'r') as f:
                    content = f.read()
                    self.assertEqual(content, f"Test content for {category} file {i}")
```

### 2. Manual Testing Checklist

- [ ] Configure backup locations (local, external, and network)
- [ ] Run a manual full backup and verify the backup file is created
- [ ] Examine the backup archive to ensure all expected files are included
- [ ] Test restoration to a different location and verify file integrity
- [ ] Configure and test automated backup scheduling
- [ ] Verify retention policies are correctly applied
- [ ] Test backup verification by intentionally corrupting a backup file
- [ ] Test recovery from failed backups

## Integration with VSAT

### 1. Add Backup UI to Settings

Integrate backup functionality into the VSAT settings dialog:

```python
# src/ui/settings_dialog.py

# Add a backup tab to the settings dialog
def _create_backup_tab(self):
    """Create the backup settings tab."""
    backup_tab = QWidget()
    layout = QVBoxLayout(backup_tab)
    
    # Enable backups checkbox
    self.enable_backups_cb = QCheckBox("Enable automated backups")
    self.enable_backups_cb.setChecked(self.backup_manager.config["backup_enabled"])
    layout.addWidget(self.enable_backups_cb)
    
    # Backup locations group
    locations_group = QGroupBox("Backup Locations")
    locations_layout = QFormLayout(locations_group)
    
    # Local backup location
    self.local_backup_path = QLineEdit()
    self.local_backup_path.setText(self.backup_manager.config["backup_locations"]["local"])
    local_browse_btn = QPushButton("Browse...")
    local_browse_btn.clicked.connect(lambda: self._browse_backup_location("local"))
    
    local_layout = QHBoxLayout()
    local_layout.addWidget(self.local_backup_path)
    local_layout.addWidget(local_browse_btn)
    locations_layout.addRow("Local:", local_layout)
    
    # External backup location
    self.external_backup_path = QLineEdit()
    self.external_backup_path.setText(self.backup_manager.config["backup_locations"]["external"])
    external_browse_btn = QPushButton("Browse...")
    external_browse_btn.clicked.connect(lambda: self._browse_backup_location("external"))
    
    external_layout = QHBoxLayout()
    external_layout.addWidget(self.external_backup_path)
    external_layout.addWidget(external_browse_btn)
    locations_layout.addRow("External:", external_layout)
    
    # Network backup location
    self.network_backup_path = QLineEdit()
    self.network_backup_path.setText(self.backup_manager.config["backup_locations"]["network"])
    network_browse_btn = QPushButton("Browse...")
    network_browse_btn.clicked.connect(lambda: self._browse_backup_location("network"))
    
    network_layout = QHBoxLayout()
    network_layout.addWidget(self.network_backup_path)
    network_layout.addWidget(network_browse_btn)
    locations_layout.addRow("Network:", network_layout)
    
    layout.addWidget(locations_group)
    
    # Backup schedule group
    schedule_group = QGroupBox("Backup Schedule")
    schedule_layout = QFormLayout(schedule_group)
    
    # Frequency combo box
    self.backup_frequency = QComboBox()
    self.backup_frequency.addItems(["Hourly", "Daily", "Weekly"])
    current_frequency = self.backup_manager.config["backup_schedule"]["frequency"]
    self.backup_frequency.setCurrentText(current_frequency.capitalize())
    schedule_layout.addRow("Frequency:", self.backup_frequency)
    
    # Backup time
    self.backup_time = QTimeEdit()
    current_time = self.backup_manager.config["backup_schedule"]["time"]
    self.backup_time.setTime(QTime.fromString(current_time, "hh:mm"))
    schedule_layout.addRow("Time:", self.backup_time)
    
    layout.addWidget(schedule_group)
    
    # Backup content group
    content_group = QGroupBox("Backup Content")
    content_layout = QVBoxLayout(content_group)
    
    # Checkboxes for backup items
    self.backup_items = {}
    for item, enabled in self.backup_manager.config["backup_items"].items():
        checkbox = QCheckBox(item.replace("_", " ").title())
        checkbox.setChecked(enabled)
        self.backup_items[item] = checkbox
        content_layout.addWidget(checkbox)
    
    layout.addWidget(content_group)
    
    # Advanced options group
    advanced_group = QGroupBox("Advanced Options")
    advanced_layout = QFormLayout(advanced_group)
    
    # Compression level
    self.compression_level = QSpinBox()
    self.compression_level.setRange(0, 9)
    self.compression_level.setValue(self.backup_manager.config["compression_level"])
    advanced_layout.addRow("Compression Level (0-9):", self.compression_level)
    
    # Verify backups
    self.verify_backups = QCheckBox("Verify backup integrity")
    self.verify_backups.setChecked(self.backup_manager.config["verify_backups"])
    advanced_layout.addRow("", self.verify_backups)
    
    layout.addWidget(advanced_group)
    
    # Manual backup buttons
    button_layout = QHBoxLayout()
    
    backup_now_btn = QPushButton("Backup Now")
    backup_now_btn.clicked.connect(self._run_manual_backup)
    button_layout.addWidget(backup_now_btn)
    
    restore_btn = QPushButton("Restore from Backup")
    restore_btn.clicked.connect(self._restore_from_backup)
    button_layout.addWidget(restore_btn)
    
    layout.addLayout(button_layout)
    
    # Add spacer
    layout.addStretch()
    
    return backup_tab
```

### 2. Integrate with Application Startup/Shutdown

Add backup scheduler initialization to the main application:

```python
# src/main.py

# Import backup components
from src.utils.backup_manager import BackupManager
from src.utils.backup_scheduler import BackupScheduler

# Initialize in application startup
def initialize_components(self):
    # ... existing initialization code ...
    
    # Initialize backup system
    self.backup_manager = BackupManager()
    self.backup_scheduler = BackupScheduler()
    
    # Start backup scheduler if enabled
    if self.backup_manager.config["backup_enabled"]:
        self.backup_scheduler.start()
        
# Clean up on application shutdown
def cleanup_components(self):
    # ... existing cleanup code ...
    
    # Stop backup scheduler
    if hasattr(self, 'backup_scheduler'):
        self.backup_scheduler.stop()
```

## Summary and Next Steps

You've now implemented a robust local backup system for your VSAT application. This system provides:

- Automated, scheduled backups with configurable frequency
- Multiple backup locations (local, external, and network)
- Selective backup of different data categories
- Backup verification to ensure data integrity
- Flexible restoration options
- User-friendly configuration through the settings UI

### Next Steps

1. **Configure Backup Locations**: Set up your preferred backup locations, ensuring at least one is on a separate physical device
2. **Test Restoration Process**: Perform a test restoration to verify your backups are working correctly
3. **Adjust Retention Settings**: Configure retention policies based on your storage constraints and data importance
4. **Document Recovery Procedures**: Create a simple document outlining the steps to restore from backup in case of emergency
5. **Consider Encryption**: If your audio data contains sensitive information, consider adding encryption to your backups

With this backup system in place, your valuable VSAT data is now protected against accidental loss or corruption, allowing you to work with confidence.

Proceed to [Final Configuration](13_final_configuration.md) to optimize your VSAT application settings.

 