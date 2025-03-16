 # Desktop Integration
====================================================================

## Overview

Desktop integration is a crucial step in making your VSAT application accessible and user-friendly. This phase focuses on integrating VSAT with your desktop environment, allowing for seamless access and operation. By creating shortcuts, setting up file associations, and configuring environment variables, you'll enhance the user experience and streamline workflows.

## Prerequisites

Before proceeding with desktop integration, ensure you have:

1. Completed the [Final Configuration](13_final_configuration.md) phase
2. Installed VSAT with all dependencies
3. Verified that VSAT runs correctly from the command line
4. Administrator privileges on your system (for creating file associations and environment variables)

## Implementation Steps

### 1. Creating Desktop Shortcuts

Creating shortcuts allows quick access to VSAT from your desktop or start menu.

#### Windows Implementation

```python
# src/utils/desktop_integration.py
import os
import sys
import winshell
from win32com.client import Dispatch

def create_desktop_shortcut(target_path, shortcut_name, description="VSAT Application", 
                           icon_path=None, arguments=""):
    """Create a desktop shortcut for the VSAT application."""
    desktop = winshell.desktop()
    shortcut_path = os.path.join(desktop, f"{shortcut_name}.lnk")
    
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.Targetpath = target_path
    shortcut.Arguments = arguments
    shortcut.Description = description
    if icon_path:
        shortcut.IconLocation = icon_path
    shortcut.save()
    
    return shortcut_path

def create_start_menu_shortcut(target_path, shortcut_name, description="VSAT Application",
                              icon_path=None, arguments=""):
    """Create a start menu shortcut for the VSAT application."""
    start_menu = winshell.start_menu()
    programs_path = os.path.join(start_menu, "Programs", "VSAT")
    
    if not os.path.exists(programs_path):
        os.makedirs(programs_path)
        
    shortcut_path = os.path.join(programs_path, f"{shortcut_name}.lnk")
    
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.Targetpath = target_path
    shortcut.Arguments = arguments
    shortcut.Description = description
    if icon_path:
        shortcut.IconLocation = icon_path
    shortcut.save()
    
    return shortcut_path
```

Usage example:

```python
# Example usage in setup or configuration script
from utils.desktop_integration import create_desktop_shortcut, create_start_menu_shortcut

# Path to the main VSAT executable or script
vsat_path = os.path.abspath("vsat_launcher.py")
icon_path = os.path.abspath("assets/icons/vsat.ico")

# Create shortcuts
create_desktop_shortcut(
    target_path=sys.executable,  # Python interpreter
    shortcut_name="VSAT",
    description="Voice Separation & Analysis Tool",
    icon_path=icon_path,
    arguments=f'"{vsat_path}"'
)

create_start_menu_shortcut(
    target_path=sys.executable,
    shortcut_name="VSAT",
    description="Voice Separation & Analysis Tool",
    icon_path=icon_path,
    arguments=f'"{vsat_path}"'
)
```

### 2. Setting Up File Associations

File associations allow audio files to be opened directly with VSAT when double-clicked.

#### Windows Implementation

```python
# src/utils/desktop_integration.py
import os
import winreg
import subprocess

def register_file_association(extension, description, icon_path, command):
    """
    Register a file association in Windows.
    
    Args:
        extension (str): File extension including the dot (e.g., '.wav')
        description (str): Description of the file type
        icon_path (str): Path to the icon file
        command (str): Command to execute when opening this file type
    """
    # Create a progID for this file type
    prog_id = f"VSAT.{extension[1:].upper()}"
    
    # Register the progID
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, f"Software\\Classes\\{prog_id}") as key:
        winreg.SetValue(key, "", winreg.REG_SZ, description)
        
    # Set the icon
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, f"Software\\Classes\\{prog_id}\\DefaultIcon") as key:
        winreg.SetValue(key, "", winreg.REG_SZ, icon_path)
        
    # Set the command
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, f"Software\\Classes\\{prog_id}\\shell\\open\\command") as key:
        winreg.SetValue(key, "", winreg.REG_SZ, command)
        
    # Associate the extension with the progID
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, f"Software\\Classes\\{extension}") as key:
        winreg.SetValue(key, "", winreg.REG_SZ, prog_id)
    
    # Notify the system about the change
    subprocess.run(["assoc", f"{extension}={prog_id}"], shell=True)
    
    return True

def register_vsat_file_associations(vsat_path, icon_path):
    """Register common audio file types to open with VSAT."""
    extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    python_exe = sys.executable
    
    for ext in extensions:
        register_file_association(
            extension=ext,
            description=f"VSAT Audio File ({ext})",
            icon_path=icon_path,
            command=f'"{python_exe}" "{vsat_path}" --file "%1"'
        )
    
    return True
```

Usage example:

```python
# Example usage in setup or configuration script
from utils.desktop_integration import register_vsat_file_associations

# Register file associations
vsat_path = os.path.abspath("vsat_launcher.py")
icon_path = os.path.abspath("assets/icons/vsat.ico")
register_vsat_file_associations(vsat_path, icon_path)
```

### 3. Configuring Environment Variables

Setting environment variables helps VSAT locate resources and configure its behavior.

```python
# src/utils/desktop_integration.py
import os
import winreg
import ctypes

def set_environment_variable(name, value, user_level=True):
    """
    Set an environment variable at user or system level.
    
    Args:
        name (str): Name of the environment variable
        value (str): Value to set
        user_level (bool): If True, set at user level; otherwise, system level
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if user_level:
            key_path = 'Environment'
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, 
                                 winreg.KEY_ALL_ACCESS)
        else:
            # System level requires admin privileges
            key_path = r'System\CurrentControlSet\Control\Session Manager\Environment'
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, 
                                winreg.KEY_ALL_ACCESS)
        
        winreg.SetValueEx(key, name, 0, winreg.REG_EXPAND_SZ, value)
        winreg.CloseKey(key)
        
        # Notify the system about the change
        HWND_BROADCAST = 0xFFFF
        WM_SETTINGCHANGE = 0x001A
        SMTO_ABORTIFHUNG = 0x0002
        result = ctypes.c_long()
        ctypes.windll.user32.SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, 
                                               "Environment", SMTO_ABORTIFHUNG, 5000, 
                                               ctypes.byref(result))
        return True
    except Exception as e:
        print(f"Error setting environment variable: {e}")
        return False

def configure_vsat_environment(vsat_home, models_dir=None, data_dir=None):
    """Configure environment variables for VSAT."""
    variables = {
        "VSAT_HOME": vsat_home,
        "VSAT_MODELS_DIR": models_dir or os.path.join(vsat_home, "models"),
        "VSAT_DATA_DIR": data_dir or os.path.join(vsat_home, "data"),
        "VSAT_TEMP_DIR": os.path.join(vsat_home, "temp"),
        "VSAT_LOG_LEVEL": "INFO"
    }
    
    for name, value in variables.items():
        set_environment_variable(name, value)
    
    return True
```

Usage example:

```python
# Example usage in setup or configuration script
from utils.desktop_integration import configure_vsat_environment

# Configure environment variables
vsat_home = os.path.abspath(".")
configure_vsat_environment(vsat_home)
```

### 4. Setting Startup Parameters

Configure VSAT to start with optimal parameters based on your system and preferences.

```python
# src/utils/desktop_integration.py
import json
import os

def create_startup_config(config_path, settings):
    """
    Create a startup configuration file for VSAT.
    
    Args:
        config_path (str): Path to save the configuration file
        settings (dict): Dictionary of startup settings
    
    Returns:
        str: Path to the created configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(settings, f, indent=4)
    
    return config_path

def update_shortcut_with_startup_params(shortcut_path, startup_params):
    """
    Update an existing shortcut with additional startup parameters.
    
    Args:
        shortcut_path (str): Path to the shortcut file
        startup_params (str): Additional startup parameters
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        
        # Get current arguments and append new ones
        current_args = shortcut.Arguments
        if current_args:
            shortcut.Arguments = f"{current_args} {startup_params}"
        else:
            shortcut.Arguments = startup_params
            
        shortcut.save()
        return True
    except Exception as e:
        print(f"Error updating shortcut: {e}")
        return False
```

Usage example:

```python
# Example usage in setup or configuration script
from utils.desktop_integration import create_startup_config, update_shortcut_with_startup_params

# Create startup configuration
config_path = os.path.join(os.path.abspath("."), "config", "startup.json")
startup_settings = {
    "memory_optimization": True,
    "gpu_acceleration": True,
    "startup_mode": "last_session",
    "auto_update_check": True,
    "log_level": "INFO"
}
create_startup_config(config_path, startup_settings)

# Update desktop shortcut with startup parameters
desktop = winshell.desktop()
shortcut_path = os.path.join(desktop, "VSAT.lnk")
update_shortcut_with_startup_params(shortcut_path, f"--config \"{config_path}\"")
```

## Testing Desktop Integration

After implementing desktop integration, it's essential to test each component to ensure proper functionality.

### Testing Shortcuts

1. Locate the created shortcuts on your desktop and in the start menu
2. Double-click each shortcut to verify that VSAT launches correctly
3. Check that the correct icon is displayed
4. Verify that any startup parameters are applied correctly

### Testing File Associations

1. Create or locate test audio files with each of the registered extensions
2. Double-click each file to verify that VSAT opens and loads the file
3. Right-click the file and check the "Open with" menu to ensure VSAT is listed

### Testing Environment Variables

1. Open a command prompt and run `echo %VSAT_HOME%` to verify the environment variable is set
2. Launch VSAT and check logs or settings to confirm that environment variables are recognized
3. Modify an environment variable and restart VSAT to verify that changes are applied

### Testing Startup Parameters

1. Launch VSAT using the shortcut with startup parameters
2. Verify that the specified configuration is loaded
3. Check system resource usage to confirm that optimization settings are applied

## Next Steps

After completing desktop integration, you should:

1. Proceed to [Integration with External Tools](15_integration_external_tools.md) to extend VSAT's functionality
2. Consider creating multiple desktop shortcuts for different VSAT modes or configurations
3. Document your desktop integration setup for future reference

By completing this phase, you've made VSAT more accessible and user-friendly, integrating it seamlessly with your desktop environment for efficient daily use.

## References

- `src/utils/desktop_integration.py` - Desktop integration utility module
- `assets/icons/` - Directory containing application icons
- [Windows Registry API Documentation](https://docs.microsoft.com/en-us/windows/win32/sysinfo/registry-functions) - Reference for registry manipulation
- [PyWin32 Documentation](https://github.com/mhammond/pywin32) - Python extensions for Windows
- [Windows File Association Explained](https://answers.microsoft.com/en-us/windows/forum/all/windows-file-association-explained-for-desktop/cfa62c00-82e0-4d05-b302-3444ab930bb7) - Microsoft guide on file associations
- [Environment Variables in Windows](https://docs.microsoft.com/en-us/windows/deployment/usmt/usmt-recognized-environment-variables) - Microsoft documentation on environment variables