# Personalize User Interface

## Overview

This phase focuses on tailoring VSAT's user interface to match your personal workflow and preferences. As the sole user of this application, you have the freedom to customize every aspect of the interface to optimize your productivity without worrying about accommodating other users.

A personalized UI reduces cognitive load, speeds up common tasks, and creates a more enjoyable working environment. This guide will help you implement customizations to the layout, controls, keyboard shortcuts, and visualization components to create an interface that feels like it was built specifically for you.

## Prerequisites

Before personalizing the user interface, ensure you have:

- [ ] Completed ML model management implementation
- [ ] Identified your most frequent workflows and tasks
- [ ] Documented any UI pain points from your testing
- [ ] 3-5 hours of implementation time
- [ ] Backup of your current working version

## Customizing the Layout

### 1. Create a Layout Configuration System

Implement a flexible layout system that allows you to reorganize the UI:

```python
# ui/layout_manager.py

class LayoutManager:
    def __init__(self, config_path="./config/layout.json"):
        self.config_path = config_path
        self.layout_config = self._load_config()
        
    def _load_config(self):
        """Load layout configuration from file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = self._create_default_config()
            self._save_config(default_config)
            return default_config
            
    def _create_default_config(self):
        """Create default layout configuration"""
        return {
            'main_window': {
                'width': 1200,
                'height': 800,
                'position_x': None,  # Center on screen
                'position_y': None,
                'maximized': False
            },
            'panels': {
                'file_browser': {
                    'visible': True,
                    'width': 250,
                    'position': 'left'  # left, right, top, bottom
                },
                'waveform': {
                    'visible': True,
                    'height': 200,
                    'position': 'top'
                },
                'spectral_view': {
                    'visible': True,
                    'height': 300,
                    'position': 'middle'
                },
                'controls': {
                    'visible': True,
                    'height': 150,
                    'position': 'bottom'
                },
                'model_info': {
                    'visible': True,
                    'width': 250,
                    'position': 'right'
                }
            },
            'toolbars': {
                'main': {
                    'visible': True,
                    'position': 'top',
                    'items': [
                        'open_file', 'save', 'save_as', 'separator',
                        'undo', 'redo', 'separator',
                        'play', 'pause', 'stop', 'separator',
                        'analyze', 'separate'
                    ]
                },
                'edit': {
                    'visible': True,
                    'position': 'left',
                    'orientation': 'vertical',
                    'items': [
                        'select', 'cut', 'copy', 'paste', 'separator',
                        'zoom_in', 'zoom_out', 'zoom_full'
                    ]
                }
            },
            'status_bar': {
                'visible': True,
                'items': [
                    'position', 'selection', 'separator',
                    'sample_rate', 'bit_depth', 'separator',
                    'memory_usage', 'cpu_usage'
                ]
            }
        }
        
    def _save_config(self, config):
        """Save layout configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    def get_layout(self):
        """Get current layout configuration"""
        return self.layout_config
        
    def save_layout(self):
        """Save current layout to file"""
        self._save_config(self.layout_config)
        
    def update_panel(self, panel_id, properties):
        """Update panel configuration"""
        if panel_id in self.layout_config['panels']:
            self.layout_config['panels'][panel_id].update(properties)
            self._save_config(self.layout_config)
            return True
        return False
        
    def update_toolbar(self, toolbar_id, properties):
        """Update toolbar configuration"""
        if toolbar_id in self.layout_config['toolbars']:
            self.layout_config['toolbars'][toolbar_id].update(properties)
            self._save_config(self.layout_config)
            return True
        return False
        
    def reorder_toolbar_items(self, toolbar_id, items):
        """Reorder toolbar items"""
        if toolbar_id in self.layout_config['toolbars']:
            self.layout_config['toolbars'][toolbar_id]['items'] = items
            self._save_config(self.layout_config)
            return True
        return False
```

### 2. Implement a Customizable Main Window

Create a main window that applies the layout configuration:

```python
# ui/main_window.py

class CustomizableMainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.layout_manager = LayoutManager()
        self.panels = {}
        self.toolbars = {}
        
        # Load layout configuration
        self.layout = self.layout_manager.get_layout()
        
        # Apply window settings
        self._apply_window_settings()
        
        # Create UI components
        self._create_panels()
        self._create_toolbars()
        self._create_status_bar()
        
        # Bind resize event
        self.bind("<Configure>", self._on_window_resize)
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _apply_window_settings(self):
        """Apply window size and position settings"""
        window_config = self.layout['main_window']
        
        # Set window title
        self.title("VSAT - Voice Separation & Analysis Tool")
        
        # Set window size
        self.geometry(f"{window_config['width']}x{window_config['height']}")
        
        # Set window position if specified
        if window_config['position_x'] is not None and window_config['position_y'] is not None:
            self.geometry(f"+{window_config['position_x']}+{window_config['position_y']}")
            
        # Maximize if configured
        if window_config['maximized']:
            self.state('zoomed')
            
    def _create_panels(self):
        """Create and position panels according to layout"""
        # Create main frame to hold panels
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create each panel based on configuration
        for panel_id, panel_config in self.layout['panels'].items():
            if not panel_config['visible']:
                continue
                
            # Create panel frame
            panel_frame = ttk.LabelFrame(self.main_frame, text=panel_id.replace('_', ' ').title())
            
            # Position panel based on configuration
            if panel_config['position'] == 'left':
                panel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
                panel_frame.configure(width=panel_config['width'])
            elif panel_config['position'] == 'right':
                panel_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
                panel_frame.configure(width=panel_config['width'])
            elif panel_config['position'] == 'top':
                panel_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
                panel_frame.configure(height=panel_config['height'])
            elif panel_config['position'] == 'bottom':
                panel_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
                panel_frame.configure(height=panel_config['height'])
            elif panel_config['position'] == 'middle':
                panel_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
            # Store reference to panel
            self.panels[panel_id] = panel_frame
            
    def _create_toolbars(self):
        """Create toolbars according to layout"""
        # Create main toolbar frame
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Create each toolbar based on configuration
        for toolbar_id, toolbar_config in self.layout['toolbars'].items():
            if not toolbar_config['visible']:
                continue
                
            # Create toolbar frame
            toolbar_frame = ttk.Frame(self.toolbar_frame if toolbar_config['position'] == 'top' else self)
            
            # Position toolbar based on configuration
            if toolbar_config['position'] == 'top':
                toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
            elif toolbar_config['position'] == 'bottom':
                toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
            elif toolbar_config['position'] == 'left':
                toolbar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=5)
            elif toolbar_config['position'] == 'right':
                toolbar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=2, pady=5)
                
            # Add toolbar items
            self._create_toolbar_buttons(toolbar_frame, toolbar_config)
            
            # Store reference to toolbar
            self.toolbars[toolbar_id] = toolbar_frame
            
    def _create_toolbar_buttons(self, toolbar_frame, toolbar_config):
        """Create buttons for a toolbar"""
        # Define button actions
        actions = {
            'open_file': {'text': 'Open', 'command': self.open_file, 'icon': 'open.png'},
            'save': {'text': 'Save', 'command': self.save_file, 'icon': 'save.png'},
            'save_as': {'text': 'Save As', 'command': self.save_as, 'icon': 'save_as.png'},
            'undo': {'text': 'Undo', 'command': self.undo, 'icon': 'undo.png'},
            'redo': {'text': 'Redo', 'command': self.redo, 'icon': 'redo.png'},
            'play': {'text': 'Play', 'command': self.play, 'icon': 'play.png'},
            'pause': {'text': 'Pause', 'command': self.pause, 'icon': 'pause.png'},
            'stop': {'text': 'Stop', 'command': self.stop, 'icon': 'stop.png'},
            'analyze': {'text': 'Analyze', 'command': self.analyze, 'icon': 'analyze.png'},
            'separate': {'text': 'Separate', 'command': self.separate, 'icon': 'separate.png'},
            'select': {'text': 'Select', 'command': self.select_tool, 'icon': 'select.png'},
            'cut': {'text': 'Cut', 'command': self.cut, 'icon': 'cut.png'},
            'copy': {'text': 'Copy', 'command': self.copy, 'icon': 'copy.png'},
            'paste': {'text': 'Paste', 'command': self.paste, 'icon': 'paste.png'},
            'zoom_in': {'text': 'Zoom In', 'command': self.zoom_in, 'icon': 'zoom_in.png'},
            'zoom_out': {'text': 'Zoom Out', 'command': self.zoom_out, 'icon': 'zoom_out.png'},
            'zoom_full': {'text': 'Zoom Full', 'command': self.zoom_full, 'icon': 'zoom_full.png'}
        }
        
        # Determine orientation
        is_vertical = toolbar_config.get('orientation') == 'vertical'
        side = tk.TOP if is_vertical else tk.LEFT
        
        # Create buttons for each item
        for item in toolbar_config['items']:
            if item == 'separator':
                # Add separator
                ttk.Separator(
                    toolbar_frame, 
                    orient=tk.VERTICAL if not is_vertical else tk.HORIZONTAL
                ).pack(side=side, padx=5 if not is_vertical else 0, pady=5 if is_vertical else 0, fill='y' if not is_vertical else 'x')
            elif item in actions:
                # Create button with icon if available
                action = actions[item]
                button = ttk.Button(
                    toolbar_frame,
                    text=action['text'] if not is_vertical else '',
                    command=action['command']
                )
                button.pack(side=side, padx=2, pady=2)
                
    def _create_status_bar(self):
        """Create status bar according to layout"""
        if not self.layout['status_bar']['visible']:
            return
            
        # Create status bar frame
        self.status_bar = ttk.Frame(self, relief=tk.SUNKEN, border=1)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status items
        status_vars = {
            'position': tk.StringVar(value="00:00:00.000"),
            'selection': tk.StringVar(value="No selection"),
            'sample_rate': tk.StringVar(value="44.1 kHz"),
            'bit_depth': tk.StringVar(value="16 bit"),
            'memory_usage': tk.StringVar(value="Memory: 0%"),
            'cpu_usage': tk.StringVar(value="CPU: 0%")
        }
        
        # Create each status item
        for item in self.layout['status_bar']['items']:
            if item == 'separator':
                ttk.Separator(self.status_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill='y')
            elif item in status_vars:
                ttk.Label(
                    self.status_bar,
                    textvariable=status_vars[item],
                    padding=(5, 2)
                ).pack(side=tk.LEFT)
                
        # Store status variables for later updates
        self.status_vars = status_vars
        
    def _on_window_resize(self, event):
        """Handle window resize event"""
        # Only update if this is the main window (not a child frame)
        if event.widget == self:
            # Update layout configuration
            if self.state() != 'zoomed':  # Not maximized
                self.layout['main_window']['width'] = self.winfo_width()
                self.layout['main_window']['height'] = self.winfo_height()
                self.layout['main_window']['position_x'] = self.winfo_x()
                self.layout['main_window']['position_y'] = self.winfo_y()
                
    def _on_close(self):
        """Handle window close event"""
        # Save current layout
        self.layout['main_window']['maximized'] = (self.state() == 'zoomed')
        self.layout_manager.save_layout()
        
        # Close application
        self.destroy()
```

### 3. Create a Layout Customization Dialog

Implement a dialog to allow easy customization of the layout:

```python
# ui/layout_customizer.py

class LayoutCustomizer:
    def __init__(self, parent, layout_manager):
        self.parent = parent
        self.layout_manager = layout_manager
        
    def show(self):
        """Show the layout customization dialog"""
        # Create dialog window
        dialog = tk.Toplevel(self.parent)
        dialog.title("Customize Layout")
        dialog.geometry("600x500")
        dialog.resizable(True, True)
        
        # Set up notebook for tabs
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        panels_tab = ttk.Frame(notebook)
        toolbars_tab = ttk.Frame(notebook)
        general_tab = ttk.Frame(notebook)
        
        notebook.add(panels_tab, text="Panels")
        notebook.add(toolbars_tab, text="Toolbars")
        notebook.add(general_tab, text="General")
        
        # Fill panels tab
        self._setup_panels_tab(panels_tab)
        
        # Fill toolbars tab
        self._setup_toolbars_tab(toolbars_tab)
        
        # Fill general tab
        self._setup_general_tab(general_tab)
        
        # Add buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame,
            text="Apply",
            command=lambda: self._apply_changes(dialog)
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Reset to Default",
            command=self._reset_to_default
        ).pack(side=tk.RIGHT, padx=5)
        
        return dialog
        
    def _setup_panels_tab(self, parent):
        """Set up the panels configuration tab"""
        # Get current layout
        layout = self.layout_manager.get_layout()
        
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add panel configuration options
        row = 0
        self.panel_vars = {}
        
        ttk.Label(scrollable_frame, text="Panel", font=("Arial", 10, "bold")).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(scrollable_frame, text="Visible", font=("Arial", 10, "bold")).grid(row=row, column=1, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Position", font=("Arial", 10, "bold")).grid(row=row, column=2, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Size", font=("Arial", 10, "bold")).grid(row=row, column=3, padx=5, pady=5)
        
        row += 1
        
        for panel_id, panel_config in layout['panels'].items():
            # Panel name
            ttk.Label(
                scrollable_frame,
                text=panel_id.replace('_', ' ').title()
            ).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
            
            # Visible checkbox
            visible_var = tk.BooleanVar(value=panel_config['visible'])
            ttk.Checkbutton(
                scrollable_frame,
                variable=visible_var
            ).grid(row=row, column=1, padx=5, pady=5)
            
            # Position combobox
            position_var = tk.StringVar(value=panel_config['position'])
            ttk.Combobox(
                scrollable_frame,
                textvariable=position_var,
                values=['left', 'right', 'top', 'bottom', 'middle'],
                width=10,
                state="readonly"
            ).grid(row=row, column=2, padx=5, pady=5)
            
            # Size entry
            if 'width' in panel_config:
                size_var = tk.StringVar(value=str(panel_config['width']))
                size_label = "Width:"
            else:
                size_var = tk.StringVar(value=str(panel_config['height']))
                size_label = "Height:"
                
            size_frame = ttk.Frame(scrollable_frame)
            size_frame.grid(row=row, column=3, padx=5, pady=5)
            
            ttk.Label(size_frame, text=size_label).pack(side=tk.LEFT)
            ttk.Entry(
                size_frame,
                textvariable=size_var,
                width=5
            ).pack(side=tk.LEFT, padx=2)
            
            # Store variables
            self.panel_vars[panel_id] = {
                'visible': visible_var,
                'position': position_var,
                'size': size_var,
                'size_type': 'width' if 'width' in panel_config else 'height'
            }
            
            row += 1
```

## Configuring Keyboard Shortcuts

### 1. Create a Shortcut Manager

Implement a system to define and manage keyboard shortcuts:

```python
# ui/shortcut_manager.py

class ShortcutManager:
    def __init__(self, config_path="./config/shortcuts.json"):
        self.config_path = config_path
        self.shortcuts = self._load_shortcuts()
        
    def _load_shortcuts(self):
        """Load shortcuts from configuration file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default shortcuts
            default_shortcuts = self._create_default_shortcuts()
            self._save_shortcuts(default_shortcuts)
            return default_shortcuts
            
    def _create_default_shortcuts(self):
        """Create default keyboard shortcuts"""
        return {
            # File operations
            'open_file': {
                'key': 'o',
                'modifier': 'Control',
                'description': 'Open audio file'
            },
            'save_file': {
                'key': 's',
                'modifier': 'Control',
                'description': 'Save current file'
            },
            'save_as': {
                'key': 's',
                'modifier': 'Control+Shift',
                'description': 'Save as new file'
            },
            
            # Playback controls
            'play_pause': {
                'key': 'space',
                'modifier': None,
                'description': 'Play/pause audio'
            },
            'stop': {
                'key': 'Escape',
                'modifier': None,
                'description': 'Stop playback'
            },
            'rewind': {
                'key': 'Left',
                'modifier': None,
                'description': 'Rewind 5 seconds'
            },
            'forward': {
                'key': 'Right',
                'modifier': None,
                'description': 'Forward 5 seconds'
            },
            
            # Editing
            'undo': {
                'key': 'z',
                'modifier': 'Control',
                'description': 'Undo last action'
            },
            'redo': {
                'key': 'y',
                'modifier': 'Control',
                'description': 'Redo last undone action'
            },
            'cut': {
                'key': 'x',
                'modifier': 'Control',
                'description': 'Cut selection'
            },
            'copy': {
                'key': 'c',
                'modifier': 'Control',
                'description': 'Copy selection'
            },
            'paste': {
                'key': 'v',
                'modifier': 'Control',
                'description': 'Paste at cursor'
            },
            'select_all': {
                'key': 'a',
                'modifier': 'Control',
                'description': 'Select all audio'
            },
            
            # View controls
            'zoom_in': {
                'key': 'plus',
                'modifier': 'Control',
                'description': 'Zoom in'
            },
            'zoom_out': {
                'key': 'minus',
                'modifier': 'Control',
                'description': 'Zoom out'
            },
            'zoom_fit': {
                'key': '0',
                'modifier': 'Control',
                'description': 'Zoom to fit'
            }
        }
        
    def _save_shortcuts(self, shortcuts):
        """Save shortcuts to configuration file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(shortcuts, f, indent=2)
            
    def get_shortcuts(self):
        """Get all keyboard shortcuts"""
        return self.shortcuts
        
    def get_shortcut(self, action):
        """Get keyboard shortcut for a specific action"""
        return self.shortcuts.get(action)
        
    def set_shortcut(self, action, key, modifier=None):
        """Set a keyboard shortcut"""
        if action in self.shortcuts:
            self.shortcuts[action]['key'] = key
            self.shortcuts[action]['modifier'] = modifier
            self._save_shortcuts(self.shortcuts)
            return True
        return False
        
    def get_binding(self, action):
        """Get the key binding string for an action"""
        if action not in self.shortcuts:
            return None
            
        shortcut = self.shortcuts[action]
        
        # Build key binding string
        binding = ""
        if shortcut['modifier']:
            # Handle multiple modifiers
            modifiers = shortcut['modifier'].split('+')
            for mod in modifiers:
                if mod == 'Control':
                    binding += '<Control-'
                elif mod == 'Alt':
                    binding += '<Alt-'
                elif mod == 'Shift':
                    binding += '<Shift-'
                    
            # Add the key
            binding += f"{shortcut['key']}>"
        else:
            # Just the key
            if len(shortcut['key']) == 1:
                binding = shortcut['key']
            else:
                binding = f"<{shortcut['key']}>"
                
        return binding
```

## Customizing Visualization Components

### 1. Create Visualization Settings Manager

Implement a system to customize visualization appearance:

```python
# visualizations/vis_settings.py

class VisualizationSettings:
    def __init__(self, config_path="./config/visualization.json"):
        self.config_path = config_path
        self.settings = self._load_settings()
        
    def _load_settings(self):
        """Load visualization settings from file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default settings
            default_settings = self._create_default_settings()
            self._save_settings(default_settings)
            return default_settings
            
    def _create_default_settings(self):
        """Create default visualization settings"""
        return {
            'waveform': {
                'background_color': '#000000',
                'wave_color': '#00FF00',
                'selection_color': '#FF0000',
                'cursor_color': '#FFFFFF',
                'grid_color': '#333333',
                'show_grid': True,
                'grid_interval': 1.0,  # seconds
                'amplitude_scale': 1.0
            },
            'spectrogram': {
                'color_map': 'viridis',  # Options: viridis, inferno, plasma, magma, etc.
                'background_color': '#000000',
                'min_frequency': 0,
                'max_frequency': 20000,
                'fft_size': 2048,
                'window_function': 'hann',
                'show_grid': True,
                'grid_color': '#333333'
            },
            'voice_display': {
                'colors': [
                    '#FF0000',  # Voice 1
                    '#00FF00',  # Voice 2
                    '#0000FF',  # Voice 3
                    '#FFFF00',  # Voice 4
                    '#FF00FF'   # Voice 5
                ],
                'label_font': 'Arial 10',
                'show_labels': True,
                'opacity': 0.7
            }
        }
        
    def _save_settings(self, settings):
        """Save visualization settings to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(settings, f, indent=2)
            
    def get_all_settings(self):
        """Get all visualization settings"""
        return self.settings
        
    def get_component_settings(self, component):
        """Get settings for a specific visualization component"""
        return self.settings.get(component, {})
        
    def update_settings(self, component, settings):
        """Update settings for a visualization component"""
        if component in self.settings:
            self.settings[component].update(settings)
            self._save_settings(self.settings)
            return True
        return False
```

## Conclusion

By implementing these UI customization features, you've created a highly personalized interface that aligns perfectly with your workflow. The custom layout, keyboard shortcuts, and visualization settings will significantly improve your productivity and enjoyment when working with VSAT.

### Next Steps

1. **Experiment with layouts**: Try different panel arrangements to find the most efficient setup
2. **Customize keyboard shortcuts**: Map shortcuts to match your muscle memory from other applications
3. **Fine-tune visualizations**: Adjust color schemes and display options for optimal clarity
4. **Create specialized layouts**: Consider creating purpose-specific layouts for different tasks

Proceed to [Create Personal Documentation](09_personal_documentation.md) to document your workflows and customizations. 