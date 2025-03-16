# ML Model Management

## Overview

Effective management of machine learning models is essential for VSAT's voice separation and analysis capabilities. This phase focuses on implementing a robust system for model versioning, updates, and specialized model creation to enhance the application's audio processing abilities.

As the sole user of VSAT, you have the unique opportunity to tailor the ML models to your specific audio processing needs, ensuring optimal performance for your particular use cases. This guide will help you establish a sustainable ML model management framework that maintains model integrity while enabling continuous improvement.

## Prerequisites

Before implementing ML model management features, ensure you have:

- [ ] Completed error recovery and resilience implementation
- [ ] Identified core ML models used across the application
- [ ] Assessed current model performance on your typical audio files
- [ ] 5-7 hours of implementation time
- [ ] 2-3 GB of disk space for model storage and versioning
- [ ] Backup of your current working version and models

## Setting Up Model Versioning

### 1. Create a Model Registry System

Implement a central registry to track all ML models used in the application:

```python
# ml/model_registry.py

class ModelRegistry:
    def __init__(self, registry_path="./models/registry.json"):
        self.registry_path = registry_path
        self.models = {}
        self._ensure_registry_exists()
        self._load_registry()
        
    def _ensure_registry_exists(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w') as f:
                json.dump({}, f)
                
    def _load_registry(self):
        """Load model registry from disk"""
        with open(self.registry_path, 'r') as f:
            self.models = json.load(f)
            
    def _save_registry(self):
        """Save model registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2)
            
    def register_model(self, model_id, model_path, metadata):
        """Register a new model or version in the registry"""
        if model_id not in self.models:
            self.models[model_id] = {
                'versions': [],
                'current_version': None,
                'created_at': datetime.now().isoformat()
            }
            
        # Add new version
        version = len(self.models[model_id]['versions']) + 1
        version_info = {
            'version': version,
            'path': model_path,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
        
        self.models[model_id]['versions'].append(version_info)
        self.models[model_id]['current_version'] = version
        
        self._save_registry()
        return version
        
    def get_model_path(self, model_id, version=None):
        """Get path to a model, optionally for a specific version"""
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found in registry")
            
        if version is None:
            # Use current version
            version = self.models[model_id]['current_version']
            
        # Find version info
        for ver_info in self.models[model_id]['versions']:
            if ver_info['version'] == version:
                return ver_info['path']
                
        raise KeyError(f"Version {version} of model {model_id} not found")
        
    def list_models(self):
        """List all registered models"""
        return {
            model_id: {
                'current_version': info['current_version'],
                'versions_count': len(info['versions'])
            }
            for model_id, info in self.models.items()
        }
        
    def get_model_versions(self, model_id):
        """Get all versions of a specific model"""
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found in registry")
            
        return self.models[model_id]['versions']
        
    def set_current_version(self, model_id, version):
        """Set the current version of a model"""
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found in registry")
            
        # Verify version exists
        version_exists = any(
            v['version'] == version for v in self.models[model_id]['versions']
        )
        
        if not version_exists:
            raise KeyError(f"Version {version} of model {model_id} not found")
            
        self.models[model_id]['current_version'] = version
        self._save_registry()
```

### 2. Implement Model Versioning in Core ML Components

Update your core ML components to use the model registry:

```python
# ml/voice_separator.py

class VoiceSeparator:
    def __init__(self, model_id="voice_separator_default"):
        self.model_id = model_id
        self.registry = ModelRegistry()
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the current version of the model from registry"""
        try:
            model_path = self.registry.get_model_path(self.model_id)
            return self._load_model_from_path(model_path)
        except KeyError:
            # If model not found in registry, use default
            return self._load_default_model()
            
    def _load_model_from_path(self, model_path):
        """Load model from file path"""
        return torch.load(model_path)
        
    def _load_default_model(self):
        """Load and register default model"""
        default_path = "./models/default/voice_separator.pt"
        
        if not os.path.exists(default_path):
            raise FileNotFoundError(
                f"Default model not found at {default_path}"
            )
            
        # Register default model
        self.registry.register_model(
            self.model_id,
            default_path,
            {
                'type': 'default',
                'description': 'Default voice separation model'
            }
        )
        
        return torch.load(default_path)
        
    def separate_voices(self, audio_data):
        """Separate voices in audio using loaded model"""
        # Use the model to separate voices
        return self.model(audio_data)
```

## Implementing Automatic Model Updates

### 1. Create a Model Update Checker

Implement a system to check for and download model updates:

```python
# ml/model_updater.py

class ModelUpdater:
    def __init__(self, update_url="https://models.vsat.example.com/updates"):
        self.update_url = update_url
        self.registry = ModelRegistry()
        
    def check_for_updates(self):
        """Check for available model updates"""
        try:
            response = requests.get(
                f"{self.update_url}/manifest.json",
                timeout=10
            )
            response.raise_for_status()
            
            available_models = response.json()
            local_models = self.registry.list_models()
            
            updates_available = {}
            
            for model_id, model_info in available_models.items():
                remote_version = model_info.get('latest_version')
                
                if model_id in local_models:
                    local_version = local_models[model_id]['current_version']
                    if remote_version > local_version:
                        updates_available[model_id] = {
                            'current': local_version,
                            'available': remote_version,
                            'size_mb': model_info.get('size_mb'),
                            'improvements': model_info.get('improvements', [])
                        }
                else:
                    # New model not in local registry
                    updates_available[model_id] = {
                        'current': None,
                        'available': remote_version,
                        'size_mb': model_info.get('size_mb'),
                        'description': model_info.get('description', '')
                    }
                    
            return updates_available
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            logging.error(f"Failed to check for model updates: {e}")
            return {}
            
    def download_model(self, model_id, version):
        """Download a specific model version"""
        try:
            # Get model info
            response = requests.get(
                f"{self.update_url}/{model_id}/{version}/info.json",
                timeout=10
            )
            response.raise_for_status()
            model_info = response.json()
            
            # Create download directory
            download_dir = f"./models/downloads/{model_id}"
            os.makedirs(download_dir, exist_ok=True)
            
            # Download model file
            model_file = f"{model_id}_v{version}.pt"
            download_path = os.path.join(download_dir, model_file)
            
            response = requests.get(
                f"{self.update_url}/{model_id}/{version}/{model_file}",
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with open(download_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    
            # Register downloaded model
            self.registry.register_model(
                model_id,
                download_path,
                {
                    'version': version,
                    'download_date': datetime.now().isoformat(),
                    'description': model_info.get('description', ''),
                    'size': total_size
                }
            )
            
            return download_path
            
        except Exception as e:
            logging.error(f"Failed to download model {model_id} v{version}: {e}")
            raise
```

### 2. Create a Model Update UI

Implement a UI for checking and applying model updates:

```python
# ui/model_update_dialog.py

class ModelUpdateDialog:
    def __init__(self, parent):
        self.parent = parent
        self.updater = ModelUpdater()
        
    def show(self):
        """Show the model update dialog"""
        # Create dialog window
        dialog = tk.Toplevel(self.parent)
        dialog.title("ML Model Updates")
        dialog.geometry("600x400")
        dialog.resizable(True, True)
        
        # Set up UI
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(
            main_frame, 
            text="ML Model Updates", 
            font=("Arial", 16)
        ).pack(pady=(0, 10))
        
        # Status indicator
        self.status_var = tk.StringVar(value="Checking for updates...")
        ttk.Label(
            main_frame, 
            textvariable=self.status_var
        ).pack(pady=(0, 10))
        
        # Model list frame
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame,
            text="Check for Updates",
            command=lambda: self.check_updates(list_frame)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Close",
            command=dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        # Initial update check
        self.parent.after(100, lambda: self.check_updates(list_frame))
        
        return dialog
        
    def check_updates(self, list_frame):
        """Check for updates and populate the list"""
        self.status_var.set("Checking for updates...")
        
        # Clear existing widgets
        for widget in list_frame.winfo_children():
            widget.destroy()
            
        try:
            updates = self.updater.check_for_updates()
            
            if not updates:
                self.status_var.set("All models are up to date")
                ttk.Label(
                    list_frame,
                    text="No updates available",
                    font=("Arial", 12)
                ).pack(pady=20)
                return
                
            self.status_var.set(f"Found {len(updates)} available updates")
            
            # Create scrollable area
            canvas = tk.Canvas(list_frame)
            scrollbar = ttk.Scrollbar(
                list_frame, orient="vertical", command=canvas.yview
            )
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Add model updates to the scrollable frame
            for model_id, update_info in updates.items():
                model_frame = ttk.Frame(scrollable_frame)
                model_frame.pack(fill=tk.X, padx=5, pady=5)
                
                info_frame = ttk.Frame(model_frame)
                info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                ttk.Label(
                    info_frame,
                    text=model_id,
                    font=("Arial", 12, "bold")
                ).pack(anchor="w")
                
                current = update_info['current'] or "Not installed"
                ttk.Label(
                    info_frame,
                    text=f"Current: v{current} → Available: v{update_info['available']}"
                ).pack(anchor="w")
                
                ttk.Label(
                    info_frame,
                    text=f"Size: {update_info.get('size_mb', 'Unknown')} MB"
                ).pack(anchor="w")
                
                if 'improvements' in update_info and update_info['improvements']:
                    improvements = ', '.join(update_info['improvements'])
                    ttk.Label(
                        info_frame,
                        text=f"Improvements: {improvements}"
                    ).pack(anchor="w")
                    
                # Update button
                ttk.Button(
                    model_frame,
                    text="Update",
                    command=lambda mid=model_id, ver=update_info['available']: 
                        self.download_update(mid, ver)
                ).pack(side=tk.RIGHT, padx=5)
                
        except Exception as e:
            self.status_var.set(f"Error checking updates: {str(e)}")
            logging.error(f"Update check failed: {e}")
            
    def download_update(self, model_id, version):
        """Download and install a model update"""
        self.status_var.set(f"Downloading {model_id} v{version}...")
        
        try:
            # Run download in a separate thread to avoid UI freeze
            def download_thread():
                try:
                    self.updater.download_model(model_id, version)
                    # Update UI from main thread
                    self.parent.after(
                        0,
                        lambda: self.status_var.set(
                            f"Successfully updated {model_id} to v{version}"
                        )
                    )
                except Exception as e:
                    self.parent.after(
                        0,
                        lambda: self.status_var.set(
                            f"Failed to download {model_id}: {str(e)}"
                        )
                    )
                    
            threading.Thread(target=download_thread).start()
            
        except Exception as e:
            self.status_var.set(f"Error initiating download: {str(e)}")
```

## Creating Specialized Models

### 1. Model Fine-Tuning System

Implement a system for fine-tuning models on your specific audio types:

```python
# ml/model_tuner.py

class ModelTuner:
    def __init__(self, base_model_id):
        self.base_model_id = base_model_id
        self.registry = ModelRegistry()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_training_data(self, audio_files, annotations=None):
        """Prepare audio files for fine-tuning"""
        dataset = []
        
        for audio_file in audio_files:
            # Load and preprocess audio
            audio_data = load_audio(audio_file)
            preprocessed = preprocess_audio(audio_data)
            
            # Get annotations if available
            if annotations and audio_file in annotations:
                label = annotations[audio_file]
            else:
                # Use default or automatic labeling
                label = self._generate_default_labels(preprocessed)
                
            dataset.append((preprocessed, label))
            
        return dataset
        
    def fine_tune(self, dataset, new_model_id, epochs=10, batch_size=8):
        """Fine-tune a model with a dataset"""
        # Load base model
        base_model_path = self.registry.get_model_path(self.base_model_id)
        model = torch.load(base_model_path)
        model.to(self.device)
        
        # Prepare data loaders
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        best_val_loss = float('inf')
        best_model = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self._compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            # Validation phase
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = self._compute_loss(outputs, labels)
                    val_loss += loss.item()
                    
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
            
        # Save the best model
        os.makedirs(f"./models/custom", exist_ok=True)
        model_path = f"./models/custom/{new_model_id}.pt"
        torch.save(best_model, model_path)
        
        # Register the new model
        self.registry.register_model(
            new_model_id,
            model_path,
            {
                'base_model': self.base_model_id,
                'training_samples': len(dataset),
                'epochs': epochs,
                'val_loss': best_val_loss,
                'custom': True
            }
        )
        
        return model_path
```

### 2. Model Testing and Evaluation System

Create tools to evaluate model performance on your specific audio:

```python
# ml/model_evaluator.py

class ModelEvaluator:
    def __init__(self):
        self.registry = ModelRegistry()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_model(self, model_id, test_audio_files):
        """Evaluate model performance on test audio files"""
        # Load model
        model_path = self.registry.get_model_path(model_id)
        model = torch.load(model_path)
        model.to(self.device)
        model.eval()
        
        results = {}
        
        for audio_file in test_audio_files:
            # Load and preprocess audio
            audio_data = load_audio(audio_file)
            preprocessed = preprocess_audio(audio_data)
            
            # Run inference
            with torch.no_grad():
                input_tensor = torch.from_numpy(preprocessed).unsqueeze(0).to(self.device)
                output = model(input_tensor)
                
            # Compute metrics
            metrics = self._compute_metrics(output.cpu().numpy(), audio_file)
            results[audio_file] = metrics
            
        return results
        
    def compare_models(self, model_ids, test_audio_file):
        """Compare multiple models on the same audio file"""
        results = {}
        
        # Load and preprocess audio once
        audio_data = load_audio(test_audio_file)
        preprocessed = preprocess_audio(audio_data)
        input_tensor = torch.from_numpy(preprocessed).unsqueeze(0)
        
        for model_id in model_ids:
            # Load model
            model_path = self.registry.get_model_path(model_id)
            model = torch.load(model_path)
            model.to(self.device)
            model.eval()
            
            # Run inference
            with torch.no_grad():
                input_gpu = input_tensor.to(self.device)
                output = model(input_gpu)
                
            # Save output for comparison
            separated_audio = self._output_to_audio(output.cpu().numpy())
            
            # Compute metrics
            metrics = self._compute_metrics(output.cpu().numpy(), test_audio_file)
            
            results[model_id] = {
                'metrics': metrics,
                'output': separated_audio
            }
            
        return results
```

## Conclusion

By implementing these ML model management features, you've created a robust system for maintaining, updating, and customizing the machine learning models used in VSAT. This will enable you to continually improve the application's voice separation and analysis capabilities to meet your specific needs.

### Next Steps

1. **Register your existing models**: Add all current models to the registry
2. **Test the update system**: Verify that the model updater works correctly
3. **Create a specialized model**: Fine-tune a model for your most common audio type
4. **Evaluate model performance**: Compare models on your typical audio files

Proceed to [Personalize User Interface](08_personalize_ui.md) to customize VSAT's interface to better suit your workflow. 