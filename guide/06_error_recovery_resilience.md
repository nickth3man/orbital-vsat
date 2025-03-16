# Error Recovery and Resilience

## Overview

This phase focuses on implementing robust error recovery mechanisms and resilience features to ensure VSAT can gracefully handle unexpected issues during operation. As the sole user of this application, you need a system that can recover from errors without data loss and maintain operational stability even under resource constraints or unexpected conditions.

This guide will help you implement checkpointing for long-running processes, create effective error recovery mechanisms, and design graceful degradation paths for when system resources are limited. By enhancing VSAT's resilience, you'll minimize disruptions to your workflow and protect your audio processing work.

## Prerequisites

Before implementing error recovery and resilience features, ensure you have:

- [ ] Completed performance optimization steps
- [ ] Identified critical processes that need protection
- [ ] Mapped potential failure points in the application
- [ ] Prepared test scenarios that simulate various failure conditions
- [ ] 4-6 hours of implementation time
- [ ] Backup of your current working version

## Implementing Error Recovery Systems

### 1. Add Checkpointing for Long Processes

Long-running processes like audio analysis and ML model training should implement checkpointing to save intermediate states:

```python
# core/checkpoint_manager.py

class CheckpointManager:
    def __init__(self, process_id, checkpoint_dir="./checkpoints"):
        self.process_id = process_id
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_count = 0
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, state_dict, metadata=None):
        """Save current process state to checkpoint file"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.process_id}_checkpoint_{self.checkpoint_count}.pkl"
        )
        
        checkpoint_data = {
            'state': state_dict,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'checkpoint_id': self.checkpoint_count
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        self.checkpoint_count += 1
        return checkpoint_path
        
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint for this process"""
        checkpoint_files = glob.glob(
            os.path.join(self.checkpoint_dir, f"{self.process_id}_checkpoint_*.pkl")
        )
        
        if not checkpoint_files:
            return None
            
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
        with open(latest_checkpoint, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        return checkpoint_data
```

### 2. Implement Automatic Recovery Logic

Add recovery logic to key processing modules:

```python
# processors/audio_processor.py

class ResilientAudioProcessor:
    def __init__(self, checkpoint_frequency=5):
        self.checkpoint_mgr = CheckpointManager("audio_processor")
        self.checkpoint_frequency = checkpoint_frequency
        
    def process_audio_file(self, file_path, options):
        # Try to load previous checkpoint if exists
        checkpoint = self.checkpoint_mgr.load_latest_checkpoint()
        
        if checkpoint:
            print(f"Resuming processing from checkpoint {checkpoint['checkpoint_id']}")
            current_state = checkpoint['state']
            start_position = current_state.get('position', 0)
        else:
            current_state = {'position': 0, 'processed_chunks': []}
            start_position = 0
            
        try:
            # Process audio file in chunks
            audio_data = load_audio(file_path)
            chunks = split_audio_into_chunks(audio_data)
            
            for i, chunk in enumerate(chunks[start_position:]):
                position = start_position + i
                processed_chunk = self._process_chunk(chunk, options)
                current_state['processed_chunks'].append(processed_chunk)
                current_state['position'] = position + 1
                
                # Save checkpoint at specified intervals
                if (position + 1) % self.checkpoint_frequency == 0:
                    self.checkpoint_mgr.save_checkpoint(
                        current_state,
                        {'file': file_path, 'progress': f"{position+1}/{len(chunks)}"}
                    )
                    
            # Final result assembly
            return self._assemble_result(current_state['processed_chunks'])
            
        except Exception as e:
            # Save checkpoint on error
            self.checkpoint_mgr.save_checkpoint(
                current_state,
                {'file': file_path, 'error': str(e)}
            )
            raise
```

### 3. Create an Error Recovery Dashboard

Implement a simple dashboard to manage and recover from errors:

```python
# ui/recovery_dashboard.py

class RecoveryDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("VSAT Recovery Dashboard")
        
        # Set up UI components
        self.setup_ui()
        
        # Load available checkpoints
        self.refresh_checkpoints()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(
            main_frame, 
            text="Error Recovery Dashboard", 
            font=("Arial", 16)
        ).pack(pady=10)
        
        # Checkpoints list
        ttk.Label(
            main_frame, 
            text="Available Recovery Points:", 
            font=("Arial", 12)
        ).pack(anchor="w", pady=(10, 5))
        
        self.checkpoint_frame = ttk.Frame(main_frame)
        self.checkpoint_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame, 
            text="Refresh", 
            command=self.refresh_checkpoints
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Recover Selected", 
            command=self.recover_selected
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Delete Selected", 
            command=self.delete_selected
        ).pack(side=tk.LEFT, padx=5)
```

## Implementing Graceful Degradation

### 1. Resource Monitoring System

Create a system to monitor available resources and adjust processing accordingly:

```python
# core/resource_monitor.py

class ResourceMonitor:
    def __init__(self, check_interval=5):
        self.check_interval = check_interval
        self.last_check = 0
        self.resource_thresholds = {
            'memory': 0.85,  # 85% memory usage
            'cpu': 0.90,     # 90% CPU usage
            'disk': 0.95     # 95% disk usage
        }
        
    def check_resources(self):
        """Check current system resources and return status"""
        current_time = time.time()
        
        # Only check at specified intervals to reduce overhead
        if current_time - self.last_check < self.check_interval:
            return self.last_status
            
        self.last_check = current_time
        
        # Get current resource usage
        memory_usage = psutil.virtual_memory().percent / 100
        cpu_usage = psutil.cpu_percent() / 100
        disk_usage = psutil.disk_usage('/').percent / 100
        
        # Determine status based on thresholds
        status = {
            'memory': {
                'usage': memory_usage,
                'critical': memory_usage > self.resource_thresholds['memory']
            },
            'cpu': {
                'usage': cpu_usage,
                'critical': cpu_usage > self.resource_thresholds['cpu']
            },
            'disk': {
                'usage': disk_usage,
                'critical': disk_usage > self.resource_thresholds['disk']
            }
        }
        
        self.last_status = status
        return status
```

### 2. Adaptive Processing Modes

Implement different processing modes based on available resources:

```python
# processors/adaptive_processor.py

class AdaptiveProcessor:
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.current_mode = "normal"  # normal, reduced, minimal
        
    def process(self, data, options):
        """Process data with adaptive resource usage"""
        # Check resource status
        resources = self.resource_monitor.check_resources()
        
        # Determine appropriate processing mode
        if resources['memory']['critical'] or resources['cpu']['critical']:
            if self.current_mode == "normal":
                self.current_mode = "reduced"
                log.warning("Switching to reduced processing mode due to resource constraints")
            elif self.current_mode == "reduced" and (
                resources['memory']['usage'] > 0.95 or resources['cpu']['usage'] > 0.95
            ):
                self.current_mode = "minimal"
                log.warning("Switching to minimal processing mode due to critical resource constraints")
        else:
            if self.current_mode != "normal":
                self.current_mode = "normal"
                log.info("Returning to normal processing mode")
                
        # Apply processing based on current mode
        if self.current_mode == "normal":
            return self._process_normal(data, options)
        elif self.current_mode == "reduced":
            return self._process_reduced(data, options)
        else:  # minimal
            return self._process_minimal(data, options)
            
    def _process_normal(self, data, options):
        """Full processing with all features enabled"""
        # Implement full processing logic
        pass
        
    def _process_reduced(self, data, options):
        """Reduced processing with some features disabled"""
        # Implement reduced processing logic
        # - Use lower resolution FFT
        # - Skip non-essential post-processing
        # - Use simplified visualization
        pass
        
    def _process_minimal(self, data, options):
        """Minimal processing with only essential features"""
        # Implement minimal processing logic
        # - Use minimum viable processing
        # - Disable all visualizations
        # - Focus only on core functionality
        pass
```

## Implementing Crash Recovery

### 1. Application State Persistence

Create a system to periodically save application state:

```python
# core/app_state_manager.py

class AppStateManager:
    def __init__(self, app_instance, save_interval=60):
        self.app = app_instance
        self.save_interval = save_interval
        self.state_file = os.path.join(
            app_instance.config.get('data_dir', './data'),
            'app_state.json'
        )
        
        # Set up periodic save timer
        self.setup_autosave()
        
    def setup_autosave(self):
        """Set up timer for periodic state saving"""
        def save_state_timer():
            self.save_state()
            # Reschedule timer
            self.app.after(self.save_interval * 1000, save_state_timer)
            
        # Initial timer setup
        self.app.after(self.save_interval * 1000, save_state_timer)
        
    def save_state(self):
        """Save current application state"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'open_files': self.app.get_open_files(),
                'active_view': self.app.get_active_view(),
                'ui_state': self.app.get_ui_state(),
                'recent_actions': self.app.get_recent_actions()
            }
            
            # Create backup of previous state file
            if os.path.exists(self.state_file):
                backup_file = f"{self.state_file}.bak"
                shutil.copy2(self.state_file, backup_file)
                
            # Write new state file
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            log.error(f"Failed to save application state: {e}")
            
    def load_state(self):
        """Load saved application state"""
        if not os.path.exists(self.state_file):
            return None
            
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            return state
        except Exception as e:
            log.error(f"Failed to load application state: {e}")
            
            # Try to load backup if available
            backup_file = f"{self.state_file}.bak"
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r') as f:
                        state = json.load(f)
                    return state
                except Exception as e2:
                    log.error(f"Failed to load backup state: {e2}")
                    
            return None
```

### 2. Crash Detection and Recovery

Implement a crash detection and recovery system:

```python
# core/crash_recovery.py

class CrashHandler:
    def __init__(self, app_instance):
        self.app = app_instance
        self.crash_file = os.path.join(
            app_instance.config.get('data_dir', './data'),
            'crash_flag.txt'
        )
        
    def set_crash_flag(self):
        """Set flag indicating application is running"""
        with open(self.crash_file, 'w') as f:
            f.write(f"{os.getpid()},{datetime.now().isoformat()}")
            
    def clear_crash_flag(self):
        """Clear crash flag on normal shutdown"""
        if os.path.exists(self.crash_file):
            os.remove(self.crash_file)
            
    def check_for_crash(self):
        """Check if previous session crashed"""
        if os.path.exists(self.crash_file):
            try:
                with open(self.crash_file, 'r') as f:
                    content = f.read().strip()
                    
                pid, timestamp = content.split(',', 1)
                
                # Check if process is still running
                try:
                    pid = int(pid)
                    os.kill(pid, 0)  # This will raise an error if process doesn't exist
                    # Process still exists, might be running or zombie
                    return False
                except (OSError, ProcessLookupError):
                    # Process doesn't exist, was a crash
                    return True
                    
            except Exception:
                # If we can't parse the file, assume crash
                return True
                
        return False
        
    def handle_crash_recovery(self):
        """Perform recovery actions after crash detection"""
        if not self.check_for_crash():
            return False
            
        log.warning("Detected crash in previous session, initiating recovery")
        
        # Load saved application state
        state_manager = AppStateManager(self.app)
        saved_state = state_manager.load_state()
        
        if saved_state:
            # Restore application state
            try:
                self.app.restore_from_state(saved_state)
                log.info("Successfully restored application state")
                return True
            except Exception as e:
                log.error(f"Failed to restore application state: {e}")
                
        # Clear crash flag
        self.clear_crash_flag()
        
        # Set new crash flag
        self.set_crash_flag()
        
        return False
```

## Testing Error Recovery Systems

### 1. Create Test Scenarios

Develop a set of test scenarios to validate your error recovery mechanisms:

1. **Forced Application Crash Test**
   - Simulate application crash during processing
   - Verify state recovery on restart

2. **Resource Exhaustion Test**
   - Artificially limit available memory/CPU
   - Verify graceful degradation behavior

3. **Long Process Interruption Test**
   - Interrupt a long-running process
   - Verify checkpoint recovery

4. **Corrupted State Recovery Test**
   - Manually corrupt state files
   - Verify fallback to backup state

### 2. Implement Test Harness

Create a test harness to automate recovery testing:

```python
# tests/recovery_tests.py

class RecoveryTestHarness:
    def __init__(self):
        self.test_results = []
        
    def run_all_tests(self):
        """Run all recovery tests"""
        self.test_forced_crash()
        self.test_resource_exhaustion()
        self.test_process_interruption()
        self.test_corrupted_state()
        
        return self.test_results
        
    def test_forced_crash(self):
        """Test recovery from forced application crash"""
        # Setup test environment
        app = TestApplication()
        app.load_test_file("test_audio.wav")
        app.start_processing()
        
        # Force crash
        app.simulate_crash()
        
        # Restart application
        new_app = TestApplication()
        
        # Check recovery
        result = {
            'test': 'forced_crash',
            'passed': new_app.is_recovered(),
            'details': new_app.get_recovery_details()
        }
        
        self.test_results.append(result)
        
    # Implement other test methods...
```

## Implementing Resilience Best Practices

### 1. Error Logging and Analysis

Enhance error logging to capture detailed information for troubleshooting:

```python
# core/error_logger.py

class EnhancedErrorLogger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up rotating file handler
        self.setup_logging()
        
    def setup_logging(self):
        """Configure enhanced logging"""
        log_file = os.path.join(self.log_dir, "vsat_errors.log")
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Get root logger and add handler
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # Set exception hook to capture unhandled exceptions
        sys.excepthook = self.handle_exception
        
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        # Get logger
        logger = logging.getLogger()
        
        # Log exception with full traceback
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        # Collect system information
        system_info = {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'memory': psutil.virtual_memory()._asdict(),
            'cpu': psutil.cpu_percent(percpu=True),
            'disk': psutil.disk_usage('/')._asdict()
        }
        
        # Log system information
        logger.error(f"System state: {json.dumps(system_info)}")
```

### 2. Implement Circuit Breakers

Add circuit breakers to prevent cascading failures:

```python
# core/circuit_breaker.py

class CircuitBreaker:
    """Circuit breaker pattern implementation to prevent cascading failures"""
    
    # Circuit states
    CLOSED = 'closed'      # Normal operation
    OPEN = 'open'          # Failing, not allowing operations
    HALF_OPEN = 'half-open'  # Testing if system has recovered
    
    def __init__(self, name, failure_threshold=5, recovery_timeout=60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
    def execute(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == self.OPEN:
            # Check if recovery timeout has elapsed
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                log.info(f"Circuit {self.name} trying half-open state")
                self.state = self.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit {self.name} is open, operation rejected"
                )
                
        try:
            result = func(*args, **kwargs)
            
            # Success - reset or close circuit
            if self.state == self.HALF_OPEN:
                log.info(f"Circuit {self.name} closing after successful operation")
                self.reset()
            
            self.last_success_time = time.time()
            return result
            
        except Exception as e:
            # Failure - increment counter and maybe open circuit
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.state == self.CLOSED and 
                self.failure_count >= self.failure_threshold):
                log.warning(
                    f"Circuit {self.name} opening after {self.failure_count} failures"
                )
                self.state = self.OPEN
                
            if self.state == self.HALF_OPEN:
                log.warning(f"Circuit {self.name} reopening after test failure")
                self.state = self.OPEN
                
            raise
            
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = self.CLOSED
        self.failure_count = 0
```

## Conclusion

By implementing these error recovery and resilience features, you've significantly improved VSAT's ability to handle unexpected conditions and recover from failures. These mechanisms will protect your work, minimize disruptions, and ensure the application can adapt to resource constraints.

### Next Steps

1. **Test thoroughly**: Run the test harness to verify all recovery mechanisms work as expected
2. **Monitor in real usage**: Pay attention to how the system behaves during your actual workflow
3. **Refine thresholds**: Adjust resource thresholds based on your specific hardware capabilities
4. **Document recovery procedures**: Note any manual steps needed for severe failure scenarios

Proceed to [ML Model Management](07_ml_model_management.md) to implement effective management of the machine learning models used in VSAT. 