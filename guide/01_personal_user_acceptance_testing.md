# Personal User Acceptance Testing

## Overview

Personal User Acceptance Testing (UAT) is a critical step in finalizing the VSAT application. Unlike traditional UAT that involves multiple stakeholders, this process is tailored specifically for you as the sole user. The goal is to systematically evaluate all aspects of the application against your specific requirements and use cases.

This guide provides a comprehensive approach to conducting effective UAT, ensuring the application meets your needs before proceeding with further optimizations. By focusing on your specific use cases and requirements, you can create a testing process that validates the functionality most important to your workflow.

The key benefits of personal UAT include:
- Validation of features against your specific use cases
- Early detection of issues that would impact your workflow
- Optimization opportunities based on your usage patterns
- Confidence in the application's reliability for your needs
- Clear documentation of application behavior and limitations

## Prerequisites

Before beginning UAT, ensure you have:

### System Requirements
- [ ] A stable installation of VSAT application
- [ ] Python 3.8 or higher
- [ ] At least 8GB of available RAM
- [ ] 20GB of free disk space for test files and results
- [ ] (Optional) CUDA-capable GPU for acceleration testing

### Testing Tools
- [ ] Access to the `tests/user_acceptance_testing.py` script
- [ ] Review of `tests/uat_scenarios.json` containing test scenarios
- [ ] Python testing packages (pytest, coverage, etc.)
- [ ] System monitoring tools (htop, iotop, nvidia-smi if applicable)
- [ ] Screen recording software for UI testing documentation

### Test Data
- [ ] A selection of your own audio files that represent your typical use cases
- [ ] Reference transcripts for accuracy testing
- [ ] Sample files with known challenging conditions
- [ ] Backup copies of all test files

### Environment
- [ ] A quiet environment for testing audio-related functionality
- [ ] A notebook or document for recording observations
- [ ] 2-3 hours of uninterrupted time for focused testing
- [ ] Second monitor (recommended) for testing UI workflows

### Knowledge Requirements
- [ ] Basic understanding of audio processing concepts
- [ ] Familiarity with your typical workflow requirements
- [ ] Understanding of performance metrics and benchmarking
- [ ] Knowledge of your specific quality requirements

## Setting Up Your Testing Framework

### 1. Install Testing Dependencies

First, ensure you have all necessary testing tools and dependencies:

```bash
# Create a virtual environment for testing
python -m venv vsat_test_env
source vsat_test_env/bin/activate  # Linux/Mac
# or
.\vsat_test_env\Scripts\activate  # Windows

# Install required packages
pip install pytest pytest-html pytest-xdist coverage
pip install numpy pandas matplotlib seaborn
pip install soundfile librosa
pip install psutil GPUtil py-cpuinfo

# Install system monitoring tools
# For Ubuntu/Debian
sudo apt-get install htop iotop linux-tools-common
# For Windows, download Process Explorer from Microsoft's website
```

### 2. Configure the Testing Environment

Create a comprehensive test configuration:

```python
# tests/config/test_config.py

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class TestAudioFile:
    path: str
    duration: float
    num_speakers: int
    environment: str
    quality: str
    expected_word_error_rate: float

@dataclass
class TestConfig:
    # Test environment settings
    workspace_dir: str
    output_dir: str
    report_dir: str
    log_level: int = logging.INFO
    
    # Hardware monitoring settings
    monitor_cpu: bool = True
    monitor_memory: bool = True
    monitor_gpu: bool = False
    monitor_disk: bool = True
    
    # Test case settings
    parallel_tests: int = 1
    retry_attempts: int = 2
    timeout_minutes: int = 30
    
    # Performance thresholds
    max_cpu_percent: float = 80.0
    max_memory_gb: float = 4.0
    max_processing_ratio: float = 2.0  # Maximum processing time / audio duration
    
    # Test audio files
    test_files: List[TestAudioFile] = None

def load_test_config() -> TestConfig:
    """Load test configuration from environment or defaults."""
    config = TestConfig(
        workspace_dir=os.getenv('VSAT_TEST_WORKSPACE', '~/vsat_test_files'),
        output_dir=os.getenv('VSAT_TEST_OUTPUT', '~/vsat_test_files/output'),
        report_dir=os.getenv('VSAT_TEST_REPORTS', '~/vsat_test_files/reports'),
    )
    
    # Expand paths
    config.workspace_dir = os.path.expanduser(config.workspace_dir)
    config.output_dir = os.path.expanduser(config.output_dir)
    config.report_dir = os.path.expanduser(config.report_dir)
    
    # Create directories
    os.makedirs(config.workspace_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.report_dir, exist_ok=True)
    
    return config
```

### 3. Set Up Test Scenarios

Create a structured test scenario framework:

```python
# tests/scenarios/test_scenarios.py

from typing import List, Dict, Any
import json
from dataclasses import dataclass, asdict

@dataclass
class TestScenario:
    id: str
    name: str
    description: str
    steps: List[str]
    expected_results: List[str]
    evaluation_criteria: Dict[str, Any]
    prerequisites: List[str]
    estimated_duration: int  # minutes
    priority: int  # 1 (highest) to 5 (lowest)
    
    def to_dict(self) -> dict:
        return asdict(self)

class TestScenarioManager:
    def __init__(self, scenario_file: str = None):
        self.scenarios: List[TestScenario] = []
        if scenario_file:
            self.load_scenarios(scenario_file)
    
    def load_scenarios(self, file_path: str):
        """Load test scenarios from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            for scenario_data in data['scenarios']:
                self.scenarios.append(TestScenario(**scenario_data))
    
    def save_scenarios(self, file_path: str):
        """Save test scenarios to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump({
                'scenarios': [s.to_dict() for s in self.scenarios]
            }, f, indent=2)
    
    def add_scenario(self, scenario: TestScenario):
        """Add a new test scenario."""
        self.scenarios.append(scenario)
    
    def get_scenario_by_id(self, scenario_id: str) -> TestScenario:
        """Get a specific test scenario by ID."""
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise ValueError(f"Scenario {scenario_id} not found")
    
    def get_scenarios_by_priority(self, priority: int) -> List[TestScenario]:
        """Get all scenarios with a specific priority level."""
        return [s for s in self.scenarios if s.priority == priority]
```

### 4. Create Example Test Scenarios

Define some initial test scenarios that cover your core use cases:

```python
# Example test scenarios
def create_example_scenarios():
    scenarios = [
        TestScenario(
            id="UAT-001",
            name="Basic Audio Transcription",
            description="Test basic audio transcription functionality with a clean recording",
            steps=[
                "Select a clear audio file with single speaker",
                "Start transcription with default settings",
                "Wait for completion",
                "Review transcript"
            ],
            expected_results=[
                "Audio file loads successfully",
                "Transcription completes without errors",
                "Transcript matches audio content with >95% accuracy"
            ],
            evaluation_criteria={
                "accuracy": {"min": 0.95, "weight": 0.4},
                "processing_time": {"max": 120, "weight": 0.3},
                "memory_usage": {"max": 2048, "weight": 0.3}
            },
            prerequisites=[
                "Clean audio file available",
                "System resources available"
            ],
            estimated_duration=15,
            priority=1
        ),
        TestScenario(
            id="UAT-002",
            name="Multi-Speaker Diarization",
            description="Test speaker separation in a group conversation",
            steps=[
                "Load multi-speaker audio file",
                "Run diarization process",
                "Verify speaker separation",
                "Check speaker consistency"
            ],
            expected_results=[
                "Correct number of speakers identified",
                "Speakers consistently labeled",
                "Accurate speaker transition points",
                "Minimal cross-talk confusion"
            ],
            evaluation_criteria={
                "speaker_accuracy": {"min": 0.90, "weight": 0.4},
                "diarization_error": {"max": 0.15, "weight": 0.4},
                "processing_efficiency": {"min": 0.8, "weight": 0.2}
            ],
            prerequisites=[
                "Multi-speaker recording available",
                "Known speaker count and timestamps"
            ],
            estimated_duration=25,
            priority=1
        )
    ]
    
    manager = TestScenarioManager()
    for scenario in scenarios:
        manager.add_scenario(scenario)
    
    return manager

# Save example scenarios
if __name__ == "__main__":
    manager = create_example_scenarios()
    manager.save_scenarios("tests/scenarios/default_scenarios.json")
```

### 5. Set Up Performance Monitoring

Create utilities for monitoring system performance during tests:

```python
# tests/monitoring/performance_monitor.py

import psutil
import GPUtil
import time
import threading
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_read_bytes: int
    disk_write_bytes: int
    gpu_utilization: float = None
    gpu_memory_used: float = None

class PerformanceMonitor:
    def __init__(self, interval: float = 1.0, gpu_enabled: bool = False):
        self.interval = interval
        self.gpu_enabled = gpu_enabled
        self.metrics: List[PerformanceMetrics] = []
        self._monitoring = False
        self._monitor_thread = None
    
    def start(self):
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            metrics = self._collect_metrics()
            self.metrics.append(metrics)
            time.sleep(self.interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_io_counters()
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu,
            memory_percent=memory,
            disk_read_bytes=disk.read_bytes,
            disk_write_bytes=disk.write_bytes
        )
        
        if self.gpu_enabled:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    metrics.gpu_utilization = gpus[0].load * 100
                    metrics.gpu_memory_used = gpus[0].memoryUsed
            except Exception as e:
                print(f"Error collecting GPU metrics: {e}")
        
        return metrics
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of collected metrics."""
        if not self.metrics:
            return {}
        
        return {
            "cpu_avg": sum(m.cpu_percent for m in self.metrics) / len(self.metrics),
            "cpu_max": max(m.cpu_percent for m in self.metrics),
            "memory_avg": sum(m.memory_percent for m in self.metrics) / len(self.metrics),
            "memory_max": max(m.memory_percent for m in self.metrics),
            "disk_read_total": self.metrics[-1].disk_read_bytes - self.metrics[0].disk_read_bytes,
            "disk_write_total": self.metrics[-1].disk_write_bytes - self.metrics[0].disk_write_bytes
        }
```

### 6. Implement Test Execution Framework

Create a robust framework for executing and monitoring tests:

```python
# tests/framework/test_executor.py

import time
import logging
import psutil
import threading
from typing import Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime
from ..monitoring.performance_monitor import PerformanceMonitor
from ..config.test_config import TestConfig
from ..scenarios.test_scenarios import TestScenario

@dataclass
class TestResult:
    scenario_id: str
    start_time: datetime
    end_time: datetime
    success: bool
    metrics: Dict[str, float]
    errors: List[str]
    logs: List[str]
    performance_metrics: Dict[str, float]

class TestExecutor:
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results: Dict[str, TestResult] = {}
        self.performance_monitor = PerformanceMonitor(
            gpu_enabled=self.config.monitor_gpu
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for test execution."""
        logger = logging.getLogger('vsat_test')
        logger.setLevel(self.config.log_level)
        
        # File handler
        fh = logging.FileHandler(
            os.path.join(self.config.report_dir, f'test_run_{time.strftime("%Y%m%d_%H%M%S")}.log')
        )
        fh.setLevel(self.config.log_level)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def execute_scenario(self, scenario: TestScenario) -> TestResult:
        """Execute a single test scenario."""
        self.logger.info(f"Starting scenario: {scenario.id} - {scenario.name}")
        
        # Start performance monitoring
        self.performance_monitor.start()
        
        start_time = datetime.now()
        result = TestResult(
            scenario_id=scenario.id,
            start_time=start_time,
            end_time=None,
            success=False,
            metrics={},
            errors=[],
            logs=[],
            performance_metrics={}
        )
        
        try:
            # Execute test steps
            for i, step in enumerate(scenario.steps, 1):
                self.logger.info(f"Executing step {i}: {step}")
                try:
                    self._execute_step(step, scenario)
                except Exception as e:
                    self.logger.error(f"Error in step {i}: {str(e)}")
                    result.errors.append(f"Step {i}: {str(e)}")
                    raise
            
            # Evaluate results against criteria
            result.metrics = self._evaluate_scenario(scenario)
            result.success = all(
                metric >= criteria['min'] if 'min' in criteria else metric <= criteria['max']
                for metric, criteria in scenario.evaluation_criteria.items()
            )
            
        except Exception as e:
            self.logger.error(f"Error in scenario {scenario.id}: {str(e)}")
            result.errors.append(str(e))
            result.success = False
        
        # Stop monitoring and collect results
        self.performance_monitor.stop()
        result.end_time = datetime.now()
        result.performance_metrics = self.performance_monitor.get_summary()
        
        self.results[scenario.id] = result
        return result
    
    def _execute_step(self, step: str, scenario: TestScenario):
        """Execute a single test step."""
        # This would be implemented based on your specific test steps
        # Example implementation:
        if "load" in step.lower() and "audio file" in step.lower():
            self._execute_load_audio_step(step)
        elif "transcription" in step.lower():
            self._execute_transcription_step(step)
        elif "diarization" in step.lower():
            self._execute_diarization_step(step)
        else:
            self.logger.warning(f"Unknown step type: {step}")
    
    def _evaluate_scenario(self, scenario: TestScenario) -> Dict[str, float]:
        """Evaluate scenario results against criteria."""
        metrics = {}
        for metric, criteria in scenario.evaluation_criteria.items():
            # Implement metric calculation based on your needs
            if metric == "accuracy":
                metrics[metric] = self._calculate_accuracy()
            elif metric == "processing_time":
                metrics[metric] = self._calculate_processing_time()
            elif metric == "memory_usage":
                metrics[metric] = self._calculate_memory_usage()
            else:
                self.logger.warning(f"Unknown metric: {metric}")
                metrics[metric] = 0.0
        
        return metrics
    
    def _calculate_accuracy(self) -> float:
        """Calculate accuracy metrics."""
        # Implement accuracy calculation
        return 0.0
    
    def _calculate_processing_time(self) -> float:
        """Calculate processing time metrics."""
        # Implement processing time calculation
        return 0.0
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage metrics."""
        # Implement memory usage calculation
        return 0.0
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        report_path = os.path.join(
            self.config.report_dir,
            f'test_report_{time.strftime("%Y%m%d_%H%M%S")}'
        )
        
        # Generate HTML report
        self._generate_html_report(f"{report_path}.html")
        
        # Generate JSON report
        self._generate_json_report(f"{report_path}.json")
        
        # Generate performance graphs
        self._generate_performance_graphs(f"{report_path}_performance.png")
    
    def _generate_html_report(self, path: str):
        """Generate HTML test report."""
        import jinja2
        import os
        
        # Load template
        template_loader = jinja2.FileSystemLoader(searchpath="./tests/templates")
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template("report_template.html")
        
        # Prepare report data
        report_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': [
                {
                    'id': result.scenario_id,
                    'name': scenario.name,
                    'success': result.success,
                    'duration': (result.end_time - result.start_time).total_seconds(),
                    'metrics': result.metrics,
                    'errors': result.errors,
                    'performance': result.performance_metrics
                }
                for scenario_id, result in self.results.items()
                if (scenario := self.scenario_manager.get_scenario_by_id(scenario_id))
            ]
        }
        
        # Render template
        output_text = template.render(report_data)
        
        # Write to file
        with open(path, 'w') as f:
            f.write(output_text)
    
    def _generate_json_report(self, path: str):
        """Generate JSON test report."""
        import json
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'scenario_id': result.scenario_id,
                    'success': result.success,
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'metrics': result.metrics,
                    'errors': result.errors,
                    'performance_metrics': result.performance_metrics
                }
                for result in self.results.values()
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def _generate_performance_graphs(self, path: str):
        """Generate performance visualization graphs."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the plot style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot CPU usage
        cpu_data = [result.performance_metrics['cpu_avg'] for result in self.results.values()]
        scenario_names = [result.scenario_id for result in self.results.values()]
        
        sns.barplot(x=scenario_names, y=cpu_data, ax=ax1)
        ax1.set_title('Average CPU Usage by Scenario')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot memory usage
        memory_data = [result.performance_metrics['memory_avg'] for result in self.results.values()]
        sns.barplot(x=scenario_names, y=memory_data, ax=ax2)
        ax2.set_title('Average Memory Usage by Scenario')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot execution time
        time_data = [(result.end_time - result.start_time).total_seconds() 
                    for result in self.results.values()]
        sns.barplot(x=scenario_names, y=time_data, ax=ax3)
        ax3.set_title('Execution Time by Scenario')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
```

### 7. Create Report Templates

Create HTML templates for test reports:

```html
<!-- tests/templates/report_template.html -->
<!DOCTYPE html>
<html>
<head>
    <title>VSAT Personal UAT Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .summary {
            margin-bottom: 30px;
        }
        .scenario {
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .success {
            border-left: 5px solid #4CAF50;
        }
        .failure {
            border-left: 5px solid #f44336;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .metric {
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .errors {
            color: #f44336;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>VSAT Personal UAT Report</h1>
            <p>Generated on {{ timestamp }}</p>
        </div>
        
        <div class="summary">
            <h2>Test Summary</h2>
            <p>Total Scenarios: {{ results|length }}</p>
            <p>Successful: {{ results|selectattr('success')|list|length }}</p>
            <p>Failed: {{ results|rejectattr('success')|list|length }}</p>
        </div>
        
        <div class="scenarios">
            <h2>Test Results</h2>
            {% for result in results %}
            <div class="scenario {{ 'success' if result.success else 'failure' }}">
                <h3>{{ result.name }} ({{ result.id }})</h3>
                <p>Status: {{ 'Passed' if result.success else 'Failed' }}</p>
                <p>Duration: {{ '%.2f'|format(result.duration) }} seconds</p>
                
                <div class="metrics">
                    {% for name, value in result.metrics.items() %}
                    <div class="metric">
                        <strong>{{ name|title }}:</strong>
                        <span>{{ '%.2f'|format(value) }}</span>
                    </div>
                    {% endfor %}
                    
                    {% for name, value in result.performance.items() %}
                    <div class="metric">
                        <strong>{{ name|replace('_', ' ')|title }}:</strong>
                        <span>{{ '%.2f'|format(value) }}</span>
                    </div>
                    {% endfor %}
                </div>
                
                {% if result.errors %}
                <div class="errors">
                    <h4>Errors:</h4>
                    <ul>
                        {% for error in result.errors %}
                        <li>{{ error }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
```

## Test Execution and Monitoring

### 1. Running Tests

To execute your personal UAT test suite:

```python
# tests/run_uat.py

import os
import sys
import argparse
from framework.test_executor import TestExecutor
from config.test_config import TestConfig
from scenarios.test_scenarios import TestScenarioManager

def parse_args():
    parser = argparse.ArgumentParser(description='Run VSAT Personal UAT')
    parser.add_argument('--config', type=str, default='config/uat_config.json',
                       help='Path to test configuration file')
    parser.add_argument('--scenarios', type=str, default='scenarios/uat_scenarios.json',
                       help='Path to test scenarios file')
    parser.add_argument('--report-dir', type=str, default='reports',
                       help='Directory for test reports')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = TestConfig.from_json(args.config)
    config.report_dir = args.report_dir
    config.log_level = args.log_level
    
    # Create report directory if it doesn't exist
    os.makedirs(config.report_dir, exist_ok=True)
    
    # Load test scenarios
    scenario_manager = TestScenarioManager()
    scenario_manager.load_scenarios(args.scenarios)
    
    # Initialize test executor
    executor = TestExecutor(config)
    
    # Execute all scenarios
    for scenario in scenario_manager.get_all_scenarios():
        print(f"\nExecuting scenario: {scenario.name}")
        result = executor.execute_scenario(scenario)
        print(f"Status: {'Passed' if result.success else 'Failed'}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
    
    # Generate reports
    executor.generate_report()
    print(f"\nTest reports generated in: {config.report_dir}")

if __name__ == '__main__':
    main()
```

### 2. Real-time Monitoring

During test execution, you can monitor the system in real-time:

```python
# tests/monitor_uat.py

import time
import psutil
import GPUtil
import curses
from datetime import datetime

def monitor_system(stdscr):
    """Monitor system resources in real-time during UAT."""
    # Initialize colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    
    while True:
        try:
            # Clear screen
            stdscr.clear()
            
            # Get current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stdscr.addstr(0, 0, f"VSAT UAT System Monitor - {current_time}\n")
            stdscr.addstr(1, 0, "=" * 50 + "\n\n")
            
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            color = (1 if cpu_percent < 70 else 2 if cpu_percent < 90 else 3)
            stdscr.addstr(3, 0, "CPU Usage: ")
            stdscr.addstr(f"{cpu_percent:.1f}%\n", curses.color_pair(color))
            
            # Memory Usage
            memory = psutil.virtual_memory()
            color = (1 if memory.percent < 70 else 2 if memory.percent < 90 else 3)
            stdscr.addstr(4, 0, "Memory Usage: ")
            stdscr.addstr(f"{memory.percent:.1f}% (Used: {memory.used/1024/1024/1024:.1f}GB, "
                         f"Total: {memory.total/1024/1024/1024:.1f}GB)\n",
                         curses.color_pair(color))
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            color = (1 if disk.percent < 70 else 2 if disk.percent < 90 else 3)
            stdscr.addstr(5, 0, "Disk Usage: ")
            stdscr.addstr(f"{disk.percent}% (Used: {disk.used/1024/1024/1024:.1f}GB, "
                         f"Total: {disk.total/1024/1024/1024:.1f}GB)\n",
                         curses.color_pair(color))
            
            # GPU Usage (if available)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    for i, gpu in enumerate(gpus):
                        color = (1 if gpu.load < 0.7 else 2 if gpu.load < 0.9 else 3)
                        stdscr.addstr(6+i, 0, f"GPU {i} Usage: ")
                        stdscr.addstr(f"{gpu.load*100:.1f}% (Memory: {gpu.memoryUsed}MB/"
                                    f"{gpu.memoryTotal}MB, Temp: {gpu.temperature}°C)\n",
                                    curses.color_pair(color))
            except Exception:
                stdscr.addstr(6, 0, "GPU monitoring not available\n")
            
            # Network I/O
            net_io = psutil.net_io_counters()
            stdscr.addstr(8, 0, f"Network I/O: Sent: {net_io.bytes_sent/1024/1024:.1f}MB, "
                               f"Received: {net_io.bytes_recv/1024/1024:.1f}MB\n")
            
            # Update screen
            stdscr.refresh()
            time.sleep(1)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            stdscr.addstr(10, 0, f"Error: {str(e)}\n")
            stdscr.refresh()
            time.sleep(1)

def main():
    curses.wrapper(monitor_system)

if __name__ == '__main__':
    main()
```

### 3. Analyzing Results

After test execution, analyze the generated reports:

1. **HTML Report**: Open the generated HTML report in your browser for a visual overview of test results.
2. **JSON Report**: Use the JSON report for programmatic analysis or integration with other tools.
3. **Performance Graphs**: Review the performance graphs to identify resource usage patterns and potential bottlenecks.
4. **Log Files**: Check the detailed log files for specific test execution information and error messages.

### 4. Common Issues and Solutions

When running personal UAT, you might encounter these common issues:

1. **Resource Constraints**:
   - *Symptom*: High CPU/memory usage affecting test execution
   - *Solution*: Adjust test configuration to limit concurrent operations or increase resource allocation

2. **Audio Processing Errors**:
   - *Symptom*: Transcription or diarization failures
   - *Solution*: Verify audio file format and quality, check system audio configuration

3. **Performance Degradation**:
   - *Symptom*: Slower than expected processing times
   - *Solution*: Monitor system resources, close unnecessary applications, optimize test scenarios

4. **Report Generation Failures**:
   - *Symptom*: Missing or incomplete test reports
   - *Solution*: Verify write permissions, ensure sufficient disk space

### 5. Best Practices

Follow these best practices for effective personal UAT:

1. **Test Environment**:
   - Use a clean test environment
   - Close unnecessary applications
   - Ensure stable system resources

2. **Test Data**:
   - Use representative audio samples
   - Include edge cases in test scenarios
   - Maintain test data versioning

3. **Test Execution**:
   - Run tests in a consistent order
   - Monitor system resources during execution
   - Document any deviations from expected behavior

4. **Results Analysis**:
   - Review all generated reports
   - Track performance metrics over time
   - Document and investigate failures

5. **Maintenance**:
   - Regularly update test scenarios
   - Clean up old test reports
   - Keep test framework dependencies updated

## Conclusion

Personal User Acceptance Testing is crucial for ensuring VSAT meets your specific requirements and use cases. By following this guide and utilizing the provided framework, you can:

- Systematically validate VSAT functionality
- Monitor system performance during testing
- Generate comprehensive test reports
- Identify and resolve issues early
- Maintain consistent testing practices

Remember to regularly update your test scenarios as your requirements evolve and to document any specific configurations or modifications made during testing.

---

## Appendix: UAT Checklist

Use this checklist to ensure you've covered all essential testing areas:

- [ ] Tested all file formats you commonly use
- [ ] Verified speaker separation with different numbers of speakers
- [ ] Tested transcription with vocabulary common to your domain
- [ ] Verified search functionality with typical search terms
- [ ] Tested export formats you regularly use
- [ ] Verified performance with your typical file sizes
- [ ] Tested error scenarios and recovery
- [ ] Evaluated UI efficiency for your common workflows
- [ ] Checked accessibility features relevant to your needs
- [ ] Monitored system resource usage during testing
- [ ] Documented all findings thoroughly
- [ ] Generated and saved test reports

## References

- `tests/user_acceptance_testing.py` - Main UAT framework script
- `tests/uat_scenarios.json` - Default test scenarios
- `tests/README_UAT.md` - Additional documentation on the UAT framework 