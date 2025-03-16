 # Personal Usage Monitoring

## Overview

Personal usage monitoring allows you to track and analyze how you use VSAT, helping you optimize your workflow, identify patterns, and improve productivity. This guide covers setting up logging for long-term analysis, configuring performance tracking, and implementing alert mechanisms.

## Prerequisites

Before proceeding with personal usage monitoring, ensure you have:

1. Completed the [Integration with External Tools](15_integration_external_tools.md) phase
2. Basic understanding of logging and metrics collection
3. Sufficient disk space for storing logs and usage data
4. Defined your monitoring goals and metrics of interest

## Implementation Steps

### 1. Configuring Logging for Long-Term Analysis

Set up a comprehensive logging system to track VSAT usage over time.

```python
# src/utils/usage_logger.py
import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional
from logging.handlers import RotatingFileHandler

class UsageLogger:
    """Logger for tracking VSAT usage patterns."""
    
    def __init__(self, log_dir: str = None, max_log_size_mb: int = 10, 
                backup_count: int = 5):
        """Initialize the usage logger."""
        self.log_dir = log_dir or os.environ.get("VSAT_LOG_DIR", "./logs/usage")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up the main logger
        self.logger = logging.getLogger("vsat_usage")
        self.logger.setLevel(logging.INFO)
        
        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Set up file handler with rotation
        log_file = os.path.join(self.log_dir, "usage.log")
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_log_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Initialize session data
        self.session_start = datetime.datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d%H%M%S")
        self.actions = []
        
    def log_session_start(self, user_id: str, system_info: Dict[str, Any]) -> None:
        """Log the start of a VSAT session."""
        self.user_id = user_id
        self.system_info = system_info
        
        session_data = {
            "event": "session_start",
            "session_id": self.session_id,
            "user_id": user_id,
            "timestamp": self.session_start.isoformat(),
            "system_info": system_info
        }
        
        self.logger.info(json.dumps(session_data))
        
    def log_session_end(self) -> Dict[str, Any]:
        """Log the end of a VSAT session and return session summary."""
        session_end = datetime.datetime.now()
        duration = (session_end - self.session_start).total_seconds()
        
        session_data = {
            "event": "session_end",
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": session_end.isoformat(),
            "duration_seconds": duration,
            "action_count": len(self.actions)
        }
        
        self.logger.info(json.dumps(session_data))
        
        # Create session summary
        summary = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "end_time": session_end.isoformat(),
            "duration_seconds": duration,
            "user_id": self.user_id,
            "system_info": self.system_info,
            "actions": self.actions
        }
        
        # Save session summary to a JSON file
        summary_file = os.path.join(
            self.log_dir, 
            f"session_{self.session_id}.json"
        )
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
            
        return summary
        
    def log_action(self, action_type: str, details: Dict[str, Any]) -> None:
        """Log a user action within VSAT."""
        timestamp = datetime.datetime.now().isoformat()
        
        action_data = {
            "event": "user_action",
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": timestamp,
            "action_type": action_type,
            "details": details
        }
        
        self.logger.info(json.dumps(action_data))
        
        # Add to actions list for session summary
        self.actions.append({
            "timestamp": timestamp,
            "action_type": action_type,
            "details": details
        })
        
    def log_error(self, error_type: str, message: str, 
                 details: Optional[Dict[str, Any]] = None) -> None:
        """Log an error that occurred during VSAT usage."""
        error_data = {
            "event": "error",
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "error_type": error_type,
            "message": message,
            "details": details or {}
        }
        
        self.logger.error(json.dumps(error_data))
```

### 2. Setting Up Performance Tracking

Implement performance monitoring to track system resource usage during VSAT operations.

```python
# src/utils/performance_tracker.py
import os
import time
import json
import psutil
import platform
import threading
from typing import Dict, Any, List, Optional
import datetime

class PerformanceTracker:
    """Tracks system performance during VSAT usage."""
    
    def __init__(self, log_dir: str = None, sampling_interval: float = 5.0):
        """Initialize the performance tracker."""
        self.log_dir = log_dir or os.environ.get("VSAT_LOG_DIR", "./logs/performance")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.sampling_interval = sampling_interval
        self.is_tracking = False
        self.tracking_thread = None
        self.samples = []
        
        # Get system information
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "total_memory": psutil.virtual_memory().total,
            "total_disk": psutil.disk_usage('/').total
        }
        
    def _sample_performance(self) -> Dict[str, Any]:
        """Take a sample of current system performance."""
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used": psutil.virtual_memory().used,
            "disk_percent": psutil.disk_usage('/').percent,
            "disk_used": psutil.disk_usage('/').used,
            "network_sent": psutil.net_io_counters().bytes_sent,
            "network_received": psutil.net_io_counters().bytes_recv
        }
        
    def _tracking_loop(self) -> None:
        """Background thread for continuous performance tracking."""
        while self.is_tracking:
            sample = self._sample_performance()
            self.samples.append(sample)
            time.sleep(self.sampling_interval)
            
    def start_tracking(self, session_id: str) -> None:
        """Start tracking performance."""
        if self.is_tracking:
            return
            
        self.session_id = session_id
        self.start_time = datetime.datetime.now()
        self.samples = []
        self.is_tracking = True
        
        # Start tracking in a background thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
    def stop_tracking(self) -> Dict[str, Any]:
        """Stop tracking performance and return summary."""
        if not self.is_tracking:
            return None
            
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)
            
        end_time = datetime.datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calculate statistics
        if self.samples:
            cpu_values = [sample["cpu_percent"] for sample in self.samples]
            memory_values = [sample["memory_percent"] for sample in self.samples]
            
            stats = {
                "cpu_min": min(cpu_values),
                "cpu_max": max(cpu_values),
                "cpu_avg": sum(cpu_values) / len(cpu_values),
                "memory_min": min(memory_values),
                "memory_max": max(memory_values),
                "memory_avg": sum(memory_values) / len(memory_values),
                "sample_count": len(self.samples)
            }
        else:
            stats = {
                "cpu_min": 0,
                "cpu_max": 0,
                "cpu_avg": 0,
                "memory_min": 0,
                "memory_max": 0,
                "memory_avg": 0,
                "sample_count": 0
            }
            
        # Create performance report
        report = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "system_info": self.system_info,
            "statistics": stats,
            "samples": self.samples
        }
        
        # Save report to file
        report_file = os.path.join(
            self.log_dir, 
            f"performance_{self.session_id}.json"
        )
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
            
        return report
```

### 3. Creating Alert Mechanisms

Implement alerts to notify you of important events or potential issues.

```python
# src/utils/alert_manager.py
import os
import json
import datetime
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional, Callable

class AlertManager:
    """Manages alerts for VSAT usage monitoring."""
    
    def __init__(self, config_file: str = None):
        """Initialize the alert manager."""
        self.config_file = config_file or os.environ.get(
            "VSAT_ALERT_CONFIG", 
            "./config/alerts.json"
        )
        self.logger = logging.getLogger("vsat_alerts")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize alert handlers
        self.alert_handlers = {
            "console": self._handle_console_alert,
            "log": self._handle_log_alert,
            "email": self._handle_email_alert,
            "desktop": self._handle_desktop_alert
        }
        
        # Initialize alert rules
        self.alert_rules = []
        self._load_alert_rules()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load alert configuration from file."""
        if not os.path.exists(self.config_file):
            # Create default configuration
            config = {
                "enabled": True,
                "alert_methods": ["console", "log"],
                "email": {
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_address": "",
                    "to_address": ""
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Save default configuration
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
                
            return config
        else:
            # Load existing configuration
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading alert configuration: {e}")
                return {"enabled": False, "alert_methods": ["console"]}
                
    def _load_alert_rules(self) -> None:
        """Load alert rules from configuration."""
        rules_file = os.path.join(os.path.dirname(self.config_file), "alert_rules.json")
        
        if not os.path.exists(rules_file):
            # Create default rules
            default_rules = [
                {
                    "name": "High CPU Usage",
                    "condition": "cpu_percent > 90",
                    "duration_seconds": 60,
                    "severity": "warning",
                    "message": "CPU usage has been above 90% for over a minute"
                },
                {
                    "name": "High Memory Usage",
                    "condition": "memory_percent > 85",
                    "duration_seconds": 60,
                    "severity": "warning",
                    "message": "Memory usage has been above 85% for over a minute"
                },
                {
                    "name": "Low Disk Space",
                    "condition": "disk_free_percent < 10",
                    "duration_seconds": 0,
                    "severity": "critical",
                    "message": "Less than 10% disk space remaining"
                }
            ]
            
            # Save default rules
            with open(rules_file, 'w') as f:
                json.dump(default_rules, f, indent=4)
                
            self.alert_rules = default_rules
        else:
            # Load existing rules
            try:
                with open(rules_file, 'r') as f:
                    self.alert_rules = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading alert rules: {e}")
                self.alert_rules = []
                
    def _handle_console_alert(self, alert: Dict[str, Any]) -> None:
        """Handle console alert."""
        severity = alert.get("severity", "info").upper()
        print(f"[{severity}] {alert.get('message', 'Alert triggered')}")
        
    def _handle_log_alert(self, alert: Dict[str, Any]) -> None:
        """Handle log alert."""
        severity = alert.get("severity", "info").lower()
        message = alert.get("message", "Alert triggered")
        
        if severity == "critical":
            self.logger.critical(message)
        elif severity == "error":
            self.logger.error(message)
        elif severity == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
            
    def _handle_email_alert(self, alert: Dict[str, Any]) -> None:
        """Handle email alert."""
        email_config = self.config.get("email", {})
        
        if not all([
            email_config.get("smtp_server"),
            email_config.get("username"),
            email_config.get("from_address"),
            email_config.get("to_address")
        ]):
            self.logger.error("Email alert configuration incomplete")
            return
            
        try:
            msg = MIMEMultipart()
            msg["From"] = email_config["from_address"]
            msg["To"] = email_config["to_address"]
            msg["Subject"] = f"VSAT Alert: {alert.get('name', 'Alert')}"
            
            body = f"""
            Alert: {alert.get('name', 'Alert')}
            Severity: {alert.get('severity', 'info').upper()}
            Time: {datetime.datetime.now().isoformat()}
            
            {alert.get('message', 'Alert triggered')}
            
            Details:
            {json.dumps(alert.get('details', {}), indent=2)}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            server = smtplib.SMTP(
                email_config["smtp_server"], 
                email_config.get("smtp_port", 587)
            )
            server.starttls()
            server.login(
                email_config["username"], 
                email_config["password"]
            )
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
            
    def _handle_desktop_alert(self, alert: Dict[str, Any]) -> None:
        """Handle desktop notification alert."""
        try:
            import plyer.notification
            
            plyer.notification.notify(
                title=f"VSAT Alert: {alert.get('name', 'Alert')}",
                message=alert.get('message', 'Alert triggered'),
                app_name="VSAT",
                timeout=10
            )
        except ImportError:
            self.logger.error("plyer package not installed for desktop notifications")
        except Exception as e:
            self.logger.error(f"Error showing desktop notification: {e}")
            
    def trigger_alert(self, name: str, message: str, severity: str = "info", 
                     details: Optional[Dict[str, Any]] = None) -> None:
        """Trigger an alert with the specified parameters."""
        if not self.config.get("enabled", True):
            return
            
        alert = {
            "name": name,
            "message": message,
            "severity": severity,
            "timestamp": datetime.datetime.now().isoformat(),
            "details": details or {}
        }
        
        # Handle alert through configured methods
        for method in self.config.get("alert_methods", ["console"]):
            handler = self.alert_handlers.get(method)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Error handling {method} alert: {e}")
```

### 4. Implementing Usage Analytics

Create tools to analyze your VSAT usage patterns and generate insights.

```python
# src/utils/usage_analytics.py
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import datetime

class UsageAnalytics:
    """Analyzes VSAT usage patterns from logs."""
    
    def __init__(self, log_dir: str = None):
        """Initialize the usage analytics."""
        self.log_dir = log_dir or os.environ.get("VSAT_LOG_DIR", "./logs")
        self.usage_dir = os.path.join(self.log_dir, "usage")
        self.performance_dir = os.path.join(self.log_dir, "performance")
        
    def load_session_data(self, days: int = 30) -> pd.DataFrame:
        """Load session data from the past specified days."""
        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Find session files
        session_files = glob.glob(os.path.join(self.usage_dir, "session_*.json"))
        
        sessions = []
        for file_path in session_files:
            try:
                with open(file_path, 'r') as f:
                    session = json.load(f)
                    
                # Parse start time and check if within range
                start_time = datetime.datetime.fromisoformat(session["start_time"])
                if start_time >= cutoff_date:
                    sessions.append(session)
            except Exception as e:
                print(f"Error loading session file {file_path}: {e}")
                
        # Convert to DataFrame
        if not sessions:
            return pd.DataFrame()
            
        # Extract key metrics
        session_data = []
        for session in sessions:
            session_data.append({
                "session_id": session["session_id"],
                "start_time": session["start_time"],
                "end_time": session["end_time"],
                "duration_seconds": session["duration_seconds"],
                "user_id": session["user_id"],
                "action_count": len(session.get("actions", []))
            })
            
        return pd.DataFrame(session_data)
        
    def load_performance_data(self, days: int = 30) -> pd.DataFrame:
        """Load performance data from the past specified days."""
        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Find performance files
        perf_files = glob.glob(os.path.join(self.performance_dir, "performance_*.json"))
        
        all_samples = []
        for file_path in perf_files:
            try:
                with open(file_path, 'r') as f:
                    perf_data = json.load(f)
                    
                # Parse start time and check if within range
                start_time = datetime.datetime.fromisoformat(perf_data["start_time"])
                if start_time >= cutoff_date:
                    # Add session_id to each sample
                    for sample in perf_data.get("samples", []):
                        sample["session_id"] = perf_data["session_id"]
                        all_samples.append(sample)
            except Exception as e:
                print(f"Error loading performance file {file_path}: {e}")
                
        # Convert to DataFrame
        if not all_samples:
            return pd.DataFrame()
            
        return pd.DataFrame(all_samples)
        
    def generate_usage_report(self, days: int = 30, 
                             output_dir: str = None) -> Dict[str, Any]:
        """Generate a usage report for the specified time period."""
        output_dir = output_dir or os.path.join(self.log_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        session_df = self.load_session_data(days)
        performance_df = self.load_performance_data(days)
        
        if session_df.empty:
            return {"error": "No session data available"}
            
        # Calculate metrics
        total_sessions = len(session_df)
        total_duration = session_df["duration_seconds"].sum() / 3600  # hours
        avg_session_duration = session_df["duration_seconds"].mean() / 60  # minutes
        total_actions = session_df["action_count"].sum()
        
        # Generate report
        report = {
            "report_date": datetime.datetime.now().isoformat(),
            "period_days": days,
            "metrics": {
                "total_sessions": total_sessions,
                "total_usage_hours": round(total_duration, 2),
                "avg_session_minutes": round(avg_session_duration, 2),
                "total_actions": total_actions,
                "actions_per_hour": round(total_actions / total_duration if total_duration > 0 else 0, 2)
            }
        }
        
        # Generate charts if data available
        if not session_df.empty:
            # Convert start_time to datetime if it's not already
            if isinstance(session_df["start_time"][0], str):
                session_df["start_time"] = pd.to_datetime(session_df["start_time"])
                
            # Group by day
            session_df["date"] = session_df["start_time"].dt.date
            daily_usage = session_df.groupby("date").agg({
                "duration_seconds": "sum",
                "session_id": "count",
                "action_count": "sum"
            })
            
            daily_usage["hours"] = daily_usage["duration_seconds"] / 3600
            
            # Create usage chart
            plt.figure(figsize=(12, 6))
            plt.bar(daily_usage.index, daily_usage["hours"])
            plt.title(f"Daily VSAT Usage (Last {days} Days)")
            plt.xlabel("Date")
            plt.ylabel("Hours")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(output_dir, f"usage_chart_{days}days.png")
            plt.savefig(chart_path)
            plt.close()
            
            report["charts"] = {
                "daily_usage": chart_path
            }
            
        # Save report
        report_path = os.path.join(
            output_dir, 
            f"usage_report_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        return report
```

## Testing Personal Usage Monitoring

After implementing personal usage monitoring, it's important to test each component to ensure proper functionality.

### Testing Usage Logging

1. Start VSAT and perform various actions
2. Check that the usage logs are being created correctly
3. Verify that session data is being recorded accurately
4. Test log rotation by generating sufficient log data

### Testing Performance Tracking

1. Start performance tracking during VSAT usage
2. Monitor system resource usage during different operations
3. Stop tracking and verify that the performance report is generated
4. Check that the performance data is accurate and complete

### Testing Alert Mechanisms

1. Configure different alert methods (console, log, email, desktop)
2. Trigger test alerts with different severity levels
3. Verify that alerts are delivered through the configured methods
4. Test alert rules by simulating conditions that should trigger alerts

### Testing Usage Analytics

1. Generate sufficient usage data over multiple sessions
2. Run the usage analytics to generate reports
3. Verify that the reports contain accurate metrics and charts
4. Check that the insights provided are meaningful and actionable

## Next Steps

After completing personal usage monitoring, you should:

1. Proceed to [Continuous Improvement Framework](17_continuous_improvement.md) to establish a process for ongoing enhancement
2. Analyze your usage patterns to identify areas for workflow optimization
3. Configure alerts based on your specific needs and usage patterns

By completing this phase, you've implemented tools to track and analyze your VSAT usage, helping you optimize your workflow and improve productivity over time.

## References

- `src/utils/usage_logger.py` - Usage logging utility module
- `src/utils/performance_tracker.py` - Performance tracking utility module
- `src/utils/alert_manager.py` - Alert management utility module
- `src/utils/usage_analytics.py` - Usage analytics utility module
- [Python Logging Documentation](https://docs.python.org/3/library/logging.html) - Reference for Python's logging module
- [psutil Documentation](https://psutil.readthedocs.io/en/latest/) - Library for system monitoring and resource tracking
- [Pandas Documentation](https://pandas.pydata.org/docs/) - Data analysis library used for usage analytics
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html) - Plotting library for generating usage charts
- [Python Email Documentation](https://docs.python.org/3/library/email.html) - Reference for email functionality
- [Best Python Application Performance Monitoring Tools](https://betterstack.com/community/comparisons/python-application-monitoring-tools/) - Guide to Python application monitoring