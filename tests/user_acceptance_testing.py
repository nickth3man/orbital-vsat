#!/usr/bin/env python3
"""
User Acceptance Testing (UAT) for VSAT.

This script provides a framework for conducting user acceptance testing
of the VSAT application, including test scenarios, metrics collection,
and feedback reporting.
"""

import os
import sys
import logging
import json
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import PyQt6.QtWidgets as QtWidgets
from PyQt6.QtCore import Qt, QTimer

from src.ui.app import MainWindow
from src.utils.error_handler import install_global_error_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class UserAcceptanceTesting:
    """Framework for conducting user acceptance testing."""
    
    def __init__(self):
        """Initialize the UAT framework."""
        self.app = None
        self.main_window = None
        self.test_scenarios = []
        self.results = {}
        self.feedback = {}
        self.metrics = {}
        
        # Create reports directory if it doesn't exist
        self.reports_dir = Path("uat_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load test scenarios
        self._load_test_scenarios()
    
    def _load_test_scenarios(self):
        """Load test scenarios from JSON file."""
        scenarios_file = Path("tests/uat_scenarios.json")
        
        if not scenarios_file.exists():
            # Create default scenarios if file doesn't exist
            self._create_default_scenarios(scenarios_file)
        
        try:
            with open(scenarios_file, "r") as f:
                self.test_scenarios = json.load(f)
            
            logger.info(f"Loaded {len(self.test_scenarios)} test scenarios")
            
        except Exception as e:
            logger.error(f"Failed to load test scenarios: {str(e)}")
            self.test_scenarios = []
    
    def _create_default_scenarios(self, file_path: Path):
        """Create default test scenarios.
        
        Args:
            file_path: Path to save the scenarios
        """
        default_scenarios = [
            {
                "id": "basic_workflow",
                "name": "Basic Audio Processing Workflow",
                "description": "Test the basic workflow of loading an audio file, processing it, and viewing the results.",
                "steps": [
                    "Open an audio file",
                    "Process the audio with default settings",
                    "View the waveform and transcript",
                    "Play the audio and verify synchronization",
                    "Search for a word in the transcript",
                    "Export the transcript"
                ],
                "expected_results": "Audio file is processed correctly, waveform and transcript are displayed, playback is synchronized, search works, and transcript can be exported.",
                "metrics": ["time_to_complete", "errors_encountered", "user_satisfaction"]
            },
            {
                "id": "batch_processing",
                "name": "Batch Processing",
                "description": "Test the batch processing functionality with multiple audio files.",
                "steps": [
                    "Open the batch processing dialog",
                    "Add multiple audio files",
                    "Configure processing options",
                    "Start batch processing",
                    "Monitor progress",
                    "View results after completion"
                ],
                "expected_results": "Multiple files are processed correctly, progress is displayed, and results are accessible after completion.",
                "metrics": ["time_to_complete", "errors_encountered", "user_satisfaction", "processing_speed"]
            },
            {
                "id": "speaker_identification",
                "name": "Speaker Identification",
                "description": "Test the speaker identification functionality across multiple recordings.",
                "steps": [
                    "Process multiple recordings with the same speakers",
                    "Verify speaker identification across recordings",
                    "Manually correct speaker labels",
                    "Verify that corrections are applied"
                ],
                "expected_results": "Speakers are correctly identified across recordings, manual corrections work, and corrections are applied consistently.",
                "metrics": ["identification_accuracy", "time_to_correct", "user_satisfaction"]
            },
            {
                "id": "content_analysis",
                "name": "Content Analysis",
                "description": "Test the content analysis functionality.",
                "steps": [
                    "Process an audio file",
                    "Open the content analysis panel",
                    "View topic modeling results",
                    "View keyword extraction results",
                    "View sentiment analysis results"
                ],
                "expected_results": "Content analysis results are displayed correctly and provide useful insights.",
                "metrics": ["analysis_quality", "user_satisfaction"]
            },
            {
                "id": "accessibility",
                "name": "Accessibility Features",
                "description": "Test the accessibility features of the application.",
                "steps": [
                    "Enable high contrast mode",
                    "Test keyboard navigation",
                    "Test screen reader compatibility",
                    "Adjust font sizes"
                ],
                "expected_results": "Accessibility features work correctly and improve usability for users with disabilities.",
                "metrics": ["keyboard_navigation_score", "screen_reader_compatibility", "user_satisfaction"]
            }
        ]
        
        try:
            with open(file_path, "w") as f:
                json.dump(default_scenarios, f, indent=2)
            
            logger.info(f"Created default test scenarios at {file_path}")
            self.test_scenarios = default_scenarios
            
        except Exception as e:
            logger.error(f"Failed to create default test scenarios: {str(e)}")
            self.test_scenarios = []
    
    def start_application(self):
        """Start the VSAT application for testing."""
        try:
            # Create Qt application
            self.app = QtWidgets.QApplication(sys.argv)
            self.app.setApplicationName("VSAT - User Acceptance Testing")
            
            # Install global error handler
            install_global_error_handler()
            
            # Create main window
            self.main_window = MainWindow(self.app)
            self.main_window.show()
            
            logger.info("VSAT application started for UAT")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start application: {str(e)}")
            return False
    
    def run_test_scenario(self, scenario_id: str):
        """Run a specific test scenario.
        
        Args:
            scenario_id: ID of the scenario to run
        
        Returns:
            True if the scenario was run successfully, False otherwise
        """
        # Find the scenario
        scenario = next((s for s in self.test_scenarios if s["id"] == scenario_id), None)
        
        if not scenario:
            logger.error(f"Scenario with ID '{scenario_id}' not found")
            return False
        
        logger.info(f"Running test scenario: {scenario['name']}")
        
        # Display scenario information
        self._display_scenario_info(scenario)
        
        # Start timing
        start_time = time.time()
        
        # Wait for user to complete the scenario
        self._wait_for_user_completion()
        
        # Calculate time to complete
        end_time = time.time()
        time_to_complete = end_time - start_time
        
        # Collect feedback
        feedback = self._collect_feedback(scenario)
        
        # Store results
        self.results[scenario_id] = {
            "scenario": scenario,
            "time_to_complete": time_to_complete,
            "feedback": feedback,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Completed test scenario: {scenario['name']}")
        return True
    
    def _display_scenario_info(self, scenario: Dict[str, Any]):
        """Display scenario information to the user.
        
        Args:
            scenario: Scenario information
        """
        dialog = QtWidgets.QDialog(self.main_window)
        dialog.setWindowTitle(f"Test Scenario: {scenario['name']}")
        dialog.setMinimumSize(600, 400)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Scenario information
        info_group = QtWidgets.QGroupBox("Scenario Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        
        name_label = QtWidgets.QLabel(f"<b>Name:</b> {scenario['name']}")
        description_label = QtWidgets.QLabel(f"<b>Description:</b> {scenario['description']}")
        description_label.setWordWrap(True)
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(description_label)
        
        # Steps
        steps_group = QtWidgets.QGroupBox("Steps to Complete")
        steps_layout = QtWidgets.QVBoxLayout(steps_group)
        
        for i, step in enumerate(scenario['steps'], 1):
            step_label = QtWidgets.QLabel(f"{i}. {step}")
            step_label.setWordWrap(True)
            steps_layout.addWidget(step_label)
        
        # Expected results
        results_group = QtWidgets.QGroupBox("Expected Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        
        results_label = QtWidgets.QLabel(scenario['expected_results'])
        results_label.setWordWrap(True)
        results_layout.addWidget(results_label)
        
        # Add groups to main layout
        layout.addWidget(info_group)
        layout.addWidget(steps_group)
        layout.addWidget(results_group)
        
        # Start button
        start_button = QtWidgets.QPushButton("Start Test")
        start_button.clicked.connect(dialog.accept)
        layout.addWidget(start_button)
        
        # Show dialog
        dialog.exec()
    
    def _wait_for_user_completion(self):
        """Wait for the user to complete the scenario."""
        dialog = QtWidgets.QDialog(self.main_window)
        dialog.setWindowTitle("Test in Progress")
        dialog.setMinimumSize(400, 200)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "Please complete the test scenario according to the steps provided.\n"
            "Click 'Complete' when you have finished all steps."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Complete button
        complete_button = QtWidgets.QPushButton("Complete")
        complete_button.clicked.connect(dialog.accept)
        layout.addWidget(complete_button)
        
        # Show dialog
        dialog.exec()
    
    def _collect_feedback(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Collect feedback from the user.
        
        Args:
            scenario: Scenario information
            
        Returns:
            Dictionary with feedback
        """
        dialog = QtWidgets.QDialog(self.main_window)
        dialog.setWindowTitle(f"Feedback: {scenario['name']}")
        dialog.setMinimumSize(600, 400)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "Please provide feedback on your experience with this test scenario."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Metrics
        metrics_group = QtWidgets.QGroupBox("Metrics")
        metrics_layout = QtWidgets.QFormLayout(metrics_group)
        
        metric_widgets = {}
        
        for metric in scenario.get("metrics", []):
            if metric == "user_satisfaction":
                widget = QtWidgets.QSlider(Qt.Orientation.Horizontal)
                widget.setMinimum(1)
                widget.setMaximum(5)
                widget.setValue(3)
                widget.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
                widget.setTickInterval(1)
                
                # Add labels
                labels_layout = QtWidgets.QHBoxLayout()
                labels_layout.addWidget(QtWidgets.QLabel("Poor"))
                labels_layout.addStretch()
                labels_layout.addWidget(QtWidgets.QLabel("Excellent"))
                
                metric_layout = QtWidgets.QVBoxLayout()
                metric_layout.addWidget(widget)
                metric_layout.addLayout(labels_layout)
                
                metrics_layout.addRow("User Satisfaction:", metric_layout)
                
            elif metric == "errors_encountered":
                widget = QtWidgets.QSpinBox()
                widget.setMinimum(0)
                widget.setMaximum(100)
                widget.setValue(0)
                metrics_layout.addRow("Errors Encountered:", widget)
                
            elif metric == "time_to_complete":
                # This is measured automatically
                continue
                
            elif metric == "processing_speed":
                widget = QtWidgets.QComboBox()
                widget.addItems(["Very Slow", "Slow", "Acceptable", "Fast", "Very Fast"])
                widget.setCurrentIndex(2)  # Acceptable
                metrics_layout.addRow("Processing Speed:", widget)
                
            elif metric == "identification_accuracy":
                widget = QtWidgets.QSlider(Qt.Orientation.Horizontal)
                widget.setMinimum(1)
                widget.setMaximum(5)
                widget.setValue(3)
                widget.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
                widget.setTickInterval(1)
                
                # Add labels
                labels_layout = QtWidgets.QHBoxLayout()
                labels_layout.addWidget(QtWidgets.QLabel("Poor"))
                labels_layout.addStretch()
                labels_layout.addWidget(QtWidgets.QLabel("Excellent"))
                
                metric_layout = QtWidgets.QVBoxLayout()
                metric_layout.addWidget(widget)
                metric_layout.addLayout(labels_layout)
                
                metrics_layout.addRow("Identification Accuracy:", metric_layout)
                
            elif metric == "time_to_correct":
                widget = QtWidgets.QComboBox()
                widget.addItems(["Very Difficult", "Difficult", "Moderate", "Easy", "Very Easy"])
                widget.setCurrentIndex(2)  # Moderate
                metrics_layout.addRow("Ease of Correction:", widget)
                
            elif metric == "analysis_quality":
                widget = QtWidgets.QSlider(Qt.Orientation.Horizontal)
                widget.setMinimum(1)
                widget.setMaximum(5)
                widget.setValue(3)
                widget.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
                widget.setTickInterval(1)
                
                # Add labels
                labels_layout = QtWidgets.QHBoxLayout()
                labels_layout.addWidget(QtWidgets.QLabel("Poor"))
                labels_layout.addStretch()
                labels_layout.addWidget(QtWidgets.QLabel("Excellent"))
                
                metric_layout = QtWidgets.QVBoxLayout()
                metric_layout.addWidget(widget)
                metric_layout.addLayout(labels_layout)
                
                metrics_layout.addRow("Analysis Quality:", metric_layout)
                
            elif metric == "keyboard_navigation_score":
                widget = QtWidgets.QSlider(Qt.Orientation.Horizontal)
                widget.setMinimum(1)
                widget.setMaximum(5)
                widget.setValue(3)
                widget.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
                widget.setTickInterval(1)
                
                # Add labels
                labels_layout = QtWidgets.QHBoxLayout()
                labels_layout.addWidget(QtWidgets.QLabel("Poor"))
                labels_layout.addStretch()
                labels_layout.addWidget(QtWidgets.QLabel("Excellent"))
                
                metric_layout = QtWidgets.QVBoxLayout()
                metric_layout.addWidget(widget)
                metric_layout.addLayout(labels_layout)
                
                metrics_layout.addRow("Keyboard Navigation:", metric_layout)
                
            elif metric == "screen_reader_compatibility":
                widget = QtWidgets.QSlider(Qt.Orientation.Horizontal)
                widget.setMinimum(1)
                widget.setMaximum(5)
                widget.setValue(3)
                widget.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
                widget.setTickInterval(1)
                
                # Add labels
                labels_layout = QtWidgets.QHBoxLayout()
                labels_layout.addWidget(QtWidgets.QLabel("Poor"))
                labels_layout.addStretch()
                labels_layout.addWidget(QtWidgets.QLabel("Excellent"))
                
                metric_layout = QtWidgets.QVBoxLayout()
                metric_layout.addWidget(widget)
                metric_layout.addLayout(labels_layout)
                
                metrics_layout.addRow("Screen Reader Compatibility:", metric_layout)
                
            else:
                widget = QtWidgets.QLineEdit()
                metrics_layout.addRow(f"{metric.replace('_', ' ').title()}:", widget)
            
            metric_widgets[metric] = widget
        
        layout.addWidget(metrics_group)
        
        # Comments
        comments_group = QtWidgets.QGroupBox("Comments")
        comments_layout = QtWidgets.QVBoxLayout(comments_group)
        
        comments_label = QtWidgets.QLabel("Please provide any additional comments or feedback:")
        comments_label.setWordWrap(True)
        
        comments_text = QtWidgets.QTextEdit()
        
        comments_layout.addWidget(comments_label)
        comments_layout.addWidget(comments_text)
        
        layout.addWidget(comments_group)
        
        # Submit button
        submit_button = QtWidgets.QPushButton("Submit Feedback")
        submit_button.clicked.connect(dialog.accept)
        layout.addWidget(submit_button)
        
        # Show dialog
        dialog.exec()
        
        # Collect feedback
        feedback = {
            "comments": comments_text.toPlainText()
        }
        
        # Collect metric values
        for metric, widget in metric_widgets.items():
            if isinstance(widget, QtWidgets.QSlider):
                feedback[metric] = widget.value()
            elif isinstance(widget, QtWidgets.QSpinBox):
                feedback[metric] = widget.value()
            elif isinstance(widget, QtWidgets.QComboBox):
                feedback[metric] = widget.currentText()
            elif isinstance(widget, QtWidgets.QLineEdit):
                feedback[metric] = widget.text()
        
        return feedback
    
    def run_all_scenarios(self):
        """Run all test scenarios."""
        if not self.start_application():
            logger.error("Failed to start application for testing")
            return
        
        for scenario in self.test_scenarios:
            self.run_test_scenario(scenario["id"])
        
        # Generate report
        self.generate_report()
        
        # Exit application
        if self.app:
            self.app.quit()
    
    def generate_report(self):
        """Generate a report of the test results."""
        if not self.results:
            logger.warning("No test results to report")
            return
        
        # Create report data
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "scenarios_tested": len(self.results),
            "results": self.results
        }
        
        # Save report to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"uat_report_{timestamp}.json"
        
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"UAT report saved to {report_file}")
            
            # Generate HTML report
            self._generate_html_report(report, timestamp)
            
        except Exception as e:
            logger.error(f"Failed to save UAT report: {str(e)}")
    
    def _generate_html_report(self, report: Dict[str, Any], timestamp: str):
        """Generate an HTML report.
        
        Args:
            report: Report data
            timestamp: Timestamp string
        """
        html_file = self.reports_dir / f"uat_report_{timestamp}.html"
        
        try:
            # Create HTML content
            html = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <title>VSAT User Acceptance Testing Report</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 20px; }",
                "        h1 { color: #333; }",
                "        .scenario { border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px; }",
                "        .scenario h2 { margin-top: 0; color: #444; }",
                "        .metrics { margin: 10px 0; }",
                "        .metric { margin: 5px 0; }",
                "        .comments { margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }",
                "        .summary { margin: 20px 0; padding: 10px; background-color: #f0f0f0; border-radius: 5px; }",
                "    </style>",
                "</head>",
                "<body>",
                "    <h1>VSAT User Acceptance Testing Report</h1>",
                f"    <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                "    <div class='summary'>",
                f"        <p><strong>Scenarios Tested:</strong> {report['scenarios_tested']}</p>",
                "    </div>"
            ]
            
            # Add scenarios
            for scenario_id, result in report['results'].items():
                scenario = result['scenario']
                feedback = result['feedback']
                
                html.extend([
                    "    <div class='scenario'>",
                    f"        <h2>{scenario['name']}</h2>",
                    f"        <p><strong>Description:</strong> {scenario['description']}</p>",
                    f"        <p><strong>Time to Complete:</strong> {result['time_to_complete']:.2f} seconds</p>",
                    "        <div class='metrics'>",
                    "            <h3>Metrics</h3>"
                ])
                
                # Add metrics
                for metric, value in feedback.items():
                    if metric != "comments":
                        html.append(f"            <div class='metric'><strong>{metric.replace('_', ' ').title()}:</strong> {value}</div>")
                
                # Add comments
                html.extend([
                    "        </div>",
                    "        <div class='comments'>",
                    "            <h3>Comments</h3>",
                    f"            <p>{feedback.get('comments', 'No comments')}</p>",
                    "        </div>",
                    "    </div>"
                ])
            
            # Close HTML
            html.extend([
                "</body>",
                "</html>"
            ])
            
            # Write HTML file
            with open(html_file, "w") as f:
                f.write("\n".join(html))
            
            logger.info(f"HTML UAT report saved to {html_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")


def main():
    """Main entry point for UAT."""
    uat = UserAcceptanceTesting()
    uat.run_all_scenarios()


if __name__ == "__main__":
    main() 