# User Acceptance Testing (UAT) Framework for VSAT

This directory contains the User Acceptance Testing (UAT) framework for the Voice Separation & Analysis Tool (VSAT). The UAT framework is designed to facilitate structured testing of the application from an end-user perspective, collecting both quantitative metrics and qualitative feedback.

## Overview

The UAT framework consists of:

1. **Test Scenarios**: Predefined testing scenarios covering all major features of the application
2. **Testing Framework**: Python code to run the tests and collect results
3. **Reporting System**: Tools to generate reports from test results

## Getting Started

### Prerequisites

- VSAT application installed and configured
- Python 3.8 or higher
- PyQt6 installed

### Running the UAT

To run the User Acceptance Testing:

```bash
python tests/user_acceptance_testing.py
```

This will:
1. Start the VSAT application in UAT mode
2. Load the test scenarios
3. Guide you through each test scenario
4. Collect metrics and feedback
5. Generate reports in the `uat_reports` directory

## Test Scenarios

Test scenarios are defined in `tests/uat_scenarios.json`. Each scenario includes:

- **ID**: Unique identifier for the scenario
- **Name**: Descriptive name
- **Description**: Detailed description of what the scenario tests
- **Steps**: Step-by-step instructions for completing the test
- **Expected Results**: What should happen if the test is successful
- **Metrics**: What metrics to collect for this scenario

The default scenarios cover:

1. **Basic Audio Processing Workflow**: Testing the core functionality
2. **Batch Processing**: Testing the batch processing feature
3. **Speaker Identification**: Testing speaker identification across recordings
4. **Content Analysis**: Testing the content analysis features
5. **Accessibility Features**: Testing accessibility compliance
6. **Error Handling**: Testing how the application handles errors
7. **Performance Optimization**: Testing performance features
8. **Database Integration**: Testing database operations
9. **Export and Import**: Testing data export and import
10. **UI Responsiveness**: Testing UI behavior during intensive operations

## Adding New Test Scenarios

To add a new test scenario:

1. Edit `tests/uat_scenarios.json`
2. Add a new JSON object with the required fields
3. Run the UAT to test your new scenario

Example:

```json
{
  "id": "new_feature_test",
  "name": "New Feature Test",
  "description": "Test the new feature XYZ",
  "steps": [
    "Step 1: Do this",
    "Step 2: Do that",
    "Step 3: Verify result"
  ],
  "expected_results": "The new feature should work correctly",
  "metrics": ["user_satisfaction", "errors_encountered"]
}
```

## Reports

The UAT framework generates two types of reports:

1. **JSON Reports**: Raw data in JSON format
2. **HTML Reports**: User-friendly HTML reports with formatting

Reports are saved in the `uat_reports` directory with timestamps in the filename.

## Metrics

The UAT framework collects various metrics:

- **Time to Complete**: How long it takes to complete the scenario
- **Errors Encountered**: Number of errors encountered
- **User Satisfaction**: Subjective rating of satisfaction
- **Processing Speed**: Rating of processing speed
- **Identification Accuracy**: Rating of speaker identification accuracy
- **Analysis Quality**: Rating of content analysis quality
- **Keyboard Navigation Score**: Rating of keyboard navigation
- **Screen Reader Compatibility**: Rating of screen reader compatibility
- **Error Clarity**: Rating of error message clarity
- **Recovery Effectiveness**: Rating of error recovery effectiveness
- **Data Integrity**: Rating of data integrity
- **Format Compatibility**: Rating of format compatibility
- **UI Responsiveness**: Rating of UI responsiveness

## Customizing the UAT Framework

The UAT framework can be customized by:

1. Modifying `tests/user_acceptance_testing.py` to change the testing workflow
2. Editing `tests/uat_scenarios.json` to change the test scenarios
3. Adding new metrics to the feedback collection dialog

## Best Practices

When conducting UAT:

1. **Be Thorough**: Follow all steps in each scenario
2. **Be Objective**: Rate metrics based on actual experience, not expectations
3. **Provide Detailed Feedback**: Include specific details in comments
4. **Test Edge Cases**: Try unexpected inputs and actions
5. **Test Accessibility**: Verify accessibility features work as expected

## Troubleshooting

If you encounter issues with the UAT framework:

- Check the console output for error messages
- Verify that the VSAT application is properly installed
- Ensure all dependencies are installed
- Check that the `uat_reports` directory is writable

## Contributing

Contributions to the UAT framework are welcome. Please follow these steps:

1. Create a new branch for your changes
2. Make your changes
3. Test your changes
4. Submit a pull request

## License

The UAT framework is licensed under the same license as the VSAT application. 