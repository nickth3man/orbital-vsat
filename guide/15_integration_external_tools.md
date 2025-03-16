 # Integration with External Tools

## Overview

Integrating VSAT with external tools extends its functionality and allows it to work within your broader audio processing workflow. This guide covers creating compatible export formats, implementing API endpoints, and developing hooks for other applications to interact with VSAT.

## Prerequisites

Before proceeding with external tool integration, ensure you have:

1. Completed the [Desktop Integration](14_desktop_integration.md) phase
2. Familiarized yourself with VSAT's data structures and formats
3. Identified the external tools you wish to integrate with
4. Basic understanding of API development and data interchange formats

## Implementation Steps

### 1. Developing Compatible Export Formats

Create export modules that convert VSAT data to formats compatible with other audio processing tools.

```python
# src/utils/export_formats.py
import json
import csv
import xml.etree.ElementTree as ET
import yaml
import os
from typing import Dict, List, Any, Optional

class ExportManager:
    """Manages exporting VSAT data to various formats."""
    
    def __init__(self, data_dir: str = None):
        """Initialize the export manager."""
        self.data_dir = data_dir or os.environ.get("VSAT_DATA_DIR", "./data")
        self.export_dir = os.path.join(self.data_dir, "exports")
        os.makedirs(self.export_dir, exist_ok=True)
    
    def export_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """Export data to JSON format."""
        filepath = os.path.join(self.export_dir, f"{filename}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return filepath
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: str) -> str:
        """Export data to CSV format."""
        filepath = os.path.join(self.export_dir, f"{filename}.csv")
        if not data:
            return None
            
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return filepath
    
    def export_to_xml(self, data: Dict[str, Any], filename: str) -> str:
        """Export data to XML format."""
        filepath = os.path.join(self.export_dir, f"{filename}.xml")
        
        def dict_to_xml(parent, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    element = ET.SubElement(parent, key)
                    dict_to_xml(element, value)
                elif isinstance(value, list):
                    for item in value:
                        element = ET.SubElement(parent, key)
                        if isinstance(item, dict):
                            dict_to_xml(element, item)
                        else:
                            element.text = str(item)
                else:
                    element = ET.SubElement(parent, key)
                    element.text = str(value)
        
        root = ET.Element("VSATData")
        dict_to_xml(root, data)
        tree = ET.ElementTree(root)
        tree.write(filepath)
        return filepath
    
    def export_to_yaml(self, data: Dict[str, Any], filename: str) -> str:
        """Export data to YAML format."""
        filepath = os.path.join(self.export_dir, f"{filename}.yaml")
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        return filepath
    
    def export_to_praat_textgrid(self, segments: List[Dict], filename: str) -> str:
        """Export transcription segments to Praat TextGrid format."""
        filepath = os.path.join(self.export_dir, f"{filename}.TextGrid")
        
        with open(filepath, 'w') as f:
            f.write('File type = "ooTextFile"\n')
            f.write('Object class = "TextGrid"\n\n')
            
            # Find the total duration from the last segment end time
            if segments:
                xmax = max(segment.get('end_time', 0) for segment in segments)
            else:
                xmax = 0
                
            f.write(f'xmin = 0\n')
            f.write(f'xmax = {xmax}\n')
            f.write('tiers? <exists>\n')
            f.write('size = 1\n')
            f.write('item []:\n')
            f.write('    item [1]:\n')
            f.write('        class = "IntervalTier"\n')
            f.write('        name = "transcription"\n')
            f.write('        xmin = 0\n')
            f.write(f'        xmax = {xmax}\n')
            f.write(f'        intervals: size = {len(segments)}\n')
            
            for i, segment in enumerate(segments, 1):
                f.write(f'        intervals [{i}]:\n')
                f.write(f'            xmin = {segment.get("start_time", 0)}\n')
                f.write(f'            xmax = {segment.get("end_time", 0)}\n')
                f.write(f'            text = "{segment.get("text", "")}"\n')
                
        return filepath
    
    def export_to_audacity_labels(self, segments: List[Dict], filename: str) -> str:
        """Export transcription segments to Audacity label format."""
        filepath = os.path.join(self.export_dir, f"{filename}.txt")
        
        with open(filepath, 'w') as f:
            for segment in segments:
                start = segment.get('start_time', 0)
                end = segment.get('end_time', 0)
                text = segment.get('text', '')
                f.write(f'{start}\t{end}\t{text}\n')
                
        return filepath
```

### 2. Implementing API Endpoints

Create a REST API to allow external applications to interact with VSAT.

```python
# src/api/rest_api.py
from flask import Flask, request, jsonify
import os
import sys
import json
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.export_formats import ExportManager
from utils.audio_processor import AudioProcessor
from utils.transcription_manager import TranscriptionManager

app = Flask(__name__)
export_manager = ExportManager()
audio_processor = AudioProcessor()
transcription_manager = TranscriptionManager()

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """API endpoint to check if the service is running."""
    return jsonify({"status": "ok", "version": "1.0.0"})

@app.route('/api/v1/process', methods=['POST'])
def process_audio():
    """API endpoint to process an audio file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    # Save the uploaded file
    temp_dir = os.environ.get("VSAT_TEMP_DIR", "./temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    
    # Process the audio file
    try:
        result = audio_processor.process_file(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/export/<format>', methods=['POST'])
def export_data(format):
    """API endpoint to export data in various formats."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    filename = data.get('filename', 'export')
    
    try:
        if format == 'json':
            filepath = export_manager.export_to_json(data['content'], filename)
        elif format == 'csv':
            filepath = export_manager.export_to_csv(data['content'], filename)
        elif format == 'xml':
            filepath = export_manager.export_to_xml(data['content'], filename)
        elif format == 'yaml':
            filepath = export_manager.export_to_yaml(data['content'], filename)
        elif format == 'praat':
            filepath = export_manager.export_to_praat_textgrid(data['content'], filename)
        elif format == 'audacity':
            filepath = export_manager.export_to_audacity_labels(data['content'], filename)
        else:
            return jsonify({"error": f"Unsupported format: {format}"}), 400
            
        return jsonify({"success": True, "filepath": filepath})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/transcriptions', methods=['GET'])
def get_transcriptions():
    """API endpoint to retrieve available transcriptions."""
    try:
        transcriptions = transcription_manager.list_transcriptions()
        return jsonify(transcriptions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/transcriptions/<id>', methods=['GET'])
def get_transcription(id):
    """API endpoint to retrieve a specific transcription."""
    try:
        transcription = transcription_manager.get_transcription(id)
        if not transcription:
            return jsonify({"error": "Transcription not found"}), 404
        return jsonify(transcription)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_api_server(host='127.0.0.1', port=5000, debug=False):
    """Start the API server."""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_api_server(debug=True)
```

### 3. Creating Hooks for Other Applications

Develop hooks that allow other applications to trigger VSAT functionality.

```python
# src/utils/application_hooks.py
import os
import sys
import json
import subprocess
from typing import Dict, Any, List, Optional, Callable

class ApplicationHook:
    """Base class for application hooks."""
    
    def __init__(self, name: str, description: str):
        """Initialize the application hook."""
        self.name = name
        self.description = description
        
    def execute(self, *args, **kwargs) -> Any:
        """Execute the hook functionality."""
        raise NotImplementedError("Subclasses must implement execute method")

class CommandLineHook(ApplicationHook):
    """Hook for command-line applications."""
    
    def __init__(self, name: str, description: str, command: str):
        """Initialize the command-line hook."""
        super().__init__(name, description)
        self.command = command
        
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the command-line hook."""
        try:
            # Build the command with arguments
            cmd = self.command.format(*args, **kwargs)
            
            # Execute the command
            result = subprocess.run(cmd, shell=True, check=True, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
            
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "return_code": e.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class PythonFunctionHook(ApplicationHook):
    """Hook for Python function calls."""
    
    def __init__(self, name: str, description: str, function: Callable):
        """Initialize the Python function hook."""
        super().__init__(name, description)
        self.function = function
        
    def execute(self, *args, **kwargs) -> Any:
        """Execute the Python function hook."""
        try:
            result = self.function(*args, **kwargs)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class HookManager:
    """Manages application hooks."""
    
    def __init__(self):
        """Initialize the hook manager."""
        self.hooks = {}
        
    def register_hook(self, hook: ApplicationHook) -> None:
        """Register a hook with the manager."""
        self.hooks[hook.name] = hook
        
    def unregister_hook(self, name: str) -> bool:
        """Unregister a hook from the manager."""
        if name in self.hooks:
            del self.hooks[name]
            return True
        return False
        
    def get_hook(self, name: str) -> Optional[ApplicationHook]:
        """Get a hook by name."""
        return self.hooks.get(name)
        
    def list_hooks(self) -> List[Dict[str, str]]:
        """List all registered hooks."""
        return [{"name": name, "description": hook.description} 
                for name, hook in self.hooks.items()]
                
    def execute_hook(self, name: str, *args, **kwargs) -> Any:
        """Execute a hook by name."""
        hook = self.get_hook(name)
        if not hook:
            raise ValueError(f"Hook not found: {name}")
        return hook.execute(*args, **kwargs)
```

## Testing External Tool Integration

After implementing external tool integration, it's essential to test each component to ensure proper functionality.

### Testing Export Formats

1. Generate sample data in VSAT
2. Export the data to each supported format
3. Verify that the exported files are correctly formatted
4. Import the exported files into the target applications to confirm compatibility

### Testing API Endpoints

1. Use a tool like Postman or curl to test each API endpoint
2. Verify that the endpoints return the expected responses for valid inputs
3. Test error handling by providing invalid inputs
4. Check authentication and authorization if implemented

### Testing Application Hooks

1. Register test hooks with the hook manager
2. Execute each hook with test parameters
3. Verify that the hooks interact correctly with external applications
4. Test error handling and recovery mechanisms

## Next Steps

After completing external tool integration, you should:

1. Proceed to [Personal Usage Monitoring](16_personal_usage_monitoring.md) to track and analyze your VSAT usage
2. Document the API endpoints and hooks for future reference
3. Consider developing additional export formats based on your workflow needs

By completing this phase, you've extended VSAT's functionality and integrated it with your broader audio processing workflow, making it a more versatile and powerful tool.

## References

- `src/utils/export_formats.py` - Export format utility module
- `src/api/rest_api.py` - REST API implementation
- `src/utils/application_hooks.py` - Application hooks utility module
- [Flask Documentation](https://flask.palletsprojects.com/) - Web framework used for the REST API
- [REST API Best Practices](https://restfulapi.net/) - Guide for designing RESTful APIs
- [Python API Integrations Tutorial](https://zato.io/en/tutorials/main/01.html) - Guide for integrating Python applications with external services
- [Praat TextGrid Format](https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html) - Documentation for Praat TextGrid format
- [Audacity Label Track Format](https://manual.audacityteam.org/man/label_tracks.html) - Documentation for Audacity label track format