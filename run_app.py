import os
import subprocess
import sys

# Optional: Change this path to your project directory if needed
project_path = os.path.dirname(os.path.abspath(__file__))

# Path to your Streamlit app
app_path = os.path.join(project_path, "app.py")

# Launch the Streamlit app
subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
