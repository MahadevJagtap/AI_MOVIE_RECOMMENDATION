"""
Streamlit App Entrypoint.
Delegates execution to src/ui/app.py.
"""
import runpy
from pathlib import Path

target = Path(__file__).resolve().parent.parent / "src" / "ui" / "app.py"
runpy.run_path(str(target), run_name="__main__")
