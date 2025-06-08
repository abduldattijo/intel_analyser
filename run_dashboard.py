#!/usr/bin/env python3
"""
PyCharm Run Configuration for Intelligence Analysis Dashboard
Run this file to start the Streamlit dashboard
"""

import subprocess
import sys
import os


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, 'src', 'intel_dashboard.py')

    # Check if the dashboard file exists
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard file not found at: {dashboard_path}")
        print("Make sure intel_dashboard.py is in the src/ directory")
        return

    # Run Streamlit
    print("🚀 Starting Intelligence Analysis Dashboard...")
    print(f"📂 Dashboard location: {dashboard_path}")
    print("🌐 Dashboard will open in your default browser")
    print("⏹️  Press Ctrl+C in this terminal to stop the dashboard")
    print("-" * 50)

    try:
        # Run streamlit with the dashboard file
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            dashboard_path,
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")


if __name__ == "__main__":
    main()