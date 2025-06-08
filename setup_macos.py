#!/usr/bin/env python3
"""
macOS Setup Script for Intelligence Analysis POC
Run this script to set up your development environment
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is adequate"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is too old. Please use Python 3.8+")
        return False


def install_requirements():
    """Install Python packages from requirements.txt"""
    print("\n📦 Installing Python packages...")

    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found. Creating it...")
        with open("requirements.txt", "w") as f:
            f.write("""streamlit==1.32.0
spacy==3.7.4
pandas==2.2.1
networkx==3.2.1
plotly==5.19.0
transformers==4.38.2
torch==2.2.1
sentence-transformers==2.6.1
numpy==1.26.4
scikit-learn==1.4.1.post1

# Document processing libraries
PyPDF2==3.0.1
pdfplumber==0.10.0
python-docx==1.1.0
python-pptx==0.6.23
openpyxl==3.1.2""")
        print("✅ requirements.txt created with document processing libraries")

    # Install packages
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

    if success:
        print("✅ Document processing libraries installed:")
        print("   - PDF support: PyPDF2, pdfplumber")
        print("   - Word documents: python-docx")
        print("   - PowerPoint: python-pptx")
        print("   - Excel: openpyxl")

    return success


def download_spacy_model():
    """Download spaCy English language model"""
    print("\n🌐 Downloading spaCy English language model...")
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading spaCy English model"
    )


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    directories = [
        "src",
        "data",
        "data/sample_reports",
        "tests",
        "logs"
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📂 Directory already exists: {directory}")

    return True


def create_sample_data():
    """Create sample intelligence reports"""
    print("\n📄 Creating sample intelligence reports...")

    sample_reports = {
        "security_report_001.txt": """INTELLIGENCE REPORT - OPERATION NIGHTFALL
Date: 2025-06-01
Classification: SECRET

Subject: Suspicious Financial Activity - CRIMSON ENTERPRISES

Our financial intelligence unit has identified irregular transactions involving CRIMSON ENTERPRISES, 
a shell company registered in CAYMAN ISLANDS. Analysis shows $15.7 million transferred through 
SWISS NATIONAL BANK to accounts linked to VLADIMIR PETROV.

PETROV, a known associate of the EASTERN SYNDICATE, was previously flagged in Operation THUNDER. 
Cross-reference with INTERPOL database shows connections to money laundering operations in 
AMSTERDAM and DUBAI.

Recommend immediate surveillance of PETROV's known addresses in MONACO and LONDON.
""",

        "economic_report_001.txt": """ECONOMIC INTELLIGENCE BRIEFING
Date: 2025-06-03
Classification: CONFIDENTIAL

Subject: Market Manipulation - TECH SECTOR

Intelligence indicates coordinated effort to manipulate QUANTUM TECHNOLOGIES stock price. 
Key players include HEDGE FUND ALPHA and offshore entity DIGITAL VENTURES LLC.

Timeline analysis shows suspicious trading patterns preceding announcement of QUANTUM's 
merger with BERLIN-based EUROPA SYSTEMS. Total market impact estimated at $2.3 billion.

Sources within WALL STREET suggest involvement of MARCUS WEBB, former SEC investigator 
now working as consultant for APEX CAPITAL GROUP.
""",

        "threat_assessment_001.txt": """THREAT ASSESSMENT REPORT
Date: 2025-06-05
Classification: TOP SECRET

Subject: Cyber Threat Analysis - OPERATION DIGITAL STORM

Advanced persistent threat group APT-COBRA has intensified activities targeting 
CRITICAL INFRASTRUCTURE in NORTH AMERICA and EUROPE. 

Recent attacks traced to command and control servers in EASTERN EUROPE, 
specifically BUCHAREST and SOFIA data centers operated by DARK NET HOSTING.

Target profile includes POWER GRID operators and FINANCIAL INSTITUTIONS including GLOBAL TRUST BANK.
Attribution analysis suggests state-sponsored activity linked to FOREIGN ACTOR X.
"""
    }

    for filename, content in sample_reports.items():
        filepath = os.path.join("data", "sample_reports", filename)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"✅ Created sample report: {filepath}")

    return True


def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests to verify setup...")
    return run_command(
        f"{sys.executable} tests/test_processor.py",
        "Running test suite"
    )


def main():
    """Main setup function"""
    print("🚀 Intelligence Analysis POC Setup for macOS")
    print("=" * 60)

    setup_steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing Python packages", install_requirements),
        ("Downloading spaCy model", download_spacy_model),
        ("Creating sample data", create_sample_data),
    ]

    success_count = 0
    for step_name, step_function in setup_steps:
        print(f"\n🔸 {step_name}")
        if step_function():
            success_count += 1
        else:
            print(f"❌ Setup failed at: {step_name}")
            break

    print("\n" + "=" * 60)

    if success_count == len(setup_steps):
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run tests: python tests/test_processor.py")
        print("2. Start dashboard: python run_dashboard.py")
        print("3. Open your browser to: http://localhost:8501")
        print("\n💡 PyCharm users:")
        print("- Right-click run_dashboard.py → Run 'run_dashboard'")
        print("- Or use PyCharm's terminal: python run_dashboard.py")
    else:
        print("❌ Setup incomplete. Please check the error messages above.")
        print("💡 You might need to:")
        print("- Update Python to 3.8+")
        print("- Check your internet connection")
        print("- Run: pip install --upgrade pip")


if __name__ == "__main__":
    main()