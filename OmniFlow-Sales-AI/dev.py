#!/usr/bin/env python
"""OmniFlow Development Server Launcher"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def run_api():
    """Start FastAPI backend"""
    print("\n🚀 Starting FastAPI Backend (port 8000)...\n")
    subprocess.run([
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--app-dir",
        str(PROJECT_ROOT),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ], cwd=PROJECT_ROOT)

def run_web():
    """Start Django frontend"""
    print("\n🚀 Starting Django Frontend (port 8080)...\n")
    subprocess.run([sys.executable, "manage.py", "runserver", "0.0.0.0:8080"], cwd=PROJECT_ROOT / "django_app")

def run_train():
    """Train the model"""
    print("\n🚀 Training model...\n")
    subprocess.run([sys.executable, "-c", "from src.training import train_model; print(train_model())"], cwd=PROJECT_ROOT)

def run_all():
    """Start both servers"""
    print("\n" + "="*70)
    print("🚀 OmniFlow - Starting Frontend & Backend Servers")
    print("="*70 + "\n")
    
    print("[1/2] Starting FastAPI Backend (port 8000)...")
    subprocess.Popen([
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--app-dir",
        str(PROJECT_ROOT),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ], cwd=PROJECT_ROOT)
    time.sleep(2)
    
    print("[2/2] Starting Django Frontend (port 8080)...")
    subprocess.Popen([sys.executable, "manage.py", "runserver", "0.0.0.0:8080"], cwd=PROJECT_ROOT / "django_app")
    
    print("\n" + "="*70)
    print("✅ Both servers started successfully!")
    print("="*70 + "\n")
    print("🌐 Access the application:")
    print("   • Web Dashboard:    http://localhost:8080")
    print("   • API Docs:         http://localhost:8000/docs")
    print("   • API ReDoc:        http://localhost:8000/redoc")
    print("\n📊 Available Pages:")
    print("   • Dashboard:        http://localhost:8080/")
    print("   • Train Model:      http://localhost:8080/upload/")
    print("   • Make Prediction:  http://localhost:8080/predict/")
    print("   • Drift & Retrain:  http://localhost:8080/drift/")
    print("\n" + "="*70)
    print("Press Ctrl+C in each terminal window to stop servers")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="OmniFlow Development Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dev.py run-all       Start both FastAPI and Django servers
  python dev.py run-api       Start only FastAPI backend
  python dev.py run-web       Start only Django frontend
  python dev.py run-train     Train the initial ML model
        """
    )
    
    parser.add_argument(
        "command",
        choices=["run-all", "run-api", "run-web", "run-train"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "run-all":
            run_all()
        elif args.command == "run-api":
            run_api()
        elif args.command == "run-web":
            run_web()
        elif args.command == "run-train":
            run_train()
    except KeyboardInterrupt:
        print("\n\n✅ Servers stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
