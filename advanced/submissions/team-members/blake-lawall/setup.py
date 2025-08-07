#!/usr/bin/env python3
"""
Setup script for GlucoTrack Advanced Track - Blake Lawall
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    
    return result


def check_uv_installed() -> bool:
    """Check if uv is installed"""
    try:
        result = run_command("uv --version", check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_uv():
    """Install uv if not already installed"""
    if check_uv_installed():
        print("uv is already installed")
        return
    
    print("Installing uv...")
    
    if sys.platform == "win32":
        # Windows
        run_command("powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
    else:
        # macOS/Linux
        run_command("curl -LsSf https://astral.sh/uv/install.sh | sh")
    
    print("uv installed successfully!")


def setup_environment():
    """Set up the project environment"""
    print("Setting up GlucoTrack Advanced Track environment...")
    
    # Get the blake-lawall directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if uv is installed
    if not check_uv_installed():
        install_uv()
    
    # Create virtual environment
    print("Creating virtual environment...")
    run_command("uv venv .venv")
    
    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        activate_cmd = ".venv\\Scripts\\activate"
    else:
        activate_cmd = "source .venv/bin/activate"
    
    print("Installing dependencies...")
    run_command(f"{activate_cmd} && uv pip install -e .")
    
    # Install Jupyter kernel
    print("Installing Jupyter kernel...")
    run_command(f"{activate_cmd} && python -m ipykernel install --user --name=glucotrack-advanced --display-name='GlucoTrack Advanced'")
    
    print("Environment setup complete!")


def generate_notebooks():
    """Generate the Jupyter notebooks"""
    print("Generating Jupyter notebooks...")
    
    # Get the blake-lawall directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Run the notebook generator
    notebook_script = "create_notebooks.py"
    run_command(f"python {notebook_script}")
    
    print("Notebooks generated successfully!")


def main():
    """Main setup function"""
    print("=" * 60)
    print("GlucoTrack Advanced Track - Blake Lawall")
    print("Setup Script")
    print("=" * 60)
    
    try:
        # Setup environment
        setup_environment()
        
        # Generate notebooks
        generate_notebooks()
        
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if sys.platform == "win32":
            print("   .venv\\Scripts\\activate")
        else:
            print("   source .venv/bin/activate")
        print("2. Start Jupyter Lab:")
        print("   jupyter lab")
        print("3. Open the notebooks in the advanced/submissions/team-members/blake-lawall/notebooks/ directory")
        print("\nHappy coding! 🚀")
        
    except Exception as e:
        print(f"\nError during setup: {e}")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main() 