#!/usr/bin/env python3
"""
QLCCE Standalone Launcher
Starts both backend and frontend servers in a self-contained manner.
"""

import os
import sys
import subprocess
import signal
import time
import threading
import shutil
from pathlib import Path

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """Print the QLCCE banner"""
    banner = f"""
{Colors.HEADER}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ██████╗ ██╗      ██████╗ ██████╗███████╗                       ║
║  ██╔═══██╗██║     ██╔════╝██╔════╝██╔════╝                       ║
║  ██║   ██║██║     ██║     ██║     █████╗                         ║
║  ██║▄▄ ██║██║     ██║     ██║     ██╔══╝                         ║
║  ╚██████╔╝███████╗╚██████╗╚██████╗███████╗                       ║
║   ╚══▀▀═╝ ╚══════╝ ╚═════╝ ╚═════╝╚══════╝                       ║
║                                                                   ║
║   Quantum-Logarithmic Chaotic Constraint Engine                   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.END}
"""
    print(banner)

def check_python_version():
    """Ensure Python 3.9+ is being used"""
    if sys.version_info < (3, 9):
        print(f"{Colors.RED}Error: Python 3.9 or higher is required{Colors.END}")
        sys.exit(1)

def check_node():
    """Check if Node.js is installed"""
    if not shutil.which('node'):
        print(f"{Colors.RED}Error: Node.js is not installed. Please install Node.js 18+ to run the frontend.{Colors.END}")
        return False
    return True

def check_npm():
    """Check if npm is installed"""
    if not shutil.which('npm'):
        print(f"{Colors.RED}Error: npm is not installed. Please install npm to run the frontend.{Colors.END}")
        return False
    return True

def get_script_dir():
    """Get the directory containing this script"""
    return Path(__file__).parent.absolute()

def install_backend_deps(backend_dir):
    """Install backend Python dependencies"""
    print(f"{Colors.BLUE}Installing backend dependencies...{Colors.END}")
    requirements_file = backend_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"{Colors.RED}Error: requirements.txt not found in {backend_dir}{Colors.END}")
        return False
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"],
            check=True,
            capture_output=True
        )
        print(f"{Colors.GREEN}✓ Backend dependencies installed{Colors.END}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error installing backend dependencies: {e.stderr.decode()}{Colors.END}")
        return False

def install_frontend_deps(frontend_dir):
    """Install frontend Node.js dependencies"""
    print(f"{Colors.BLUE}Installing frontend dependencies...{Colors.END}")
    
    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print(f"{Colors.RED}Error: package.json not found in {frontend_dir}{Colors.END}")
        return False
    
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        print(f"{Colors.GREEN}✓ Frontend dependencies already installed{Colors.END}")
        return True
    
    try:
        subprocess.run(
            ["npm", "install"],
            cwd=frontend_dir,
            check=True,
            capture_output=True
        )
        print(f"{Colors.GREEN}✓ Frontend dependencies installed{Colors.END}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error installing frontend dependencies: {e.stderr.decode()}{Colors.END}")
        return False

class ProcessManager:
    """Manages backend and frontend processes"""
    
    def __init__(self):
        self.processes = []
        self.running = True
        
    def start_backend(self, backend_dir, port=8000):
        """Start the FastAPI backend server"""
        print(f"{Colors.BLUE}Starting backend server on port {port}...{Colors.END}")
        
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(port)],
            cwd=backend_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        self.processes.append(('backend', process))
        
        # Start a thread to read output
        def read_output(proc, name):
            for line in iter(proc.stdout.readline, b''):
                if self.running:
                    print(f"{Colors.GREEN}[Backend]{Colors.END} {line.decode().strip()}")
        
        thread = threading.Thread(target=read_output, args=(process, 'backend'), daemon=True)
        thread.start()
        
        return process
    
    def start_frontend(self, frontend_dir, port=5173):
        """Start the React frontend dev server"""
        print(f"{Colors.BLUE}Starting frontend server on port {port}...{Colors.END}")
        
        process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(port)],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        self.processes.append(('frontend', process))
        
        # Start a thread to read output
        def read_output(proc, name):
            for line in iter(proc.stdout.readline, b''):
                if self.running:
                    print(f"{Colors.YELLOW}[Frontend]{Colors.END} {line.decode().strip()}")
        
        thread = threading.Thread(target=read_output, args=(process, 'frontend'), daemon=True)
        thread.start()
        
        return process
    
    def stop_all(self):
        """Stop all running processes"""
        self.running = False
        print(f"\n{Colors.YELLOW}Shutting down QLCCE...{Colors.END}")
        
        for name, process in self.processes:
            if process.poll() is None:  # Process is still running
                print(f"{Colors.BLUE}Stopping {name}...{Colors.END}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print(f"{Colors.GREEN}QLCCE shutdown complete{Colors.END}")
    
    def wait(self):
        """Wait for all processes to complete"""
        try:
            while self.running:
                # Check if any process has died
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"{Colors.RED}{name} process exited unexpectedly{Colors.END}")
                        self.stop_all()
                        return
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_all()

def main():
    """Main entry point"""
    print_banner()
    check_python_version()
    
    script_dir = get_script_dir()
    backend_dir = script_dir / "backend"
    frontend_dir = script_dir / "frontend"
    
    # Verify directories exist
    if not backend_dir.exists():
        print(f"{Colors.RED}Error: Backend directory not found at {backend_dir}{Colors.END}")
        sys.exit(1)
    
    if not frontend_dir.exists():
        print(f"{Colors.RED}Error: Frontend directory not found at {frontend_dir}{Colors.END}")
        sys.exit(1)
    
    # Check prerequisites
    if not check_node() or not check_npm():
        print(f"{Colors.YELLOW}Frontend will not be started. You can still access the API.{Colors.END}")
        frontend_available = False
    else:
        frontend_available = True
    
    # Install dependencies
    if not install_backend_deps(backend_dir):
        sys.exit(1)
    
    if frontend_available:
        if not install_frontend_deps(frontend_dir):
            frontend_available = False
    
    # Start services
    manager = ProcessManager()
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start backend
    manager.start_backend(backend_dir, port=8000)
    
    # Give backend time to start
    time.sleep(2)
    
    # Start frontend if available
    if frontend_available:
        manager.start_frontend(frontend_dir, port=5173)
        time.sleep(3)
    
    # Print access information
    print(f"""
{Colors.GREEN}{Colors.BOLD}
════════════════════════════════════════════════════════════════════
  QLCCE is running!
════════════════════════════════════════════════════════════════════

  Frontend:     http://localhost:5173
  Backend API:  http://localhost:8000
  API Docs:     http://localhost:8000/docs

  Press Ctrl+C to stop all services
════════════════════════════════════════════════════════════════════
{Colors.END}
""")
    
    # Wait for processes
    manager.wait()

if __name__ == "__main__":
    main()
