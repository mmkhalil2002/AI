"""Install Python dependencies for the medical MCP client and servers."""
import subprocess
import sys


def config():
    """Install project dependencies using the active Python interpreter."""
    packages = ['mcp[cli]', 'openai', 'requests', 'python-dotenv', 'Pillow']
    subprocess.run([sys.executable, '-m', 'pip', 'install', *packages], check=True)
    subprocess.run([sys.executable, '-c', 'import mcp, openai, requests, dotenv, PIL'], check=True)
    print('Medical MCP dependencies installed successfully.')


if __name__ == '__main__':
    try:
        config()
    except subprocess.CalledProcessError as exc:
        print(f'Installation failed: {exc}', file=sys.stderr)
        sys.exit(1)
