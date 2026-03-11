import argparse
import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def main():
    parser = argparse.ArgumentParser(description="Kinetica")
    parser.add_argument(
        "--ui",
        choices=["cli", "dashboard"],
        default="dashboard",
        help="Selecciona la interfaz de usuario a lanzar",
    )
    args = parser.parse_args()

    if args.ui == "cli":
        from kinetica.ui.cli import run_demo
        run_demo()
    else:
        dashboard_path = os.path.join(SRC_DIR, "kinetica", "ui", "dashboard.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path], check=True)


if __name__ == "__main__":
    main()