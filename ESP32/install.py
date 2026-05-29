#!/usr/bin/env python3

# =====================================================================
# Script Name:
#     dependency_installer.py
#
# Description:
#     This script automatically verifies that required Python packages
#     exist on the system before running an application.
#
#     The script performs the following operations:
#
#     1. Checks whether each required package is already installed.
#
#     2. If the package exists:
#           - Displays confirmation message
#           - Displays installed version number
#
#     3. If package is missing:
#           - Automatically installs package using pip
#           - Verifies successful installation
#           - Displays installed version
#
#     4. If installation fails:
#           - Shows error information
#           - Continues processing remaining packages
#
#     5. Generates final summary:
#           - Total verified packages
#           - Total installed packages
#           - Failed packages
#
# Features:
#
#     • Automatic package verification
#     • Automatic package installation
#     • Version reporting
#     • Error handling
#     • User-friendly messages
#     • Continue operation after failures
#     • Upgrade packages during installation
#
# Example Output:
#
#     Checking numpy...
#     [OK] numpy already installed (Version 2.3.1)
#
#     Checking cv2...
#     [MISSING] cv2 not installed
#     Installing opencv-python...
#     [SUCCESS] Installed opencv-python
#
# Supported Package Mapping:
#
#     Import Name          Pip Package
#     ------------------------------------------
#     cv2                  opencv-python
#     PIL                  pillow
#     dotenv               python-dotenv
#     yaml                 pyyaml
#     serial               pyserial
#     sklearn              scikit-learn
#
# Usage:
#
#     python dependency_installer.py
#
# Requirements:
#
#     Python 3.8+
#     pip available in PATH
#     Internet connection for missing packages
#
# =====================================================================

import subprocess
import sys
import importlib

# Used to determine installed version numbers
from importlib.metadata import (
    version,
    PackageNotFoundError
)

# ============================================================
# Required package list
#
# Left side:
#     Import module name
#
# Right side:
#     pip installation package name
#
# Example:
#
#     cv2 -> opencv-python
#
# ============================================================

REQUIRED_PACKAGES = {

    "numpy": "numpy",

    "cv2": "opencv-python",

    "torch": "torch",

    "torchvision": "torchvision",

    "PIL": "pillow",

    "dotenv": "python-dotenv",

    "requests": "requests",

    "yaml": "pyyaml",

    "serial": "pyserial",

    "matplotlib": "matplotlib",

    "pandas": "pandas",

    "sklearn": "scikit-learn"
}


# ============================================================
# Function:
#     get_package_version()
#
# Purpose:
#     Returns installed package version.
#
# Input:
#     pip_name
#
# Output:
#     Version string
#
# ============================================================

def get_package_version(pip_name):

    try:

        return version(pip_name)

    except PackageNotFoundError:

        return "Unknown"


# ============================================================
# Function:
#     is_installed()
#
# Purpose:
#     Check if module can be imported.
#
# Returns:
#     True  -> installed
#     False -> missing
#
# ============================================================

def is_installed(import_name):

    try:

        importlib.import_module(import_name)

        return True

    except ImportError:

        return False


# ============================================================
# Function:
#     install_package()
#
# Purpose:
#     Install missing package automatically.
#
# Uses:
#     python -m pip install --upgrade
#
# ============================================================

def install_package(pip_name):

    try:

        print(
            f"\n[INFO] "
            f"Installing {pip_name}..."
        )

        subprocess.check_call(

            [
                sys.executable,

                "-m",

                "pip",

                "install",

                "--upgrade",

                pip_name
            ]

        )

        print(
            f"[SUCCESS] "
            f"{pip_name} installed successfully"
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] "
            f"Failed installing {pip_name}"
        )

        print(e)

        return False


# ============================================================
# Function:
#     verify_package()
#
# Purpose:
#     Verify package exists.
#
# Logic:
#
#     If installed:
#         Show version
#
#     Else:
#         Install package
#         Verify installation
#
# ============================================================

def verify_package(

        import_name,

        pip_name

):

    print("\n")

    print("=" * 60)

    print(

        f"Checking package: "

        f"{import_name}"

    )

    print("=" * 60)

    if is_installed(import_name):

        installed_version = (

            get_package_version(

                pip_name

            )

        )

        print(

            f"[OK] "

            f"{import_name} "

            f"already installed "

            f"(Version "

            f"{installed_version})"

        )

        return True

    print(

        f"[MISSING] "

        f"{import_name} "

        f"not installed"

    )

    success = install_package(

        pip_name

    )

    if not success:

        return False

    if is_installed(import_name):

        installed_version = (

            get_package_version(

                pip_name

            )

        )

        print(

            f"[VERIFIED] "

            f"{import_name} "

            f"installed successfully "

            f"(Version "

            f"{installed_version})"

        )

        return True

    print(

        f"[ERROR] "

        f"Installation verification failed"

    )

    return False


# ============================================================
# Main execution routine
#
# ============================================================

def main():

    print()

    print("#" * 70)

    print(

        " PYTHON DEPENDENCY "

        "VERIFICATION AND INSTALLER "

    )

    print("#" * 70)

    print()

    passed = 0

    failed = 0

    for (

        import_name,

        pip_name

    ) in REQUIRED_PACKAGES.items():

        result = verify_package(

            import_name,

            pip_name

        )

        if result:

            passed += 1

        else:

            failed += 1

    print()

    print("#" * 70)

    print("FINAL SUMMARY")

    print("#" * 70)

    print()

    print(

        f"Packages Ready : "

        f"{passed}"

    )

    print(

        f"Packages Failed: "

        f"{failed}"

    )

    print()

    if failed == 0:

        print(

            "[SUCCESS] "

            "All packages ready."

        )

    else:

        print(

            "[WARNING] "

            "Some packages failed."

        )

    print()

    print("#" * 70)


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":

    main()