###############################################################################
# PROGRAM PURPOSE
# -----------------------------------------------------------------------------
# This program discovers the network interfaces available on the local machine.
#
# For each interface, it reports:
#
#     - Interface name
#     - MAC address
#     - IPv4 address
#     - IPv6 address
#
# The information is:
#
#     1. Printed on the screen.
#     2. Saved to network_interfaces.txt.
#
# The output file is created in the current working directory. This is normally
# the directory from which you run:
#
#     python network_interfaces.py
#
#
# HOW TO USE
# -----------------------------------------------------------------------------
# 1. Save this code in a Python file, for example:
#
#        network_interfaces.py
#
# 2. Open Command Prompt, PowerShell, or a terminal.
#
# 3. Change to the directory containing the script:
#
#        cd path_to_script
#
# 4. Run:
#
#        python network_interfaces.py
#
# 5. The program prints the interface information and creates:
#
#        network_interfaces.txt
#
# The script automatically installs the psutil package if it is not installed.
###############################################################################

import os
import socket
import subprocess
import sys
from typing import Any


def import_or_install_psutil() -> Any:
    """
    Import psutil or install it automatically when it is missing.

    Returns
    -------
    module
        The imported psutil module.
    """

    try:
        import psutil

        return psutil

    except ImportError:
        print("The required package 'psutil' is not installed.")
        print("Installing psutil automatically...")

        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "psutil",
                ]
            )

        except subprocess.CalledProcessError as error:
            print("ERROR: Automatic psutil installation failed.")
            print("Install it manually using:")
            print(f'"{sys.executable}" -m pip install psutil')

            raise RuntimeError(
                "Unable to install the required psutil package."
            ) from error

        print("psutil was installed successfully.")

        import psutil

        return psutil


# Import psutil or install it before continuing.
psutil = import_or_install_psutil()


def get_local_network_interfaces() -> list[dict[str, Any]]:
    """
    Return all local network interfaces and their addresses.

    Returns
    -------
    list[dict]
        A list containing one dictionary for each interface.

    Dictionary format
    -----------------
    {
        "interface": "Ethernet",
        "mac": "00-11-22-33-44-55",
        "ipv4": ["192.168.1.25"],
        "ipv6": ["fe80::1234:5678"]
    }
    """

    interface_list: list[dict[str, Any]] = []

    # Retrieve all interfaces and all assigned addresses.
    network_interfaces = psutil.net_if_addrs()

    for interface_name, addresses in network_interfaces.items():

        interface_information: dict[str, Any] = {
            "interface": interface_name,
            "mac": "",
            "ipv4": [],
            "ipv6": [],
        }

        for address in addresses:

            # Detect IPv4 addresses.
            if address.family == socket.AF_INET:
                interface_information["ipv4"].append(address.address)

            # Detect IPv6 addresses.
            elif address.family == socket.AF_INET6:

                # On some systems, IPv6 addresses contain a scope identifier,
                # such as:
                #
                #     fe80::1234:5678%12
                #
                # The "%12" section is removed for cleaner output.
                ipv6_address = address.address.split("%")[0]

                interface_information["ipv6"].append(ipv6_address)

            # Detect the physical MAC address.
            elif address.family == psutil.AF_LINK:
                interface_information["mac"] = address.address

        interface_list.append(interface_information)

    return interface_list


def build_interface_report(
    interfaces: list[dict[str, Any]],
) -> str:
    """
    Convert the interface information into a formatted text report.

    Parameters
    ----------
    interfaces:
        Interface information returned by
        get_local_network_interfaces().

    Returns
    -------
    str
        Complete formatted report.
    """

    report_lines: list[str] = []

    report_lines.append("=" * 80)
    report_lines.append("LOCAL NETWORK INTERFACES AND IP ADDRESSES")
    report_lines.append("=" * 80)
    report_lines.append("")

    if not interfaces:
        report_lines.append("No network interfaces were detected.")

    for index, interface in enumerate(interfaces, start=1):

        report_lines.append(
            f"[{index}] Interface: {interface['interface']}"
        )

        mac_address = interface["mac"] or "Not available"

        report_lines.append(
            f"    MAC Address : {mac_address}"
        )

        if interface["ipv4"]:
            report_lines.append("    IPv4 Addresses:")

            for ipv4_address in interface["ipv4"]:
                report_lines.append(
                    f"        {ipv4_address}"
                )
        else:
            report_lines.append(
                "    IPv4 Addresses: None"
            )

        if interface["ipv6"]:
            report_lines.append("    IPv6 Addresses:")

            for ipv6_address in interface["ipv6"]:
                report_lines.append(
                    f"        {ipv6_address}"
                )
        else:
            report_lines.append(
                "    IPv6 Addresses: None"
            )

        report_lines.append("")

    report_lines.append("=" * 80)

    return "\n".join(report_lines)


def print_and_save_local_network_interfaces(
    filename: str = "network_interfaces.txt",
) -> str:
    """
    Print all network interfaces and save the same report to a file.

    The output file is stored in the current working directory.

    Parameters
    ----------
    filename:
        Name of the output file. The default is:

            network_interfaces.txt

    Returns
    -------
    str
        Full path of the created output file.
    """

    print("Detecting local network interfaces...")
    print()

    interfaces = get_local_network_interfaces()

    report = build_interface_report(interfaces)

    # Print the complete report to the screen.
    print(report, flush=True)

    # os.getcwd() returns the directory from which the Python program was run.
    current_directory = os.getcwd()

    output_file = os.path.join(
        current_directory,
        filename,
    )

    try:
        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(report)
            file.write("\n")

    except OSError as error:
        print()
        print("ERROR: The report could not be saved.")
        print(f"Reason: {error}")

        raise

    print()
    print("Interface report saved successfully:")
    print(output_file, flush=True)

    return output_file


def main() -> None:
    """
    Main program entry point.

    This function must be called for the interface discovery to run.
    """

    try:
        print_and_save_local_network_interfaces()

    except Exception as error:
        print()
        print("The program stopped because of an error:")
        print(error)

        sys.exit(1)


# This section actually starts the program.
#
# Without this call, Python only defines the functions and nothing will be
# printed or saved.
if __name__ == "__main__":
    main()