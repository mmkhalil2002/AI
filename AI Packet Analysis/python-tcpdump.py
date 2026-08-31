"""
Python TCPDump-Like Packet Data Collector
=========================================

Program Purpose
---------------
This program captures live network packets using Scapy and saves detailed
packet-level information for later statistical analysis, flow analysis,
traffic classification, and machine-learning model development.

This version does not calculate mean, variance, standard deviation, minimum,
maximum, or median during packet capture. Those measurements should be
calculated later from the generated CSV file.

The program saves:

    1. A PCAP file containing complete captured packets.
    2. A CSV file containing detailed packet metadata.
    3. A summary on the screen showing the number of captured packets,
       captured bytes, capture duration, packet rate, and throughput.

Potential Later Analysis
------------------------
The generated CSV file can be used to calculate:

    - Mean packet size
    - Median packet size
    - Minimum and maximum packet sizes
    - Packet-size variance
    - Packet-size standard deviation
    - Packet inter-arrival time
    - Flow duration
    - Idle time
    - Burst size
    - Burst duration
    - Upload and download ratios
    - Packets per second
    - Bytes per second
    - TCP flag distributions
    - Retransmission indicators
    - TCP window-size behavior
    - Payload-size distributions
    - Directional traffic behavior
    - AI traffic versus regular traffic characteristics

Captured CSV Fields
-------------------
General fields:

    packet_number
    capture_time
    unix_timestamp
    relative_time_seconds
    inter_arrival_time_seconds
    interface
    packet_summary

Layer 2 fields:

    source_mac
    destination_mac
    ether_type
    vlan_id
    vlan_priority

Network-layer fields:

    ip_version
    source_ip
    destination_ip
    ip_protocol_number
    ip_header_length_bytes
    ip_total_length_bytes
    ip_payload_length_bytes
    ip_ttl
    ip_tos
    ip_dscp
    ip_ecn
    ip_identification
    ip_flags
    ip_fragment_offset
    ip_checksum

Transport-layer fields:

    transport_protocol
    source_port
    destination_port
    transport_payload_length_bytes

TCP fields:

    tcp_sequence_number
    tcp_acknowledgment_number
    tcp_header_length_bytes
    tcp_flags
    tcp_flag_syn
    tcp_flag_ack
    tcp_flag_fin
    tcp_flag_rst
    tcp_flag_psh
    tcp_flag_urg
    tcp_flag_ece
    tcp_flag_cwr
    tcp_window_size
    tcp_checksum
    tcp_urgent_pointer
    tcp_options
    tcp_mss
    tcp_window_scale
    tcp_sack_permitted
    tcp_timestamp_value
    tcp_timestamp_echo_reply

UDP fields:

    udp_length_bytes
    udp_checksum

ICMP fields:

    icmp_type
    icmp_code
    icmp_checksum
    icmp_identifier
    icmp_sequence_number

ARP fields:

    arp_operation
    arp_sender_mac
    arp_sender_ip
    arp_target_mac
    arp_target_ip

Traffic-analysis fields:

    packet_length_bytes
    captured_length_bytes
    payload_length_bytes
    application_payload_hex
    direction
    canonical_flow_id
    directional_flow_id

Installation
------------
Install Scapy:

    pip install scapy

Windows Requirements
--------------------
1. Install Npcap.
2. Enable:

       Install Npcap in WinPcap API-compatible Mode

3. Run PowerShell or Command Prompt as Administrator.

Linux Requirements
------------------
Install libpcap when necessary:

    sudo apt update
    sudo apt install libpcap-dev

Run with sudo:

    sudo python3 python_tcpdump.py

Basic Usage
-----------
List interfaces:

    python python_tcpdump.py --list-interfaces

Capture all traffic for 60 seconds:

    python python_tcpdump.py --timeout 60

Capture HTTPS traffic:

    python python_tcpdump.py --filter "tcp port 443" --timeout 60

Capture TCP and UDP traffic:

    python python_tcpdump.py --filter "tcp or udp" --timeout 60

Capture traffic involving one IP address:

    python python_tcpdump.py --filter "host 192.168.1.100"

Windows interface example:

    python python_tcpdump.py --interface "Ethernet" --timeout 60

Linux interface example:

    sudo python3 python_tcpdump.py --interface eth0 --timeout 60

Complete example:

    python python_tcpdump.py --interface "Ethernet" --filter "tcp port 443" --timeout 60 --pcap capture.pcap --csv capture.csv

Privacy and Authorization
-------------------------
Only capture network traffic that you are authorized to inspect.

Application payload collection is disabled by default. The program can save
a limited payload preview when --payload-bytes is greater than zero. Avoid
collecting payload data when it may contain private, confidential, regulated,
or authentication information.
"""

import argparse
import csv
import os
from dotenv import load_dotenv
import sys
import time
from datetime import datetime
from typing import Any


load_dotenv()

# Configuration loaded from .env
INTERFACE = os.getenv("INTERFACE", "Ethernet")
CAPTURE_FILTER = os.getenv("CAPTURE_FILTER", "")
PACKET_COUNT = int(os.getenv("PACKET_COUNT", "0"))
TIMEOUT = float(os.getenv("TIMEOUT", "60"))
PCAP_FILE = os.getenv("PCAP_FILE", "capture.pcap")
CSV_FILE = os.getenv("CSV_FILE", "capture.csv")
LOCAL_IPS = [x.strip() for x in os.getenv("LOCAL_IPS","").split(",") if x.strip()]
PAYLOAD_PREVIEW_BYTES_ENV = int(os.getenv("PAYLOAD_PREVIEW_BYTES","0"))


from scapy.all import (
    ARP,
    ICMP,
    IP,
    IPv6,
    TCP,
    UDP,
    Dot1Q,
    Ether,
    Raw,
    get_if_list,
    sniff,
    wrpcap,
)


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Stores complete packets for the PCAP file.
captured_packets = []

# CSV output objects.
csv_file = None
csv_writer = None

# Capture timing.
capture_start_time = None
previous_packet_time = None

# Capture configuration.
selected_interface = ""
local_ip_addresses = set()
payload_preview_bytes = 0

# Running counters.
total_packets = 0
total_bytes = 0


# ============================================================================
# CSV COLUMN DEFINITIONS
# ============================================================================

CSV_FIELDS = [
    # ------------------------------------------------------------------------
    # General capture fields
    # ------------------------------------------------------------------------
    "packet_number",
    "capture_time",
    "unix_timestamp",
    "relative_time_seconds",
    "inter_arrival_time_seconds",
    "interface",
    "packet_summary",

    # ------------------------------------------------------------------------
    # Ethernet and VLAN fields
    # ------------------------------------------------------------------------
    "source_mac",
    "destination_mac",
    "ether_type",
    "vlan_id",
    "vlan_priority",

    # ------------------------------------------------------------------------
    # IP fields
    # ------------------------------------------------------------------------
    "ip_version",
    "source_ip",
    "destination_ip",
    "ip_protocol_number",
    "ip_header_length_bytes",
    "ip_total_length_bytes",
    "ip_payload_length_bytes",
    "ip_ttl",
    "ip_hop_limit",
    "ip_tos",
    "ip_traffic_class",
    "ip_dscp",
    "ip_ecn",
    "ip_identification",
    "ip_flags",
    "ip_fragment_offset",
    "ip_checksum",
    "ipv6_flow_label",

    # ------------------------------------------------------------------------
    # Transport fields
    # ------------------------------------------------------------------------
    "transport_protocol",
    "source_port",
    "destination_port",
    "transport_payload_length_bytes",

    # ------------------------------------------------------------------------
    # TCP fields
    # ------------------------------------------------------------------------
    "tcp_sequence_number",
    "tcp_acknowledgment_number",
    "tcp_header_length_bytes",
    "tcp_flags",
    "tcp_flag_syn",
    "tcp_flag_ack",
    "tcp_flag_fin",
    "tcp_flag_rst",
    "tcp_flag_psh",
    "tcp_flag_urg",
    "tcp_flag_ece",
    "tcp_flag_cwr",
    "tcp_window_size",
    "tcp_checksum",
    "tcp_urgent_pointer",
    "tcp_options",
    "tcp_mss",
    "tcp_window_scale",
    "tcp_sack_permitted",
    "tcp_timestamp_value",
    "tcp_timestamp_echo_reply",

    # ------------------------------------------------------------------------
    # UDP fields
    # ------------------------------------------------------------------------
    "udp_length_bytes",
    "udp_checksum",

    # ------------------------------------------------------------------------
    # ICMP fields
    # ------------------------------------------------------------------------
    "icmp_type",
    "icmp_code",
    "icmp_checksum",
    "icmp_identifier",
    "icmp_sequence_number",

    # ------------------------------------------------------------------------
    # ARP fields
    # ------------------------------------------------------------------------
    "arp_operation",
    "arp_sender_mac",
    "arp_sender_ip",
    "arp_target_mac",
    "arp_target_ip",

    # ------------------------------------------------------------------------
    # Packet-size and payload fields
    # ------------------------------------------------------------------------
    "packet_length_bytes",
    "captured_length_bytes",
    "payload_length_bytes",
    "application_payload_hex",

    # ------------------------------------------------------------------------
    # Flow-analysis fields
    # ------------------------------------------------------------------------
    "direction",
    "canonical_flow_id",
    "directional_flow_id",
]


# ============================================================================
# GENERAL HELPER FUNCTIONS
# ============================================================================

def safe_value(value: Any) -> Any:
    """
    Convert Scapy values into CSV-safe values.

    None is converted to an empty string.
    Other values are converted to strings unless they are numeric.
    """
    if value is None:
        return ""

    if isinstance(value, (int, float)):
        return value

    return str(value)


def format_timestamp(timestamp: float) -> str:
    """
    Convert a Unix timestamp to a readable local date and time.
    """
    return datetime.fromtimestamp(timestamp).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )


def get_layer_payload_length(layer) -> int:
    """
    Return the number of bytes following a specified protocol layer.
    """
    try:
        if layer.payload:
            return len(bytes(layer.payload))
    except Exception:
        pass

    return 0


def get_application_payload(packet) -> bytes:
    """
    Return the Raw application payload when available.

    Encrypted traffic still exposes packet sizes and timing, but its
    application data will normally be encrypted.
    """
    if packet.haslayer(Raw):
        try:
            return bytes(packet[Raw].load)
        except Exception:
            return b""

    return b""


def get_payload_preview(packet) -> str:
    """
    Return a limited hexadecimal preview of the application payload.

    A blank string is returned when payload collection is disabled.
    """
    if payload_preview_bytes <= 0:
        return ""

    payload = get_application_payload(packet)

    if not payload:
        return ""

    return payload[:payload_preview_bytes].hex()


# ============================================================================
# FLOW IDENTIFICATION
# ============================================================================

def build_directional_flow_id(
    protocol: str,
    source_ip: str,
    source_port: Any,
    destination_ip: str,
    destination_port: Any,
) -> str:
    """
    Build a directional five-tuple flow identifier.

    Packets in opposite directions produce different identifiers.

    Format:

        protocol|source_ip|source_port|destination_ip|destination_port
    """
    if not source_ip and not destination_ip:
        return ""

    return (
        f"{protocol}|"
        f"{source_ip}|{source_port}|"
        f"{destination_ip}|{destination_port}"
    )


def build_canonical_flow_id(
    protocol: str,
    source_ip: str,
    source_port: Any,
    destination_ip: str,
    destination_port: Any,
) -> str:
    """
    Build a bidirectional five-tuple flow identifier.

    Both directions of the same connection receive the same flow ID. This
    makes it easier to group request and response packets into one flow.
    """
    if not source_ip and not destination_ip:
        return ""

    endpoint_a = f"{source_ip}:{source_port}"
    endpoint_b = f"{destination_ip}:{destination_port}"

    first, second = sorted([endpoint_a, endpoint_b])

    return f"{protocol}|{first}|{second}"


def determine_direction(source_ip: str, destination_ip: str) -> str:
    """
    Classify a packet as outbound, inbound, local, or unknown.

    The classification depends on local IP addresses supplied with the
    --local-ip option.

    Examples:

        --local-ip 192.168.1.25
        --local-ip 192.168.1.25 --local-ip 10.0.0.25
    """
    if not local_ip_addresses:
        return "unknown"

    source_is_local = source_ip in local_ip_addresses
    destination_is_local = destination_ip in local_ip_addresses

    if source_is_local and destination_is_local:
        return "local"

    if source_is_local:
        return "outbound"

    if destination_is_local:
        return "inbound"

    return "other"


# ============================================================================
# TCP OPTION EXTRACTION
# ============================================================================

def extract_tcp_options(tcp_layer) -> dict:
    """
    Extract selected TCP options useful for later traffic analysis.

    Extracted options include:

        - Maximum Segment Size
        - Window Scale
        - SACK Permitted
        - TCP timestamp value
        - TCP timestamp echo reply
    """
    result = {
        "tcp_options": "",
        "tcp_mss": "",
        "tcp_window_scale": "",
        "tcp_sack_permitted": 0,
        "tcp_timestamp_value": "",
        "tcp_timestamp_echo_reply": "",
    }

    try:
        options = tcp_layer.options or []
        result["tcp_options"] = repr(options)

        for option_name, option_value in options:
            if option_name == "MSS":
                result["tcp_mss"] = option_value

            elif option_name == "WScale":
                result["tcp_window_scale"] = option_value

            elif option_name == "SAckOK":
                result["tcp_sack_permitted"] = 1

            elif option_name == "Timestamp":
                if isinstance(option_value, tuple) and len(option_value) == 2:
                    result["tcp_timestamp_value"] = option_value[0]
                    result["tcp_timestamp_echo_reply"] = option_value[1]

    except Exception:
        pass

    return result


# ============================================================================
# PACKET FIELD EXTRACTION
# ============================================================================

def create_empty_record() -> dict:
    """
    Create an empty CSV record containing every configured field.
    """
    return {field: "" for field in CSV_FIELDS}


def extract_ethernet_fields(packet, record: dict) -> None:
    """
    Extract Ethernet and VLAN information.
    """
    if packet.haslayer(Ether):
        ethernet = packet[Ether]

        record["source_mac"] = safe_value(ethernet.src)
        record["destination_mac"] = safe_value(ethernet.dst)
        record["ether_type"] = safe_value(ethernet.type)

    if packet.haslayer(Dot1Q):
        vlan = packet[Dot1Q]

        record["vlan_id"] = safe_value(vlan.vlan)
        record["vlan_priority"] = safe_value(vlan.prio)


def extract_ipv4_fields(packet, record: dict) -> None:
    """
    Extract IPv4 header information.
    """
    if not packet.haslayer(IP):
        return

    ip_layer = packet[IP]

    header_length = int(ip_layer.ihl or 0) * 4
    total_length = int(ip_layer.len or len(ip_layer))
    payload_length = max(total_length - header_length, 0)

    tos = int(ip_layer.tos or 0)

    record["ip_version"] = 4
    record["source_ip"] = safe_value(ip_layer.src)
    record["destination_ip"] = safe_value(ip_layer.dst)
    record["ip_protocol_number"] = safe_value(ip_layer.proto)
    record["ip_header_length_bytes"] = header_length
    record["ip_total_length_bytes"] = total_length
    record["ip_payload_length_bytes"] = payload_length
    record["ip_ttl"] = safe_value(ip_layer.ttl)
    record["ip_tos"] = tos
    record["ip_dscp"] = tos >> 2
    record["ip_ecn"] = tos & 0x03
    record["ip_identification"] = safe_value(ip_layer.id)
    record["ip_flags"] = safe_value(ip_layer.flags)
    record["ip_fragment_offset"] = safe_value(ip_layer.frag)
    record["ip_checksum"] = safe_value(ip_layer.chksum)


def extract_ipv6_fields(packet, record: dict) -> None:
    """
    Extract IPv6 header information.
    """
    if not packet.haslayer(IPv6):
        return

    ipv6_layer = packet[IPv6]
    payload_length = int(ipv6_layer.plen or 0)
    traffic_class = int(ipv6_layer.tc or 0)

    record["ip_version"] = 6
    record["source_ip"] = safe_value(ipv6_layer.src)
    record["destination_ip"] = safe_value(ipv6_layer.dst)
    record["ip_protocol_number"] = safe_value(ipv6_layer.nh)
    record["ip_header_length_bytes"] = 40
    record["ip_total_length_bytes"] = 40 + payload_length
    record["ip_payload_length_bytes"] = payload_length
    record["ip_hop_limit"] = safe_value(ipv6_layer.hlim)
    record["ip_traffic_class"] = traffic_class
    record["ip_dscp"] = traffic_class >> 2
    record["ip_ecn"] = traffic_class & 0x03
    record["ipv6_flow_label"] = safe_value(ipv6_layer.fl)


def extract_tcp_fields(packet, record: dict) -> None:
    """
    Extract TCP header, flag, window, sequence, acknowledgment, checksum,
    payload-size, and TCP-option information.
    """
    if not packet.haslayer(TCP):
        return

    tcp_layer = packet[TCP]
    tcp_flags = int(tcp_layer.flags)

    record["transport_protocol"] = "TCP"
    record["source_port"] = safe_value(tcp_layer.sport)
    record["destination_port"] = safe_value(tcp_layer.dport)
    record["transport_payload_length_bytes"] = get_layer_payload_length(
        tcp_layer
    )

    record["tcp_sequence_number"] = safe_value(tcp_layer.seq)
    record["tcp_acknowledgment_number"] = safe_value(tcp_layer.ack)
    record["tcp_header_length_bytes"] = int(tcp_layer.dataofs or 0) * 4
    record["tcp_flags"] = safe_value(tcp_layer.flags)

    record["tcp_flag_fin"] = 1 if tcp_flags & 0x01 else 0
    record["tcp_flag_syn"] = 1 if tcp_flags & 0x02 else 0
    record["tcp_flag_rst"] = 1 if tcp_flags & 0x04 else 0
    record["tcp_flag_psh"] = 1 if tcp_flags & 0x08 else 0
    record["tcp_flag_ack"] = 1 if tcp_flags & 0x10 else 0
    record["tcp_flag_urg"] = 1 if tcp_flags & 0x20 else 0
    record["tcp_flag_ece"] = 1 if tcp_flags & 0x40 else 0
    record["tcp_flag_cwr"] = 1 if tcp_flags & 0x80 else 0

    record["tcp_window_size"] = safe_value(tcp_layer.window)
    record["tcp_checksum"] = safe_value(tcp_layer.chksum)
    record["tcp_urgent_pointer"] = safe_value(tcp_layer.urgptr)

    tcp_options = extract_tcp_options(tcp_layer)
    record.update(tcp_options)


def extract_udp_fields(packet, record: dict) -> None:
    """
    Extract UDP header and length information.
    """
    if not packet.haslayer(UDP):
        return

    udp_layer = packet[UDP]

    record["transport_protocol"] = "UDP"
    record["source_port"] = safe_value(udp_layer.sport)
    record["destination_port"] = safe_value(udp_layer.dport)
    record["transport_payload_length_bytes"] = get_layer_payload_length(
        udp_layer
    )
    record["udp_length_bytes"] = safe_value(udp_layer.len)
    record["udp_checksum"] = safe_value(udp_layer.chksum)


def extract_icmp_fields(packet, record: dict) -> None:
    """
    Extract ICMP message information.
    """
    if not packet.haslayer(ICMP):
        return

    icmp_layer = packet[ICMP]

    record["transport_protocol"] = "ICMP"
    record["icmp_type"] = safe_value(icmp_layer.type)
    record["icmp_code"] = safe_value(icmp_layer.code)
    record["icmp_checksum"] = safe_value(icmp_layer.chksum)

    if hasattr(icmp_layer, "id"):
        record["icmp_identifier"] = safe_value(icmp_layer.id)

    if hasattr(icmp_layer, "seq"):
        record["icmp_sequence_number"] = safe_value(icmp_layer.seq)


def extract_arp_fields(packet, record: dict) -> None:
    """
    Extract ARP request or response information.
    """
    if not packet.haslayer(ARP):
        return

    arp_layer = packet[ARP]

    record["transport_protocol"] = "ARP"
    record["arp_operation"] = safe_value(arp_layer.op)
    record["arp_sender_mac"] = safe_value(arp_layer.hwsrc)
    record["arp_sender_ip"] = safe_value(arp_layer.psrc)
    record["arp_target_mac"] = safe_value(arp_layer.hwdst)
    record["arp_target_ip"] = safe_value(arp_layer.pdst)

    # Use ARP protocol addresses as source and destination IP fields.
    record["source_ip"] = safe_value(arp_layer.psrc)
    record["destination_ip"] = safe_value(arp_layer.pdst)


# ============================================================================
# PACKET PROCESSING
# ============================================================================

def process_packet(packet) -> None:
    """
    Process one captured packet and write a detailed record to the CSV file.

    This function does not calculate statistical results. It records the raw
    fields needed to calculate statistics later.
    """
    global previous_packet_time
    global total_packets
    global total_bytes

    packet_timestamp = float(packet.time)
    packet_number = total_packets + 1
    packet_length = len(packet)

    total_packets += 1
    total_bytes += packet_length

    # Save the complete packet for PCAP output.
    captured_packets.append(packet)

    # Create a CSV record with every field initialized.
    record = create_empty_record()

    # Calculate timing features.
    relative_time = packet_timestamp - capture_start_time

    if previous_packet_time is None:
        inter_arrival_time = 0.0
    else:
        inter_arrival_time = packet_timestamp - previous_packet_time

    previous_packet_time = packet_timestamp

    # Fill general fields.
    record["packet_number"] = packet_number
    record["capture_time"] = format_timestamp(packet_timestamp)
    record["unix_timestamp"] = packet_timestamp
    record["relative_time_seconds"] = relative_time
    record["inter_arrival_time_seconds"] = inter_arrival_time
    record["interface"] = selected_interface
    record["packet_summary"] = packet.summary()

    # Extract protocol-layer fields.
    extract_ethernet_fields(packet, record)
    extract_ipv4_fields(packet, record)
    extract_ipv6_fields(packet, record)
    extract_tcp_fields(packet, record)
    extract_udp_fields(packet, record)
    extract_icmp_fields(packet, record)
    extract_arp_fields(packet, record)

    # Packet and application payload sizes.
    application_payload = get_application_payload(packet)

    record["packet_length_bytes"] = packet_length
    record["captured_length_bytes"] = len(bytes(packet))
    record["payload_length_bytes"] = len(application_payload)
    record["application_payload_hex"] = get_payload_preview(packet)

    # Build flow identifiers.
    protocol = str(record["transport_protocol"])
    source_ip = str(record["source_ip"])
    destination_ip = str(record["destination_ip"])
    source_port = record["source_port"]
    destination_port = record["destination_port"]

    record["direction"] = determine_direction(
        source_ip,
        destination_ip,
    )

    record["directional_flow_id"] = build_directional_flow_id(
        protocol,
        source_ip,
        source_port,
        destination_ip,
        destination_port,
    )

    record["canonical_flow_id"] = build_canonical_flow_id(
        protocol,
        source_ip,
        source_port,
        destination_ip,
        destination_port,
    )

    # Write the record to CSV.
    if csv_writer is not None:
        csv_writer.writerow(record)
        csv_file.flush()

    # Print a compact real-time line.
    source_endpoint = source_ip
    destination_endpoint = destination_ip

    if source_port != "":
        source_endpoint = f"{source_endpoint}:{source_port}"

    if destination_port != "":
        destination_endpoint = f"{destination_endpoint}:{destination_port}"

    print(
        f"{packet_number:07d} | "
        f"{record['capture_time']} | "
        f"{protocol:<5} | "
        f"{source_endpoint:<30} -> "
        f"{destination_endpoint:<30} | "
        f"{packet_length:5d} bytes | "
        f"IAT {inter_arrival_time:.6f} s | "
        f"{record['direction']}"
    )


# ============================================================================
# OUTPUT FILE FUNCTIONS
# ============================================================================

def open_csv_output(csv_filename: str) -> None:
    """
    Create the CSV file and write all column headings.
    """
    global csv_file
    global csv_writer

    csv_file = open(
        csv_filename,
        mode="w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=CSV_FIELDS,
        extrasaction="ignore",
    )

    csv_writer.writeheader()
    csv_file.flush()


def close_csv_output() -> None:
    """
    Close the CSV file safely.
    """
    global csv_file
    global csv_writer

    if csv_file is not None:
        csv_file.close()

    csv_file = None
    csv_writer = None


def save_pcap_output(pcap_filename: str) -> None:
    """
    Save complete captured packets to a PCAP file.
    """
    if not captured_packets:
        print("No PCAP file was created because no packets were captured.")
        return

    wrpcap(pcap_filename, captured_packets)

    print(f"PCAP file saved         : {os.path.abspath(pcap_filename)}")


# ============================================================================
# CAPTURE SUMMARY
# ============================================================================

def print_capture_summary() -> None:
    """
    Print only basic capture totals.

    Detailed statistical analysis is intentionally left for later processing
    of the generated CSV file.
    """
    print("\n" + "=" * 78)
    print("PACKET CAPTURE SUMMARY")
    print("=" * 78)

    if capture_start_time is None:
        print("Capture did not start.")
        return

    duration = max(time.time() - capture_start_time, 0.000001)
    packet_rate = total_packets / duration
    throughput_bps = total_bytes * 8 / duration
    throughput_mbps = throughput_bps / 1_000_000

    print(f"Capture duration        : {duration:,.6f} seconds")
    print(f"Captured packets        : {total_packets:,}")
    print(f"Captured bytes          : {total_bytes:,} bytes")
    print(f"Packet rate             : {packet_rate:,.2f} packets/second")
    print(f"Average throughput      : {throughput_bps:,.2f} bits/second")
    print(f"Average throughput      : {throughput_mbps:,.6f} Mbps")
    print("=" * 78)


# ============================================================================
# NETWORK INTERFACE DISPLAY
# ============================================================================

def show_interfaces() -> None:
    """
    Display network interfaces detected by Scapy.
    """
    print("\n" + "=" * 78)
    print("AVAILABLE NETWORK INTERFACES")
    print("=" * 78)

    interfaces = get_if_list()

    if not interfaces:
        print("No interfaces were detected.")
    else:
        for index, interface in enumerate(interfaces, start=1):
            print(f"{index:3d}. {interface}")

    print("=" * 78)


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_arguments():
    """
    Define command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Capture detailed packet metadata for later statistical and "
            "machine-learning analysis."
        )
    )

    parser.add_argument(
        "-i",
        "--interface",
        default=None,
        help=(
            "Interface to capture from. When omitted, Scapy uses its "
            "default interface."
        ),
    )

    parser.add_argument(
        "-f",
        "--filter",
        default="",
        help=(
            'BPF filter such as "tcp", "udp", "tcp port 443", or '
            '"host 192.168.1.100".'
        ),
    )

    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=0,
        help="Maximum packets to capture. Zero means unlimited.",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=0,
        help="Capture duration in seconds. Zero means unlimited.",
    )

    parser.add_argument(
        "--pcap",
        default="capture.pcap",
        help="PCAP output filename.",
    )

    parser.add_argument(
        "--csv",
        default="capture.csv",
        help="CSV output filename.",
    )

    parser.add_argument(
        "--local-ip",
        action="append",
        default=[],
        help=(
            "Local IP address used to label packets as inbound or outbound. "
            "This option can be supplied more than once."
        ),
    )

    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=0,
        help=(
            "Maximum application-payload bytes saved as hexadecimal per "
            "packet. Default 0 disables payload collection."
        ),
    )

    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="Display available interfaces and exit.",
    )

    return parser.parse_args()


# ============================================================================
# MAIN TCPDUMP CAPTURE FUNCTION
# ============================================================================

def tcpdump_capture(
        interface=INTERFACE,
        capture_filter=CAPTURE_FILTER,
        packet_count=PACKET_COUNT,
        timeout=TIMEOUT,
        pcap_filename=PCAP_FILE,
        csv_filename=CSV_FILE,
        local_ips=LOCAL_IPS,
        maximum_payload_bytes=PAYLOAD_PREVIEW_BYTES_ENV,
    ):
    """
    Capture detailed packet-level information for later analysis.

    This function does not calculate packet-size statistical measurements.
    It saves raw packet fields so those measurements can be calculated later.

    Parameters
    ----------
    interface:
        Interface name, such as Ethernet, Wi-Fi, eth0, or wlan0.

    capture_filter:
        Optional BPF capture filter.

    packet_count:
        Maximum number of packets. Zero means unlimited.

    timeout:
        Maximum capture duration in seconds. Zero means unlimited.

    pcap_filename:
        Output PCAP filename.

    csv_filename:
        Output CSV filename.

    local_ips:
        List of local IP addresses used to determine packet direction.

    maximum_payload_bytes:
        Maximum number of application payload bytes saved as hexadecimal.
        Zero disables payload collection.
    """
    global capture_start_time
    global previous_packet_time
    global selected_interface
    global local_ip_addresses
    global payload_preview_bytes
    global total_packets
    global total_bytes

    if packet_count < 0:
        raise ValueError("Packet count cannot be negative.")

    if timeout < 0:
        raise ValueError("Timeout cannot be negative.")

    if maximum_payload_bytes < 0:
        raise ValueError("Payload byte limit cannot be negative.")

    # Reset capture state.
    captured_packets.clear()
    total_packets = 0
    total_bytes = 0
    previous_packet_time = None

    selected_interface = interface or "Scapy default"
    local_ip_addresses = set(local_ips or [])
    payload_preview_bytes = maximum_payload_bytes

    open_csv_output(csv_filename)

    print("\n" + "=" * 78)
    print("PYTHON TCPDUMP DATA COLLECTOR")
    print("=" * 78)
    print(f"Interface               : {selected_interface}")
    print(f"Capture filter          : {capture_filter or 'No filter'}")
    print(f"Packet limit            : {packet_count or 'Unlimited'}")
    print(f"Time limit              : {timeout or 'Unlimited'}")
    print(f"Local IP addresses      : {sorted(local_ip_addresses) or 'Not set'}")
    print(f"Payload preview         : {payload_preview_bytes} bytes")
    print(f"PCAP output             : {os.path.abspath(pcap_filename)}")
    print(f"CSV output              : {os.path.abspath(csv_filename)}")
    print("=" * 78)
    print("Press Ctrl+C to stop capturing.\n")

    capture_start_time = time.time()

    sniff_options = {
        "prn": process_packet,
        "store": False,
    }

    if interface:
        sniff_options["iface"] = interface

    if capture_filter:
        sniff_options["filter"] = capture_filter

    if packet_count > 0:
        sniff_options["count"] = packet_count

    if timeout > 0:
        sniff_options["timeout"] = timeout

    try:
        sniff(**sniff_options)

    except KeyboardInterrupt:
        print("\nPacket capture stopped by the user.")

    except PermissionError:
        print("\nERROR: Packet capture permission was denied.")
        print("Run as Administrator on Windows or use sudo on Linux.")

    except OSError as error:
        print(f"\nNetwork capture error: {error}")
        print(
            "Verify the interface name and confirm that Npcap or libpcap "
            "is installed."
        )

    except Exception as error:
        print(
            f"\nUnexpected capture error: "
            f"{type(error).__name__}: {error}"
        )

    finally:
        close_csv_output()
        save_pcap_output(pcap_filename)

        print(f"CSV file saved          : {os.path.abspath(csv_filename)}")

        print_capture_summary()


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main() -> None:
    """
    Read command-line arguments and start packet capture.
    """
    args = parse_arguments()

    if args.list_interfaces:
        show_interfaces()
        return

    if args.count < 0:
        print("ERROR: Packet count cannot be negative.")
        sys.exit(1)

    if args.timeout < 0:
        print("ERROR: Timeout cannot be negative.")
        sys.exit(1)

    if args.payload_bytes < 0:
        print("ERROR: Payload byte limit cannot be negative.")
        sys.exit(1)

    tcpdump_capture(
        interface=args.interface,
        capture_filter=args.filter,
        packet_count=args.count,
        timeout=args.timeout,
        pcap_filename=args.pcap,
        csv_filename=args.csv,
        local_ips=args.local_ip,
        maximum_payload_bytes=args.payload_bytes,
    )


if __name__ == "__main__":
    main()