"""Download public myocardial-infarction EKG images for educational testing.

Source: Wikimedia Commons category "ECG of myocardial infarction".
The original image, source page, author, and license are recorded in
ekg_sources.csv. These images are not for clinical diagnosis.

Run on Windows:
    py download_ekg_heart_attack_images.py

Output folder:
    ekg-data-MMDDYY\
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
CATEGORY = "Category:ECG of myocardial infarction"
USER_AGENT = "MedGemmaEducationalEKGDownloader/1.0"
SUPPORTED_FORMATS = {"JPEG", "PNG"}


def read_env_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE settings without requiring python-dotenv."""
    settings: dict[str, str] = {}
    if not path.is_file():
        return settings
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        settings[key.strip()] = value.strip().strip('"').strip("'")
    return settings


def image_dimension(settings: dict[str, str], name: str) -> int:
    """Read an output dimension; zero means retain the original dimension."""
    value = settings.get(name, "0")
    try:
        number = int(value)
        if 0 <= number <= 8192:
            return number
    except ValueError:
        pass
    raise ValueError(
        f"{name} must be a whole number from 0 to 8192. "
        "Use 0 to retain the original dimension."
    )


def download_settings(script_directory: Path) -> tuple[int, str, int, int]:
    """Return image count, folder extension, width, and height from .env."""
    settings = read_env_file(script_directory / ".env")

    count_text = settings.get("EKG_IMAGE_COUNT", "10")
    try:
        image_count = int(count_text)
        if not 1 <= image_count <= 50:
            raise ValueError
    except ValueError as error:
        raise ValueError("EKG_IMAGE_COUNT must be from 1 to 50.") from error

    folder_extension = settings.get(
        "EKG_FOLDER_EXTENSION", datetime.now().strftime("%m%d%y")
    )
    try:
        parsed_date = datetime.strptime(folder_extension, "%m%d%y")
        if parsed_date.strftime("%m%d%y") != folder_extension:
            raise ValueError
    except ValueError as error:
        raise ValueError(
            "EKG_FOLDER_EXTENSION must be a valid MMDDYY date, such as 092226."
        ) from error

    width = image_dimension(settings, "EKG_IMAGE_WIDTH")
    height = image_dimension(settings, "EKG_IMAGE_HEIGHT")
    return image_count, folder_extension, width, height


def load_pillow():
    """Import Pillow, installing it for this Python interpreter if missing."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        print("Pillow is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageOps
    return Image, ImageOps


def api_request(parameters: dict[str, str]) -> dict:
    """Call the public Wikimedia Commons API and return decoded JSON."""
    query = urllib.parse.urlencode({"format": "json", **parameters})
    request = urllib.request.Request(
        f"{COMMONS_API}?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def category_file_titles(limit: int) -> list[str]:
    """Return raster-file titles from the myocardial-infarction ECG category."""
    titles: list[str] = []
    continuation: str | None = None
    while len(titles) < limit:
        parameters = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": CATEGORY,
            "cmnamespace": "6",
            "cmlimit": "50",
        }
        if continuation:
            parameters["cmcontinue"] = continuation
        data = api_request(parameters)
        members = data.get("query", {}).get("categorymembers", [])
        for member in members:
            title = member.get("title", "")
            if Path(title.removeprefix("File:")).suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }:
                titles.append(title)
                if len(titles) >= limit:
                    break
        continuation = data.get("continue", {}).get("cmcontinue")
        if not continuation:
            break
    return titles


def image_information(title: str) -> dict[str, str]:
    """Return original-file URL and attribution fields for one Commons file."""
    data = api_request(
        {
            "action": "query",
            "prop": "imageinfo",
            "titles": title,
            "iiprop": "url|size|mime|extmetadata",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    info = (page.get("imageinfo") or [{}])[0]
    metadata = info.get("extmetadata", {})

    def value(name: str) -> str:
        return str(metadata.get(name, {}).get("value", ""))

    return {
        "title": title.removeprefix("File:"),
        "url": str(info.get("url", "")),
        "page_url": str(info.get("descriptionurl", "")),
        "author": value("Artist"),
        "license": value("LicenseShortName"),
        "license_url": value("LicenseUrl"),
    }


def download_bytes(url: str) -> bytes:
    """Download one original image from Wikimedia's file server."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    count, extension, configured_width, configured_height = download_settings(
        script_directory
    )
    output_directory = script_directory / f"ekg-data-{extension}"
    output_directory.mkdir(parents=True, exist_ok=True)
    Image, ImageOps = load_pillow()

    print(f"Finding up to {count} myocardial-infarction EKG images...")
    titles = category_file_titles(count)
    if not titles:
        raise RuntimeError("No compatible EKG images were returned by Commons.")

    source_rows: list[dict[str, str]] = []
    saved = 0
    for title in titles:
        try:
            info = image_information(title)
            if not info["url"]:
                continue
            image_bytes = download_bytes(info["url"])
            with Image.open(io.BytesIO(image_bytes)) as original:
                if original.format not in SUPPORTED_FORMATS:
                    continue
                original_width, original_height = original.size
                destination_width = configured_width or original_width
                destination_height = configured_height or original_height
                saved += 1
                destination = output_directory / f"heart-attack-ekg-{saved:02d}.png"
                converted = original.convert("RGB")
                if (destination_width, destination_height) == original.size:
                    finished = converted
                else:
                    finished = ImageOps.pad(
                        converted,
                        (destination_width, destination_height),
                        method=Image.Resampling.LANCZOS,
                        color="white",
                    )
                finished.save(destination, format="PNG")

            source_rows.append(
                {
                    "saved_file": destination.name,
                    "original_title": info["title"],
                    "source_page": info["page_url"],
                    "author": info["author"],
                    "license": info["license"],
                    "license_url": info["license_url"],
                }
            )
            print(
                f"Created: {destination.name} | "
                f"original: {original_width}x{original_height} | "
                f"destination: {destination_width}x{destination_height}"
            )
        except Exception as error:
            print(f"Skipped {title}: {error}")

    if not source_rows:
        raise RuntimeError("The source returned no usable JPG or PNG images.")

    source_file = output_directory / "ekg_sources.csv"
    with source_file.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    print(f"\nSaved {saved} EKG images in:")
    print(output_directory)
    print(f"Attribution and licenses: {source_file.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nDownload failed: {error}")
        raise SystemExit(1)
