"""Download public chest X-ray examples for multiple diseases.

All images are saved directly in one folder named
``xray-data-MMDDYY``. No disease subfolders are created. The disease name is
included in each filename, for example ``pneumonia-001.jpg``.

The images are obtained through the Wikimedia Commons API. They are intended
for education and software testing, not medical diagnosis or model training.
An ``xray-image-attribution.json`` file records each image's source and license.
"""

import html
import json
import os
import re
import shutil
import time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# Number of images to download for EACH X-ray disease.
XRAY_IMAGE_COUNT = 3

# Folder extension in MMDDYY format.
XRAY_FOLDER_EXTENSION = "091426"

# Final image dimensions. A value of 0 maintains the original dimension.
XRAY_IMAGE_WIDTH = 0
XRAY_IMAGE_HEIGHT = 0

# Chest X-ray diseases/conditions to search for.
XRAY_DISEASES = (
    "pneumonia",
    "tuberculosis",
    "COVID-19",
    "atelectasis",
    "cardiomegaly",
    "pleural effusion",
    "pneumothorax",
    "pulmonary edema",
    "emphysema",
    "pulmonary fibrosis",
    "lung mass",
    "lung nodule",
    "consolidation",
    "infiltration",
    "pleural thickening",
    "hiatal hernia",
)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Educational-Xray-Downloader/1.0"
REQUEST_TIMEOUT = 45
SEARCH_LIMIT = 50
SUPPORTED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def safe_name(value):
    """Convert a disease label to a safe lowercase filename component."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def plain_text(value):
    """Remove simple HTML markup returned by Commons metadata."""
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def api_request(parameters):
    """Return one decoded JSON response from Wikimedia Commons."""
    query = urlencode({"format": "json", "formatversion": 2, **parameters})
    request = Request(f"{COMMONS_API}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.load(response)


def search_candidates(disease):
    """Find likely chest X-ray files for one disease."""
    data = api_request(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f'filetype:bitmap "chest x-ray" {disease}',
            "gsrnamespace": 6,
            "gsrlimit": SEARCH_LIMIT,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
        }
    )
    return data.get("query", {}).get("pages", [])


def download_bytes(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return response.read()


def prepare_image(image_bytes):
    """Validate image data and optionally resize it; return bytes and suffix."""
    try:
        from PIL import Image
    except ImportError as error:
        if XRAY_IMAGE_WIDTH or XRAY_IMAGE_HEIGHT:
            raise RuntimeError(
                "Resizing requires Pillow. Install it with: py -m pip install pillow"
            ) from error
        # Without Pillow, preserve the source bytes. Commons URLs normally end
        # with a useful extension, but JPEG is the safest fallback below.
        return image_bytes, ".jpg"

    with Image.open(BytesIO(image_bytes)) as source:
        source.load()
        image_format = source.format
        suffix = SUPPORTED_FORMATS.get(image_format, ".jpg")
        picture = source.convert("RGB") if suffix == ".jpg" else source.copy()

        if XRAY_IMAGE_WIDTH or XRAY_IMAGE_HEIGHT:
            width = XRAY_IMAGE_WIDTH or picture.width
            height = XRAY_IMAGE_HEIGHT or picture.height
            picture = picture.resize((width, height), Image.Resampling.LANCZOS)

        output = BytesIO()
        save_format = "JPEG" if suffix == ".jpg" else image_format
        save_options = {"quality": 95} if save_format == "JPEG" else {}
        picture.save(output, format=save_format, **save_options)
        return output.getvalue(), suffix


def attribution_record(disease, page, info, filename):
    metadata = info.get("extmetadata", {})

    def field(name):
        return plain_text(metadata.get(name, {}).get("value", ""))

    return {
        "filename": filename,
        "disease_search": disease,
        "commons_title": page.get("title", ""),
        "description": field("ImageDescription"),
        "artist": field("Artist"),
        "license": field("LicenseShortName"),
        "license_url": field("LicenseUrl"),
        "source_page": info.get("descriptionurl", ""),
        "original_url": info.get("url", ""),
    }


def validate_settings():
    if not isinstance(XRAY_IMAGE_COUNT, int) or XRAY_IMAGE_COUNT < 1:
        raise ValueError("XRAY_IMAGE_COUNT must be a positive integer.")
    if not re.fullmatch(r"\d{6}", str(XRAY_FOLDER_EXTENSION)):
        raise ValueError("XRAY_FOLDER_EXTENSION must use MMDDYY, for example 092926.")
    for name, value in (
        ("XRAY_IMAGE_WIDTH", XRAY_IMAGE_WIDTH),
        ("XRAY_IMAGE_HEIGHT", XRAY_IMAGE_HEIGHT),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be zero or a positive integer.")


def main():
    validate_settings()
    output_folder = Path(__file__).resolve().parent / f"xray-data-{XRAY_FOLDER_EXTENSION}"
    output_folder.mkdir(parents=True, exist_ok=True)
    attribution = []
    total = 0

    print(f"Saving all X-ray images in: {output_folder}")
    print(f"Target: {XRAY_IMAGE_COUNT} image(s) for each of {len(XRAY_DISEASES)} diseases")

    for disease in XRAY_DISEASES:
        downloaded = 0
        disease_name = safe_name(disease)
        print(f"\nSearching for {disease}...")
        try:
            candidates = search_candidates(disease)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            print(f"  Search failed: {error}")
            continue

        for page in candidates:
            if downloaded >= XRAY_IMAGE_COUNT:
                break
            image_info = page.get("imageinfo") or []
            if not image_info:
                continue
            info = image_info[0]
            if not str(info.get("mime", "")).startswith("image/"):
                continue
            try:
                image_bytes = download_bytes(info["url"])
                final_bytes, suffix = prepare_image(image_bytes)
                filename = f"{disease_name}-{downloaded + 1:03d}{suffix}"
                destination = output_folder / filename
                destination.write_bytes(final_bytes)
            except (KeyError, OSError, HTTPError, URLError, TimeoutError, RuntimeError) as error:
                print(f"  Skipped {page.get('title', 'unknown file')}: {error}")
                continue

            downloaded += 1
            total += 1
            attribution.append(attribution_record(disease, page, info, filename))
            print(f"  [{downloaded}/{XRAY_IMAGE_COUNT}] {filename}")
            time.sleep(0.25)

        if downloaded < XRAY_IMAGE_COUNT:
            print(f"  Warning: found only {downloaded} usable image(s) for {disease}.")

    metadata_path = output_folder / "xray-image-attribution.json"
    metadata_path.write_text(json.dumps(attribution, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFinished. Downloaded {total} image(s) into {output_folder.name}")
    print(f"Attribution information: {metadata_path.name}")


if __name__ == "__main__":
    main()
