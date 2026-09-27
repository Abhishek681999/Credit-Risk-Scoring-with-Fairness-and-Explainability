"""Download NSE capital-market UDiFF bhavcopies, one ZIP per trading day."""

import argparse
import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://nsearchives.nseindia.com/content/cm"
OUTPUT_DIR = Path("data/market/bhavcopy")
KNOWN_DATE = date(2026, 9, 25)  # The sample file you downloaded
PAUSE_SECONDS = 1.5


def filename(day):
    return f"BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"


def valid_zip(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            return (
                len(names) == 1
                and names[0].lower().endswith(".csv")
                and archive.testzip() is None
            )
    except zipfile.BadZipFile:
        return False


def download(day):
    destination = OUTPUT_DIR / filename(day)

    if destination.exists():
        if valid_zip(destination.read_bytes()):
            return "already downloaded"
        destination.unlink()  # Replace an incomplete or damaged file.

    url = f"{BASE_URL}/{filename(day)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/zip,application/octet-stream,*/*",
        },
    )

    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                data = response.read()

            if not valid_zip(data):
                raise RuntimeError(f"Response is not a valid CSV ZIP: {url}")

            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(destination)
            return f"downloaded ({len(data):,} bytes)"

        except HTTPError as error:
            if error.code == 404:
                return "no file (holiday or unavailable date)"
            if error.code in (401, 403):
                raise RuntimeError(
                    f"NSE denied access (HTTP {error.code}). Stop here; "
                    "use NSE's website archive rather than retrying repeatedly."
                ) from error
            if error.code != 429 and error.code < 500:
                raise
            problem = f"HTTP {error.code}"

        except URLError as error:
            problem = str(error.reason)

        if attempt < 2:
            time.sleep(3 * (attempt + 1))

    raise RuntimeError(f"Failed to download {day}: {problem}")


def days_between(start, end):
    day = start
    while day <= end:
        if day.weekday() < 5:  # Monday to Friday
            yield day
        day += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-09-25")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Download only the known 25 September 2026 sample",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.probe:
        print(f"{KNOWN_DATE}: {download(KNOWN_DATE)}")
        return

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start > end:
        parser.error("--start must be on or before --end")
    if end > date.today():
        parser.error("--end cannot be in the future")

    counts = {"downloaded": 0, "already downloaded": 0, "no file": 0}

    for day in days_between(start, end):
        result = download(day)
        print(f"{day}: {result}", flush=True)

        for category in counts:
            if result.startswith(category):
                counts[category] += 1
                break

        if result.startswith("downloaded"):
            time.sleep(PAUSE_SECONDS)

    print(f"\nSummary: {counts}")
    print(f"Files saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()