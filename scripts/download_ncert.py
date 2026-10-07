"""
Download NCERT textbooks from official sources.

Downloads EVS/Science textbooks for specified grades.
"""

import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# NCERT textbook URLs (2024-25 edition)
# Note: These URLs may change. Verify at https://ncert.nic.in/textbook.php
NCERT_URLS = {
    "evs": {
        3: "https://ncert.nic.in/pdf/publication/scert/0367Looking%20Around_III.pdf",
        4: "https://ncert.nic.in/pdf/publication/scert/0368Looking%20Around_IV.pdf",
        5: "https://ncert.nic.in/pdf/publication/scert/0369Looking%20Around_V.pdf",
    },
    "science": {
        6: "https://ncert.nic.in/pdf/publication/scert/0651Science.pdf",
        7: "https://ncert.nic.in/pdf/publication/scert/0752Science.pdf",
        8: "https://ncert.nic.in/pdf/publication/scert/0853Science.pdf",
    }
}


def download_file(url: str, output_path: Path) -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=output_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download NCERT textbooks")
    parser.add_argument(
        "--subject",
        type=str,
        default="evs",
        choices=["evs", "science"],
        help="Subject to download"
    )
    parser.add_argument(
        "--grades",
        nargs="+",
        type=int,
        help="Grades to download (e.g., 3 4 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for PDFs"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    urls = NCERT_URLS.get(args.subject, {})
    grades = args.grades if args.grades else list(urls.keys())
    
    logger.info(f"Downloading {args.subject} textbooks for grades: {grades}")
    
    for grade in grades:
        if grade not in urls:
            logger.warning(f"No URL configured for {args.subject} grade {grade}")
            continue
        
        url = urls[grade]
        filename = f"ncert_{args.subject}_grade{grade}.pdf"
        output_path = output_dir / filename
        
        if output_path.exists():
            logger.info(f"Already exists: {output_path}")
            continue
        
        logger.info(f"Downloading Grade {grade}...")
        download_file(url, output_path)
    
    logger.info(f"Download complete. Files saved to: {output_dir}")
    logger.info("\\nIMPORTANT: These materials are used under NCERT educational use terms.")
    logger.info("See docs/data_rights.md for licensing information.")


if __name__ == "__main__":
    main()
