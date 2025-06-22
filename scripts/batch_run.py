from pathlib import Path
import argparse
import json
import tqdm

from trustguard.orchestrator import analyse_listing
from trustguard.ingest import load_listings

def main(input_csv: Path, output_json: Path):
    results = []
    for listing in tqdm.tqdm(
        load_listings(input_csv),
        desc="Scanning listings",
        unit="listing"
    ):
        results.append(analyse_listing(listing))

    output_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print(f"✓ Done. Wrote {len(results)} records to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TrustGuard+ on a CSV of listings and emit reports.json"
    )
    parser.add_argument(
        "--csv", type=Path, required=True,
        help="Path to the input CSV of product listings"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("reports.json"),
        help="Path for the output JSON report"
    )
    args = parser.parse_args()
    main(args.csv, args.out)
