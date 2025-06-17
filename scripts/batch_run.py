from pathlib import Path
import argparse, json, tqdm
from trustguard.orchestrator import analyse_listing
from trustguard.ingest import load_listings  # will import relative

def main(csv):
    res = []
    for lst in tqdm.tqdm(load_listings(Path(csv)), desc="Scanning"):
        res.append(analyse_listing(lst))
    Path("reports.json").write_text(json.dumps(res, indent=2))
    print("✓ Done. reports.json created.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    main(args.csv)