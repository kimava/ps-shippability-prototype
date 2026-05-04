"""CLI entrypoint for Pay & Ship scoring prototype."""

from pathlib import Path

from pay_ship_scoring.config import ScoringConfig
from pay_ship_scoring.data_loader import load_listings, load_signal_library
from pay_ship_scoring.output_writer import write_scored_output
from pay_ship_scoring.scorer import score_listings


def run() -> None:
    project_root = Path(__file__).resolve().parents[1]
    listings_path = project_root / "data" / "listings.csv"
    signals_path = project_root / "data" / "signal_library.csv"
    output_path = project_root / "outputs" / "scored_listings.csv"

    listings_df = load_listings(listings_path)
    signal_library_df = load_signal_library(signals_path)
    config = ScoringConfig()

    result_df = score_listings(listings_df, signal_library_df, config)
    write_scored_output(result_df, output_path)

    print(f"Scored {len(result_df)} listings -> {output_path}")


if __name__ == "__main__":
    run()

