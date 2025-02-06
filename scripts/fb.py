import argparse
import pandas as pd
from utils.bds import load_bds_data
from utils.fb import load_fb_data
from utils.gpt import process_csv
from utils.helpers import load_geojson, filter_geojson

# python script.py --state "Vietnam" --city "Danang"
def main(state="vietnam", city="danang", output_csv="data/vietnam/danang/fb_cleaned.csv"):
    print(f"Loading data for {city}, {state}...")
    df = load_fb_data(state, city)
    if df.empty:
        print("No data available.")
        return
    
    print("Processing CSV...")
    process_csv(df, output_csv)
    df = pd.read_csv(output_csv)
    
    print(f"Processed data saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airbnb Data Visualization CLI")
    parser.add_argument("--state", type=str, default="vietnam", help="State name (e.g., Vietnam)")
    parser.add_argument("--city", type=str, default="danang", help="City name (e.g., Danang)")
    parser.add_argument("--output_csv", type=str, default="data/vietnam/danang/fb_cleaned.csv", help="Output CSV file path")
    args = parser.parse_args()
    main(args.state, args.city, args.output_csv)
