import openai
import pandas as pd
import json
import time
import re
import os

# OpenAI processing function
def process_with_openai(text):
    prompt = """
    Extract listing information from Facebook post:
    - address: extract from post
    - street:  extract from post
    - Ward:  guess from ["Thanh Binh Ward","Chinh Gian Ward","Vinh Trung Ward","Thac Gian Ward","Tam Thuan Ward","Thanh Khe Tay Ward","Thanh Khe Dong Ward","Xuan Ha Ward","Tan Chinh Ward","An Khe Ward","Phuoc Ninh Ward","Hai Chau II Ward","Hoa Thuan Dong Ward","Nam Duong Ward","Binh Hien Ward","Binh Thuan Ward","Hoa Cuong Bac Ward","Hoa Cuong Nam Ward","Khue Trung Ward","Tho Quang Ward","Nai Hien Dong Ward","Man Thai Ward","An Hai Bac Ward","Phuoc My Ward","An Hai Tay Ward","An Hai Dong Ward","Hoa Quy Ward","Hoa Hai Ward","Hoa Lien Ward","Hoa Son Ward","Hoa Phat Ward","Hoa An Ward","Hoa Nhon Ward","Hoa Tho Tay Ward","Hoa Tho Dong Ward","Hoa Xuan Ward","Hoa Phong Ward","Hoa Chau Ward","Hoa Tien Ward","Hoa Phuoc Ward","Hoa Khe Ward","Thuan Phuoc Ward","Thach Thang Ward","Hai Chau I Ward","My An Ward","Khue My Ward","Hoa Hiep Nam Ward","Hoa Khanh Bac Ward","Hoa Khanh Nam Ward","Hoa Minh Ward"]
    - District:  guess from ["Thanh Khe District","Hai Chau District","Cam Le District","Hoa Vang District","Lien Chieu District","Son Tra District","Ngu Hanh Son District"]
    - city: Danang etc
    - priceBil: price in billion
    - propertyType: land, house, apartment
    - area: in meter square
    - bedroom: number of bedroom
    - road: road wide
    - Output JSON only (no other text)

    Example input: "Selling house in Thanh Khe, 3 bedrooms, 100m2, 2B VND, road wide 5m5
    Example output: {"address": "...", "street": "...", "ward": "...", "district": "...", "city": "Danang", "priceBil": 2, "propertyType": "house", "area": 100, "bedroom": 3, "road": "5.5" }
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            timeout=6000,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text },
            ],
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return None

# Function to clean and parse JSON
def parse_json(json_str):
    try:
        json_str = re.sub(r'```json|```', '', json_str.strip())  # Remove markdown artifacts
        return json.loads(json_str)
    except Exception as e:
        return None

def fix_multilines_text(text):
    if not isinstance(text, str):  
        return ""  # Convert None or other types to an empty string
    
    # Replace line breaks with a space
    fixed_text = re.sub(r'[\r\n]+', ' ', text).strip()
    
    return fixed_text

# Process CSV function
def process_csv(df, output_csv):
    # Check if the output file exists
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        df = pd.concat([df_existing, df]).drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)
    else:
        df['processed_column'] = None

    # Process only unprocessed rows
    for index, row in df.iterrows():
        if pd.isna(row.get('processed_column')):
            processed_text = process_with_openai(row['text']) if pd.notna(row['text']) else None
            parsed_data = parse_json(processed_text) if processed_text else None
            
            df.at[index, 'processed_column'] = processed_text
            if parsed_data:
                if isinstance(parsed_data, dict):
                    for key, value in parsed_data.items():
                        df.at[index, key] = value
                elif isinstance(parsed_data, list):
                    for item in parsed_data:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                df.at[index, key] = value
            
            # fix bad linebreak
            df.at[index, 'text'] = fix_multilines_text(row.get('text'))
            df.at[index, 'processed_column'] = fix_multilines_text(row.get('processed_column'))
            df.at[index, 'title'] = fix_multilines_text(row.get('title'))
            
            # Drop all columns that start with 'topComment'
            df.drop(columns=[col for col in df.columns if col.startswith('topComment')], inplace=True)

            # Save progress after each row
            df.to_csv(output_csv, index=False)
            print(f"Processed row {index+1}/{len(df)} and saved progress.")
    
    print(f"Processing complete. Output saved to {output_csv}")

# Example Usage:
# process_csv("input.csv", "output.csv")
