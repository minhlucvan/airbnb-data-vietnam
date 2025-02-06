import pandas as pd

def load_bds_data(selected_state, selected_city):
    # Load data and geojson based on the selections
    df = pd.read_csv(f'data/{selected_state}/{selected_city}/bds.csv')

    ## remove all imageUrls/* columns
    df = df.drop(columns=[col for col in df.columns if 'imageUrls' in col])

    for row in df.itertuples():
        if not pd.isnull(df.at[row.Index, 'price']):
            if 'triệu/m²' in df.at[row.Index, 'price']:
                priceExt = df.at[row.Index, 'priceExt']
                df.at[row.Index, 'priceExt'] = df.at[row.Index, 'price']
                df.at[row.Index, 'price'] = priceExt
            elif '/m²' in df.at[row.Index, 'price']:
                priceExt = df.at[row.Index, 'priceExt']
                df.at[row.Index, 'priceExt'] = df.at[row.Index, 'price']
                df.at[row.Index, 'price'] = priceExt
            elif 'triệu' in df.at[row.Index, 'price']:
                # convert ~48,63 triệu -> 4.863 priceBil
                price = df.at[row.Index, 'price']
                priceMil = price.replace(' triệu', '').replace(',', '.')
                priceMil = float(priceMil)
                priceBil = priceMil / 1000
                priceVnd = priceMil * 1000000
                df.at[row.Index, 'priceMil'] = priceMil
                df.at[row.Index, 'priceBil'] = priceBil
                df.at[row.Index, 'priceVnd'] = priceVnd
            elif 'tỷ' in df.at[row.Index, 'price']:
                # convert ~48,63 tỷ -> 48.63 priceBil
                price = df.at[row.Index, 'price']
                priceBil = price.replace(' tỷ', '').replace(',', '.').replace('~', '')
                priceBil = float(priceBil)
                priceVnd = priceBil * 1000000000
                priceMil = priceVnd / 1000000
                df.at[row.Index, 'priceBil'] = priceBil
                df.at[row.Index, 'priceVnd'] = priceVnd
                df.at[row.Index, 'priceMil'] = priceMil
            elif 'Thỏa thuận' in df.at[row.Index, 'price']:
                df.at[row.Index, 'price'] = None
                
        if pd.notnull(df.at[row.Index, 'priceExt']):
            priceExt = df.at[row.Index, 'priceExt'].replace('~', '')
            if 'triệu/m²' in df.at[row.Index, 'priceExt']:
                # ~48,63 triệu/m² -> 48.63
                priceExt = priceExt.replace(' triệu/m²', '').replace(',', '.')
                df.at[row.Index, 'pricePerM2'] = float(priceExt)


    for row in df.itertuples():
        # has price but no priceBil
        if pd.notnull(df.at[row.Index, 'price']) and pd.isnull(df.at[row.Index, 'priceBil']):
            if 'tỷ' in df.at[row.Index, 'price']:
                price = df.at[row.Index, 'price']
                priceBil = price.replace(' tỷ', '').replace(',', '.').replace('~', '')
                priceBil = float(priceBil)
                priceVnd = priceBil * 1000000000
                priceMil = priceVnd / 1000000
                df.at[row.Index, 'priceBil'] = priceBil
                df.at[row.Index, 'priceVnd'] = priceVnd
                df.at[row.Index, 'priceMil'] = priceMil
                
    # propertyType = 'Land' if bedroom is NaN else 'House'
    df['propertyType'] = df.apply(lambda x: 'Land' if pd.isnull(x['bedroom']) else 'House', axis=1)
    
    # propertyType = 'Apartment' if title contains 'chung cư' 'căn hộ'
    df['propertyType'] = df.apply(lambda x: 'Apartment' if 'chung cư' in x['title'].lower() or 'căn hộ' in x['title'].lower() else x['propertyType'], axis=1)
            
    # drop row if both price and priceExt are NaN
    df = df.dropna(subset=['price', 'priceExt'], how='all')
    
    return df