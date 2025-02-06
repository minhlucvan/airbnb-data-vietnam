import pandas as pd

def load_fb_data(selected_state, selected_city):
    # Load data and geojson based on the selections
    df = pd.read_csv(f'data/{selected_state}/{selected_city}/fb_cleaned.csv')
    
    # remove all attachments columns
    df = df.drop(columns=[col for col in df.columns if 'attachments' in col])
    
    # drop all sharedPost columns
    df = df.drop(columns=[col for col in df.columns if 'sharedPost' in col])
    
    # drop all textReference columns
    df = df.drop(columns=[col for col in df.columns if 'textReference' in col])
    
    # time
    df['time'] = pd.to_datetime(df['time'])
    
    # date
    df['date'] = df['time'].dt.date
    
    return df