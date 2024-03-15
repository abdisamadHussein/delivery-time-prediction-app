import pickle
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

model = joblib.load(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl")

def predict_timetkaen(payload):
    df = pd.DataFrame([payload])
 
    df_num = df[['Delivery_person_Ratings', 'Vehicle_condition', 'distance', 'multiple_deliveries']]

    OH_encoder = OneHotEncoder(handle_unknown='ignore')

    to_binary_cols = df[['Weatherconditions', 'Road_traffic_density', 'Festival', 'City']]

    df_binary = pd.DataFrame(OH_encoder.fit_transform(to_binary_cols).toarray())
    binary_columns_names = OH_encoder.get_feature_names_out(to_binary_cols.columns)
    binary_columns_names = [name.split('_')[-1] for name in binary_columns_names]
    df_binary.columns = binary_columns_names

    if 'Yes' in df_binary.columns:
        df_binary['festival'] = 1.0
        df_binary.drop(columns=['Yes'], inplace=True)
    elif 'No' in df_binary.columns:
        df_binary['festival'] = 0.0
        df_binary.drop(columns=['No'], inplace=True)

    df_transformed = pd.concat([df_num, df_binary], axis=1)
    
    cols = ['Delivery_person_Ratings', 'Vehicle_condition',
            'distance', 'multiple_deliveries', 'Fog',
            'Sandstorms', 'Stormy', 'Sunny', 'Windy', 'High', 'Jam', 'Low ', 'Medium ', 'festival', 'Semi-Urban ', 'Urban ']

 
    missing_cols = set(cols) - set(df_transformed.columns)
    for col in missing_cols:
        df_transformed[col] = 0
    df_transformed = df_transformed[cols]

    pred = model.predict(df_transformed[['Delivery_person_Ratings', 'Vehicle_condition', 'distance', 'multiple_deliveries',
                                            'Sunny', 'Low ', 'festival', 'Urban ']])
    return pred

