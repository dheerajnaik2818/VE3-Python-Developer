import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load and preprocess the data
def load_and_train_model():
    df = pd.read_csv('dataset/housing_data.csv')
    
    X = df[['bedrooms', 'bathrooms', 'size', 'location']]
    y = df['price']
    
    # Preprocessing
    numeric_features = ['bedrooms', 'bathrooms', 'size']
    categorical_features = ['location']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'model.pkl')

def load_model():
    return joblib.load('model.pkl')

def predict_price(features):
    model = load_model()
    return model.predict(features)[0]

if __name__ == '__main__':
    load_and_train_model()
