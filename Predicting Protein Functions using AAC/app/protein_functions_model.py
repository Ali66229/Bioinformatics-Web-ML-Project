import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_data():
    # Load AAC features and original labels
    features_file = "aac_features.csv"
    labels_file = "processed_mappings.csv"

    # Load feature data
    aac_df = pd.read_csv(features_file)

    # Load labels and merge with features
    labels_df = pd.read_csv(labels_file)
    data = aac_df.merge(labels_df, on="Protein_ID")

    # Filter GO terms with at least 10 associated proteins
    filtered_data = data.groupby("GO_Terms").filter(lambda x: len(x) >= 10)

    # Separate features (X) and labels (y)
    X = filtered_data.iloc[:, 1:-1]  # All AAC columns
    y = filtered_data["GO_Terms"]    # GO_Terms column

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

    return X_resampled, y_resampled, label_encoder

def train_model():
    X_resampled, y_resampled, label_encoder = load_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Return the trained model and label encoder
    return model, label_encoder

def make_prediction(model, label_encoder, input_data):
    # Ensure the input data is in the correct format
    input_features = input_data.iloc[:, 1:].values

    # Make predictions
    predictions = model.predict(input_features)
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Return the predictions
    return predicted_labels.tolist()
