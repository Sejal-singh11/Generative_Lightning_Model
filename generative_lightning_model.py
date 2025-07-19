print("Running generative lightning model...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Step 1: Generate Synthetic Lightning Data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'day_of_year': np.random.randint(1, 366, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'temperature': np.random.uniform(20, 40, n_samples),
        'humidity': np.random.uniform(40, 90, n_samples),
        'latitude': np.random.uniform(25, 50, n_samples),
        'longitude': np.random.uniform(-125, -65, n_samples),
    }

    # Label: likelihood of lightning (simulated)
    lightning_prob = (
        0.3 * np.sin(2 * np.pi * data['day_of_year'] / 365) +
        0.2 * (data['humidity'] - 40) / 50 +
        0.3 * (data['temperature'] - 20) / 20 +
        np.random.normal(0, 0.1, n_samples)
    )

    data['lightning_prob'] = np.clip(lightning_prob, 0, 1)
    return pd.DataFrame(data)

# Step 2: Train Model
df = generate_synthetic_data()

X = df[['day_of_year', 'hour_of_day', 'temperature', 'humidity']]
y = df['lightning_prob']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Generate New Data (Generative Part)
def generate_new_predictions(n_samples=100):
    new_data = generate_synthetic_data(n_samples)
    X_new = new_data[['day_of_year', 'hour_of_day', 'temperature', 'humidity']]
    new_data['predicted_lightning'] = model.predict(X_new)
    return new_data

# Step 4: Visualize
generated = generate_new_predictions(200)

plt.figure(figsize=(10, 6))
plt.scatter(generated['longitude'], generated['latitude'], c=generated['predicted_lightning'], cmap='hot', s=50)
plt.colorbar(label='Predicted Lightning Probability')
plt.title('Generated Lightning Event Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()
