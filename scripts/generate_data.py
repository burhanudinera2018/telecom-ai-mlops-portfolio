# scripts/generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

data = {
    'customer_id': range(1, n+1),
    'tenure': np.random.randint(1, 72, n),
    'monthly_charges': np.random.uniform(20, 120, n),
    'total_charges': np.random.uniform(100, 5000, n),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
    'paperless_billing': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'monthly_avg_gb': np.random.exponential(50, n).astype(int),
    'num_complaints': np.random.poisson(0.5, n),
    'num_calls_to_care': np.random.poisson(2, n),
    'has_internet': np.random.choice([0, 1], n, p=[0.2, 0.8]),
    'has_streaming_tv': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'has_streaming_movies': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'churn': np.random.choice([0, 1], n, p=[0.73, 0.27])  # 27% churn rate
}

df = pd.DataFrame(data)
df.to_csv('data/raw/customer_data.csv', index=False)
print("✅ Data generated")