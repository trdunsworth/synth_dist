import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from faker import Faker
import argparse

def generate_dataset(num_records = 10000):
    # Generate call_id column
    call_ids_full = [fake.uuid4() for _ in range(1, num_records + 1)]
    
    # Generate datetime column with random dates across 2024-2025
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31, 23, 59, 59)
    date_range = (end_date - start_date).total_seconds()
    
    random_seconds = np.random.randint(0, date_range, size=num_records)
    # Convert numpy integers to Python integers before using in timedelta
    datetimes_full = [start_date + timedelta(seconds=int(sec)) for sec in sorted(random_seconds)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'call_id': call_ids_full,
        'event_time': datetimes_full
    })

    