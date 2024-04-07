from datetime import datetime

import holidays
import pandas as pd
import requests
import torch
import torch.nn as nn
from joblib import load

"""Root ARSO API URL"""
ARSO = "https://www.vreme.si/api/1.0/location/?location="


"""Name, revision, weather URL, display name"""
LOCATIONS = (
    ("kum", 1, "Kum", "Kum"),
    ("lovrenska-jezera", 1, "Rogla", "Lovrenška jezera"),
    ("osp", 1, "Kubed", "Osp"),
    ("storzic", 1, "Kranj", "Storžič"),
    ("vrsic", 1, "Vr%C5%A1i%C4%8D", "Vršič"),
)


# Grafana Auth
INFLUX_URL = "https://eu-central-1-1.aws.cloud2.influxdata.com/api/v2/write"
INFLUX_PASS = "G08mLIh5-rL0OVBA9H20IgUQ7UZXRMKbj6f8XMaaVnfqRBpoEjeTvGIaHzfnRR50j6hKv-lxYGH7KszRdHfVRg=="
INFLUX_ORG = "3b32e5cf87c07aef"
INFLUX_BUCKET = "Predictions"


# Model features and target
features = ["temperature", "rain", "month", "day_of_month", "day_of_week", "is_national_holiday", "is_school_holiday"]
target = "count"


# List of national holidays
national_holidays = holidays.SI()


# List of school holidays
school_holidays = [["2021-12-25", "2022-01-02"], ["2022-02-21", "2022-02-25"], ["2022-02-28", "2022-03-04"], ["2022-04-27", "2022-05-02"], ["2022-06-25", "2022-08-31"], ["2022-10-31", "2022-11-04"], ["2022-12-26", "2023-01-02"], ["2023-01-30", "2023-02-01"], ["2023-02-06", "2023-02-10"], ["2023-04-27", "2023-05-02"], ["2023-06-26", "2023-08-31"],["2023-10-30", "2023-11-3"], ["2023-12-25", "2024-01-02"], ["2024-02-19", "2024-02-23"], ["2024-02-26", "2023-03-01"], ["2024-4-27", "2023-05-02"], ["2024-06-26", "2024-08-31"]]
school_holidays = [[datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")] for start, end in school_holidays]


def is_school_holiday(date):
    """Is the date during school holiday"""
    for start, end in school_holidays:
        if start <= date <= end:
            return True
    return False


class HikerPredictor(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.softplus = nn.Softplus()

        self.double()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softplus(x)
        return x


# Instantiate the model
input_size = len(features)
hidden_size_1 = 64
hidden_size_2 = 32
output_size = 1
model = HikerPredictor(input_size, hidden_size_1, hidden_size_2, output_size)


for name, revision, url, display in LOCATIONS:
    today = int(datetime.now().timestamp())

    # Get forecast from ARSO
    forecast = requests.get(ARSO + url).json()["forecast3h"]["features"][0]["properties"]["days"]
    results = {}

    # Load our scaler
    scaler = load(f"../models/{name}-{revision:0>2}.bin")

    # Load our model
    model.load_state_dict(torch.load(f"../models/{name}-{revision:0>2}.pth"))
    model.eval()

    # Iterate over each day in the data
    for day in forecast:
        dt = datetime.fromisoformat(day["date"])
        total_temp = 0
        total_rain = 0
        count = 0

        # Iterate over the timeline for each day
        for timeline in day["timeline"]:
            # Extract the temperature and rain data
            total_temp += float(timeline["t"])
            total_rain += float(timeline["tp_acc"])
            count += 1

        # Calculate the average temperature for each day
        avg_temp = total_temp / count if count else 0

        # Prepare model parameters
        parameters = pd.DataFrame([{
            "temperature": avg_temp,
            "rain": total_rain,
            "month": dt.month,
            "day_of_month": dt.day,
            "day_of_week": dt.weekday(),
            "is_national_holiday": dt in national_holidays or dt.weekday() >= 5,
            "is_school_holiday": is_school_holiday(dt),
        }], columns=features)

        # Transform model parameters
        parameters = scaler.transform(parameters)
        parameters = torch.tensor(parameters, dtype=torch.float64)

        # Run our predictions
        predicted = model(parameters)
        results[day["date"]] = int(predicted)

    print(display, results)

    # Upload predictions to Influx
    for date, result in results.items():
        timestamp = int(datetime.fromisoformat(date).timestamp())

        data = f"hikers,location={name},generated={today} count={result} {timestamp}"
        headers = {"Authorization": f"Token {INFLUX_PASS}"}
        params = {"org": INFLUX_ORG, "bucket": INFLUX_BUCKET, "precision": "s"}

        response = requests.post(INFLUX_URL, data=data, headers=headers, params=params)

        print(data)
        print(response.status_code, response.text)
