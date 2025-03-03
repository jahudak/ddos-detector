import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


class DDoSDataPreprocessor:
    def __init__(self):
        print("Data Processor initialized.")
        self.components_df = None
        self.events_df = None
        self.preprocessed_df = None
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def process_data(self, components, events):
        self.components_df = pd.read_csv(components)
        self.events_df = pd.read_csv(events)

        self.merge_data()

        # Remove irrelevant columns
        self.remove_irrelevant_columns()

        # Remove times with 0 values
        self.preprocessed_df = self.preprocessed_df[
            (self.preprocessed_df["Start time"] != "0")
            & (self.preprocessed_df["End time"] != "0")
        ]

        self.convert_time_to_date_time()
        self.extract_time_data()
        self.convert_time_to_cyclical()
        self.normalize_data()
        self.convert_to_float32()  # might not be necessary
        # self.encode_data()
        # Warning: encoding increases the size of the data !!!

        # Perfrom some Fourier transformation
        # just an example
        # self.fourier_transformation(self.preprocessed_df["Start_time_hour_cos"])

        # self.preprocessed_df.to_csv("data/preprocessed_data.csv", index=False)

        # using parquet for better performance
        self.preprocessed_df.to_parquet(
            "data/preprocessed_data.parquet", compression="snappy"
        )

        print("Preprocessed data saved.")

    def merge_data(self):
        common_columns = set(self.components_df.columns) & set(self.events_df.columns)
        common_columns.discard("Attack ID")

        self.events_df = self.events_df.drop(columns=common_columns)

        self.preprocessed_df = pd.merge(
            self.components_df, self.events_df, on="Attack ID"
        )

    def remove_irrelevant_columns(self):
        columns_to_drop = ["Card", "Significant flag", "Whitelist flag"]
        self.preprocessed_df = self.preprocessed_df.drop(columns=columns_to_drop)

    def convert_time_to_date_time(self):
        self.preprocessed_df["Start time"] = pd.to_datetime(
            self.preprocessed_df["Start time"]
        )

        self.preprocessed_df["End time"] = pd.to_datetime(
            self.preprocessed_df["End time"]
        )

    def extract_time_data(self):
        # Extract time data
        self.preprocessed_df["Start_time_hour"] = self.preprocessed_df[
            "Start time"
        ].dt.hour
        self.preprocessed_df["Start_time_weekday"] = self.preprocessed_df[
            "Start time"
        ].dt.weekday
        self.preprocessed_df["Start_time_dayofyear"] = self.preprocessed_df[
            "Start time"
        ].dt.dayofyear

        self.preprocessed_df["End_time_hour"] = self.preprocessed_df["End time"].dt.hour
        self.preprocessed_df["End_time_weekday"] = self.preprocessed_df[
            "End time"
        ].dt.weekday
        self.preprocessed_df["End_time_dayofyear"] = self.preprocessed_df[
            "End time"
        ].dt.dayofyear

    def convert_time_to_cyclical(self):
        # Convert Start time to cyclical
        self.preprocessed_df["Start_time_hour_sin"] = np.sin(
            2 * np.pi * self.preprocessed_df["Start time"].dt.hour / 24
        )
        self.preprocessed_df["Start_time_hour_cos"] = np.cos(
            2 * np.pi * self.preprocessed_df["Start time"].dt.hour / 24
        )

        self.preprocessed_df["Start_time_weekday_sin"] = np.sin(
            2 * np.pi * self.preprocessed_df["Start time"].dt.weekday / 7
        )
        self.preprocessed_df["Start_time_weekday_cos"] = np.cos(
            2 * np.pi * self.preprocessed_df["Start time"].dt.weekday / 7
        )

        self.preprocessed_df["Start_time_dayofyear_sin"] = np.sin(
            2 * np.pi * self.preprocessed_df["Start time"].dt.dayofyear / 365
        )
        self.preprocessed_df["Start_time_dayofyear_cos"] = np.cos(
            2 * np.pi * self.preprocessed_df["Start time"].dt.dayofyear / 365
        )

        # Convert End time to cyclical
        self.preprocessed_df["End_time_hour_sin"] = np.sin(
            2 * np.pi * self.preprocessed_df["End time"].dt.hour / 24
        )
        self.preprocessed_df["End_time_hour_cos"] = np.cos(
            2 * np.pi * self.preprocessed_df["End time"].dt.hour / 24
        )

        self.preprocessed_df["End_time_weekday_sin"] = np.sin(
            2 * np.pi * self.preprocessed_df["End time"].dt.weekday / 7
        )
        self.preprocessed_df["End_time_weekday_cos"] = np.cos(
            2 * np.pi * self.preprocessed_df["End time"].dt.weekday / 7
        )

        self.preprocessed_df["End_time_dayofyear_sin"] = np.sin(
            2 * np.pi * self.preprocessed_df["End time"].dt.dayofyear / 365
        )
        self.preprocessed_df["End_time_dayofyear_cos"] = np.cos(
            2 * np.pi * self.preprocessed_df["End time"].dt.dayofyear / 365
        )

    def fourier_transformation(self, time_series_data):
        number_of_samples = len(time_series_data)
        sample_placing = 1.0

        yf = fft(time_series_data)
        xf = fftfreq(number_of_samples, sample_placing)[: number_of_samples // 2]

        plt.figure(figsize=(12, 6))
        plt.plot(xf, 2.0 / number_of_samples * np.abs(yf[0 : number_of_samples // 2]))
        plt.title(f"Frequency Domain Representation of {time_series_data.name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def normalize_data(self):
        columns_to_normalize = [
            "Packet speed",
            "Data speed",
            "Avg packet len",
            "Source IP count",
            "Avg source IP count",
        ]
        self.preprocessed_df[columns_to_normalize] = self.preprocessed_df[
            columns_to_normalize
        ].apply(pd.to_numeric, errors="coerce")  # ensure that the data is numeric
        self.preprocessed_df[columns_to_normalize] = self.scaler.fit_transform(
            self.preprocessed_df[columns_to_normalize]
        )

    def convert_to_float32(self):
        # Note: might not be necessary but for optimazing memory usage it can be useful
        columns_to_convert_to_float32 = [
            "Packet speed",
            "Data speed",
            "Avg packet len",
            "Source IP count",
            "Avg source IP count",
        ]
        self.preprocessed_df[columns_to_convert_to_float32] = self.preprocessed_df[
            columns_to_convert_to_float32
        ].astype(np.float32)

    def encode_data(self):
        # label encoding, excluded for now as the data will be too large to handle
        # might be useful for some models
        self.preprocessed_df["Attack code"] = self.label_encoder.fit_transform(
            self.preprocessed_df["Attack code"]
        )
        self.preprocessed_df = pd.get_dummies(
            self.preprocessed_df, columns=["Victim IP"]
        )
