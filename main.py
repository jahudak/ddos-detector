from ddos_data_preprocessor import DDoSDataPreprocessor
from ddos_data_module import DDoSDataModule
import os


def main():
    # if preprocessed_data.paq is not available, run the following code
    # else skip this step

    if not os.path.exists("data/preprocessed_data.parquet"):
        data_preprocessor = DDoSDataPreprocessor()
        data_preprocessor.process_data(
            "data/SCLDDoS2024_SetA_components.csv", "data/SCLDDoS2024_SetA_events.csv"
        )

    data_module = DDoSDataModule(
        data_path="data/preprocessed_data.parquet", batch_size=64
    )

    data_module.load_data()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()


if __name__ == "__main__":
    main()
