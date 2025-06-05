import json
import os

import pandas as pd


def read_json_to_df(relative_path_to_file):
    current_dir = os.getcwd()
    json_path = os.path.join(current_dir, relative_path_to_file)

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    data = [
        {
            "image_index": int(key),
            "true": sorted(value["manual"] + value["pred"])
        }
        for key, value in json_data.items()
    ]

    df = pd.DataFrame(data).sort_values(by="image_index").reset_index(drop=True)

    return df


class ReaderJson:

    def __init__(self, relative_path_to_file: str):
        self.df = read_json_to_df(relative_path_to_file)

    def __getitem__(self, item):
        return self.df.iloc[item]['true'], self.df.iloc[item]['pred']

    def print_data(self):
        print(self.df)

    def return_data(self):
        return self.df


if __name__ == '__main__':
    relative_path = 'raw_data/Données_CN_V1/SCORPIO_LWIR/SCORPIO-LW_C1_REF_N1.json'
    a = ReaderJson(relative_path)

    data = a.return_data()
    print(data)
