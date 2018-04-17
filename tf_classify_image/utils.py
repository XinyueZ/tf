import pandas as pd
import numpy as np
from downloader import Downloader
from config import (DATA_SOURCE_FILE, OBJECT_NAME, HOST_PATH,
                    CSV_COLUMN_NAMES, OBJECT_WIDTH, OBJECT_HEIGHT)


def download_all_images_photos():
    ds = pd.read_table(DATA_SOURCE_FILE, dtype={
                       'LEVEL1': str, 'LEVEL2': str, "LEVEL3": str})
    i = 0
    while i < ds.shape[0]:
        line = ds.ix[i]
        path = HOST_PATH.format(
            line["LEVEL1"].strip(),
            line["LEVEL2"].strip(),
            line["LEVEL3"].strip()
        )
        object_name = OBJECT_NAME.format(
            line["MULTIMEDIA_ID"],
            OBJECT_WIDTH,
            OBJECT_HEIGHT,
            line["EXTENSION"],
        )

        downloader = Downloader(path, "./img")
        downloader.download(object_name)
        i += 1


download_all_images_photos()
