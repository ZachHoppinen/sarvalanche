import pytest

import pandas as pd
from pathlib import Path

from sarvalanche.io.loaders import Sentinel1RTCLoader

def test_parse_time_opera_rtc(tmp_path):
    # Example OPERA RTC file
    fname = "OPERA_L2_RTC-S1_T093-197867-IW3_20210426T013646Z_20250903T222329Z_S1A_30_v1.0_VV.tif"
    file_path = tmp_path / fname
    file_path.write_text("dummy")  # create empty file for path

    # Instantiate the loader (or DummyLoader)
    loader = Sentinel1RTCLoader()

    t = loader._parse_time(file_path)
    assert t is not None
    assert isinstance(t, pd.Timestamp)
    # Check exact timestamp from filename
    assert t == pd.Timestamp("2021-04-26T01:36:46")
