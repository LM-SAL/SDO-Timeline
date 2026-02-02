"""
Basic configuration file for SDO data scraping.
"""

from datetime import UTC, datetime

__all__ = ["CY_END", "DATASETS", "MAP_4", "TIME_FORMATS"]

CY_END = int(str(datetime.now(tz=UTC).year + 1)[2:])
TIME_FORMATS = [
    "%d-%b-%y %H:%M:%S",  # 06-Apr-10 21:11:55
    "%Y.%m.%d",  # 2010.05.18
    "%y-%j-%H:%M:%S",  # YY-DOY-HH:MM:SS, 10-096-06:26:28
    "%Y.%m.%d_%H:%M:%S",  # 2010.11.10_06:01:20
    "%d-%b-%Y %H:%M:%S",  # 9-Apr-2010 07:30:00
]
MAP_4 = {
    0: "Roll Maneuvers",
    1: "Momentum Management Maneuvers",
    2: "Station-Keeping Maneuvers",
    3: "Eclipse Season Begins",
    4: "Eclipse Season Ends",
    5: "Battery Voltage Adjustments",
    6: "EVE Crucform Maneuvers",
    7: "EVE FOV & Offpoint Maneuvers",
    8: "GT/PZT Calibrations",
    9: "CCD Bakeout",
    10: "Regulus Offpoint",
    11: "Comet Offpoint",
    12: "Camera Anomaly",
    13: "Transit Ops/Transit",
    14: "IRU Operations",
    15: "load shed",
    16: "Load/Software Anomaly/Error",
    17: "Table Parity Error",
    18: "Misc Instrument Errors Not Listed Above",
    19: "Misc Tests/Special Ops",
}
DATASETS = {
    "jsocobs_info": {
        "fURL": "https://aia.lmsal.com/public/jsocobs_info{}.html",
        "RANGE": range(10, CY_END),
    },
    "spacecraft_night": {
        "URL": "https://aia.lmsal.com/public/sdo_spacecraft_night.txt",
        "SKIP_ROWS": [0, 1, 2, 3],
    },
    "text_block_1": {
        "URL": "./data_1.txt",
    },
    "text_block_2": {
        "URL": "./data_2.txt",
    },
    "text_block_3": {
        "URL": "./data_3.txt",
    },
    "text_block_4": {
        "URL": "./data_4.txt",
    },
    "hmi_obs_cov": {
        "fURL": "http://jsoc.stanford.edu/doc/data/hmi/cov2/cov{}.html",
        "RANGE": range(10, CY_END),
        "MONTH_RANGE": range(1, 13),
    },
    # This site has a whole range of text files and its easier to scrape the urls that way.
    # Assumption is that each text file on this page has the same structure
    "jsocinst_calibrations": {
        "URL": "https://aia.lmsal.com/public/jsocinst_calibrations.html",
        "SKIP_ROWS": [0],
        "SCRAPE": True,
    },
}
