"""
Checks for the date handling and deduplication.

Run with ``python test_scraper.py``.
"""

import pandas as pd

from scraper import _format_date, _has_time, _process_end_time, drop_duplicates, format_times, relabel_instruments

# "25:10" is 01:10 the next day, while "18:25" is just a time
assert _format_date("12/10 18:25", "2020") == pd.Timestamp("2020-12-10 18:25")
assert _format_date("1/31 25:10", "2020") == pd.Timestamp("2020-02-01 01:10")

# End times are a time of day on the start date, unless the event goes past midnight
times = _process_end_time(
    pd.DataFrame({"Start": ["2010-01-22 21:12:47", "2010-01-22 23:59:00"], "End": ["21:12:55", "00:10:00"]}),
)
assert times["End"].tolist() == [pd.Timestamp("2010-01-22 21:12:55"), pd.Timestamp("2010-01-23 00:10:00")]

assert not _has_time("2025.09.03")
assert not _has_time("11/2")
assert all(map(_has_time, ["18:15", "4/4 05.50", "2010.05.18_00:00:00", "06-Apr-10 21:11:55"]))

# Events of the same instrument within 5 minutes of the first event of a group are combined, nothing is dropped
events = pd.DataFrame({
    "Start Time": pd.to_datetime(["2020-01-01 10:00", "2020-01-01 10:03", "2020-01-01 10:03", "2020-01-01 12:00"]),
    "End Time": pd.to_datetime(pd.Series(["2020-01-01 10:30", "2020-01-01 11:00", None, None])),
    "Instrument": ["AIA", "AIA", "HMI", "SDO"],
    "Source": ["a.txt", "b.txt", "b.txt", "c.txt"],
    "Comment": ["Event A", "Event B", "Event B", "Event C"],
    "Start Date Only": [False, False, False, True],
    "End Date Only": [False, False, False, True],
})
merged = drop_duplicates(events)
assert merged["Comment"].tolist() == ["Event A and Event B", "Event B", "Event C"]
assert merged["Source"].tolist() == ["a.txt and b.txt", "b.txt", "c.txt"]
assert merged["Instrument"].tolist() == ["AIA", "HMI", "SDO"]
assert merged["End Time"].iloc[0] == pd.Timestamp("2020-01-01 11:00")

# Dates without a time of day are written without one
assert format_times(merged["Start Time"], merged["Start Date Only"]).tolist() == [
    "2020-01-01 10:00:00",
    "2020-01-01 10:03:00",
    "2020-01-01",
]

# SDO events that only name one instrument are moved to that instrument
labels = pd.DataFrame({
    "Instrument": ["SDO", "SDO", "SDO", "AIA"],
    "Comment": ["HMI thermal adjustment", "EVE FOV and AIA/HMI flat field maneuvers", "Earth eclipse", "HMI too"],
})
assert relabel_instruments(labels).tolist() == ["HMI", "SDO", "SDO", "AIA"]
