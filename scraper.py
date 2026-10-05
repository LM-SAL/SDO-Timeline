"""
Simple scraper for SDO event timeline data.
"""

import contextlib
import io
import re
import sys
from datetime import UTC, date, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import icalendar
import pandas as pd
import recurring_ical_events
import requests
from bs4 import BeautifulSoup
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from config import DATASETS, MAP_4, TIME_FORMATS

if TYPE_CHECKING:
    from pandas.api.typing import NaTType

logger.remove()
logger.add(sys.stderr, level="INFO")

# Month/day with an optional year and time, e.g., "4/4", "9/4/2010 19:47" or "4/4 05.50"
DATE_PATTERN = re.compile(r"\d{1,2}/\d{1,2}(?:/\d{2,4})?(?:\s+\d{1,2}[:.]\d{2})?")
SESSION = requests.Session()
SESSION.mount(
    "http",
    HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])),
)


def _get(url: str) -> str:
    """
    Download a URL, retrying on server errors.

    Parameters
    ----------
    url : str
        URL to download.

    Returns
    -------
    str
        The text of the response.
    """
    response = SESSION.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def _has_time(date: str) -> bool:
    """
    Check if a date string has a time of day.

    Parameters
    ----------
    date : str
        Date string, e.g., "18:15", "4/4 05.50" or "2010.05.18_00:00:00" have times while "11/2" does not.

    Returns
    -------
    bool
        True if there is a time of day.
    """
    return re.search(r"(^|[\s_])\d{1,2}[:.]\d{2}", date) is not None


def _format_date(
    date: str, year: str | None = None, start_date: datetime | NaTType | None = None
) -> pd.Timestamp | NaTType:
    """
    Format the given date.

    Parameters
    ----------
    date : str
        Date string from the html file.
    year : str, optional
        The year of the provided dates, if it is not present in the date.
    start_date : datetime.datetime
        The start date of the dataset.
        Defaults to None.

    Returns
    -------
    pandas.Timestamp
        New date.

    Raises
    ------
    ValueError
        If ``start_date`` is not provided but required.
    """
    if year is None:
        return pd.Timestamp(date)
    # Only date e., '11/2' assuming month/day
    if len(date) in {4, 5}:
        # Only a time, which is on the start date
        if "/" not in date:
            if start_date is None:
                msg = f"Start date is required for this format: {date}"
                raise ValueError(msg)
            new_date = pd.Timestamp(str(start_date.date()) + " " + date)
        else:
            new_date = pd.Timestamp(f"{year}-{date}")
    # Only time - e.g., '18:15'
    # Year missing - e.g., '12/10 18:15'
    elif len(date) in {9, 10, 11, 12}:
        day, time = date.split(" ")[:2]
        # Some times are past midnight, e.g., "25:10" is 01:10 the next day
        hour, _, rest = time.partition(":")
        new_date = pd.Timestamp(f"{day}/{year} {int(hour) % 24}:{rest}") + pd.Timedelta(days=int(hour) // 24)
    # Anything else, e.g., a full date, only the first date is used
    else:
        try:
            # This catches 2010.05.01 - 02
            new_date = pd.Timestamp(date.split("-", maxsplit=1)[0])
        except ValueError:
            idx = len(date) // (len(date) // 10)
            new_date = pd.Timestamp(f"{year}-{date[:idx]}")
    return new_date


def _clean_date(date: str, *, extra_replace: bool = False) -> str:
    """
    Remove any non-numeric characters from the date.

    Parameters
    ----------
    date : str
        Date to clean.
    extra_replace : bool, optional
        Whether to replace more characters, by default False.

    Returns
    -------
    str
        Cleaned date.
    """
    date = (
        " "
        .join(date.split())
        .replace("UT", "")
        .replace(" TBD", "")
        .replace("ongoing", "")
        .replace("AIA", "")
        .replace("HMI", "")
        # Very specific dates
        # 2018-10/16 10:00 - 21:00
        .replace("- 21:00", "")
    ).split("-")[0]
    if extra_replace:
        # Some hours are 4/4 05.50 so we replace them here
        # However, sometimes the date is 2010.05.01 - 02
        date = date.replace(".", ":")
    return date


def _process_time(data: pd.DataFrame, column: int = 0) -> pd.DataFrame:
    """
    Reformats all the time columns to have a consistent format.

    This modifies the dataframe in place.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with timestamps.
    column : int, optional
        The column to process, by default 0.

    Returns
    -------
    pd.DataFrame
        The dataframe with reformatted timestamps.

    Raises
    ------
    ValueError
        If no suitable time format is found.
    """
    for time_format in TIME_FORMATS:
        try:
            data[data.columns[column]] = data.iloc[:, column].apply(
                lambda x, time_format=time_format: datetime.strptime(x, time_format)  # ruff: ignore[call-datetime-strptime-without-zone]
            )
            return data  # ruff: ignore[try-consider-else]
        except Exception as e:  # ruff: ignore[blind-except]
            logger.debug(f"Time format {time_format} did not work for {data.iloc[0, column]} for column {column}: {e}")
    msg = f"Could not find a suitable time format: {data.iloc[0, column]} or failed assignment to DataFrame."
    raise ValueError(
        msg,
    )


def _process_end_time(data: pd.DataFrame, column: int = 1) -> pd.DataFrame:
    # End times are only a time of day, so take the date from the start time
    start = pd.to_datetime(data.iloc[:, 0])
    end = pd.to_datetime(start.dt.strftime("%m/%d/%Y") + " " + data.iloc[:, column])
    # The event went past midnight if it ends before it starts
    data[data.columns[column]] = end.where(end >= start, end + pd.Timedelta(days=1))
    return data


def _process_data(data: pd.DataFrame, filepath: str, title: str) -> pd.DataFrame:
    """
    Add the instrument and comment columns.

    Certain files have no comments or have a comment (or FSN) in the third column.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to process.
    filepath : str
        Path to the file.
    title : str
        The first line of the file, used as the comment with any comment on the row added to it.

    Returns
    -------
    pd.DataFrame
        Processed dataframe.
    """
    data = data.rename(columns={"Start Date/Time": "Start Time", "Unnamed: 2": "Comment"})
    if "FSN" in data.columns:
        data["Comment"] = "FSN " + data.pop("FSN").astype("string").str.removesuffix(".0")
    if "Comment" in data.columns:
        data["Comment"] = (title + ": " + data["Comment"].astype("string")).fillna(title)
    else:
        data["Comment"] = title
    data["Instrument"] = "AIA" if "AIA" in filepath else "HMI" if "HMI" in filepath else "SDO"
    return data.loc[:, ["Start Time", "End Time", "Instrument", "Comment"]]


def _reformat_data(data: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """
    Due to the fact that the text files are not consistent.

    We need to reformat them.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to reformat.
    filepath : str
        Path to the file.

    Returns
    -------
    pd.DataFrame
        Reformatted dataframe.
    """
    if "_1" in filepath:
        times = data[0].str.split(expand=True)
        data = pd.DataFrame({"Start Time": times[0], "End Time": times[1], "Comment": data[1]})
    elif "_2" in filepath or "_3" in filepath:
        data.columns = ["Start Time", "Comment"]
    elif "_4" in filepath:
        data = data.iloc[:, [1, 0]]
        data.columns = ["Start Time", "Comment"]
        data["Comment"] = data["Comment"].apply(lambda x: MAP_4[x])
    return data


def process_txt(filepath: str, skip_rows: list[int] | None) -> pd.DataFrame:
    """
    Process a text file.

    Parameters
    ----------
    filepath : str
        File path of the text file.
    skip_rows : list, None
        What rows to skip.

    Returns
    -------
    pd.DataFrame
        Dataframe with the data from the text file.
    """
    if "http" in filepath:
        text = _get(filepath)
        title = text.splitlines()[0].strip()
        new_data = pd.read_fwf(
            io.StringIO(text),
            header=None if "sdo_spacecraft_night" in filepath else 0,
            skiprows=skip_rows,
        )
        new_data = _process_time(new_data)
        new_data[new_data.columns[1]] = new_data.iloc[:, 1].apply(
            lambda x: pd.Timestamp(str(x).replace(":stol_", "")) if ":stol_" in str(x) else x,
        )
        if "sdo_spacecraft_night" not in filepath:
            new_data = _process_end_time(new_data)
        if len(new_data.columns) in {2, 3}:
            new_data = _process_data(new_data, filepath, title)
        elif len(new_data.columns) > 3:  # ruff: ignore[magic-value-comparison]
            # Comments get split over several columns, except in the night file where column 2 is the duration
            # and column 3 marks the lunar transits
            first = 3 if "sdo_spacecraft_night" in filepath else 2
            comment = new_data.iloc[:, first:].apply(lambda row: " ".join(row.dropna().astype(str)), axis=1)
            new_data = new_data.iloc[:, [0, 1]]
            new_data.columns = ["Start Time", "End Time"]
            new_data["Comment"] = comment.replace("", None)
            with contextlib.suppress(Exception):
                new_data = _process_time(new_data, 1)
            new_data = _process_data(new_data, filepath, title)
    else:
        new_data = pd.read_csv(filepath, header=None, sep="    ", skiprows=skip_rows, engine="python")
        new_data = _reformat_data(new_data, filepath)
        new_data["Start Date Only"] = ~new_data["Start Time"].map(_has_time)
        new_data = _process_time(new_data)
        if "End Time" in new_data.columns:
            new_data = _process_time(new_data, 1)
        new_data["Instrument"] = new_data["Comment"].map(
            lambda x: "AIA" if "AIA" in x else "HMI" if "HMI" in x else None,
        )
    new_data["Source"] = filepath.rsplit("/", maxsplit=1)[-1]
    return new_data


def process_events(url: str) -> pd.DataFrame:
    """
    Process the SDO spacecraft operations file.

    It lists events under a heading for each type of event.

    Parameters
    ----------
    url : str
        URL of the text file.

    Returns
    -------
    pd.DataFrame
        Dataframe with the data from the text file.
    """
    events, heading = [], None
    for line in _get(url).splitlines()[1:]:
        if match := re.match(r'\s*startTime="([^"]+)".*stopTime="([^"]+)"\s*(.*)', line):
            start, stop, note = match.groups()
            events.append({
                "Start Time": datetime.strptime(start, "%y-%j-%H:%M:%S.%f"),  # ruff: ignore[call-datetime-strptime-without-zone]
                "End Time": datetime.strptime(stop, "%y-%j-%H:%M:%S.%f"),  # ruff: ignore[call-datetime-strptime-without-zone]
                "Instrument": "SDO",
                "Comment": f"{heading}: {note.strip()}" if note.strip() else heading,
            })
        elif line.strip():
            heading = line.strip()
    new_data = pd.DataFrame(events)
    new_data["Source"] = url.rsplit("/", maxsplit=1)[-1]
    return new_data


def _ics_time(value: date) -> pd.Timestamp | NaTType:
    # All-day events are dates, all other times are converted to UTC like the other sources
    if isinstance(value, datetime) and value.tzinfo:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return pd.Timestamp(value)


def process_ics(url: str) -> pd.DataFrame:
    """
    Process a JSOC Google calendar, including every repeat of recurring events.

    Parameters
    ----------
    url : str
        URL of the iCalendar file.

    Returns
    -------
    pd.DataFrame
        Dataframe with the events from the calendar.
    """
    calendar = icalendar.Calendar.from_ical(_get(url))
    name = str(calendar.get("X-WR-CALNAME"))
    # ponytail: fixed look-ahead, recurring "TBD" calibrations have no end date
    stop = datetime.now(tz=UTC) + timedelta(days=180)
    events = []
    for event in recurring_ical_events.of(calendar).between(datetime(2010, 1, 1, tzinfo=UTC), stop):
        start, end = event.start, event.end
        all_day = not isinstance(start, datetime)
        summary = str(event.get("SUMMARY", "")).strip()
        description = " ".join(BeautifulSoup(str(event.get("DESCRIPTION", "")), "html.parser").get_text(" ").split())
        events.append({
            "Start Time": _ics_time(start),
            # The end of an all-day event is the day after it finishes
            "End Time": _ics_time(max(start, end - timedelta(days=1)) if all_day else end),
            "Instrument": name if name in {"AIA", "HMI"} else "SDO",
            "Comment": f"{summary}: {description}" if description else summary,
            "Start Date Only": all_day,
            "End Date Only": all_day,
        })
    new_data = pd.DataFrame(events)
    new_data["Source"] = f"JSOC {name} calendar"
    return new_data


def process_html(url: str) -> pd.DataFrame | None:  # ruff: ignore[too-many-locals]
    """
    Process an html file.

    Parameters
    ----------
    url : str
        URL of the html file.

    Returns
    -------
    pd.DataFrame, None
        Dataframe with the data from the html file, or None if there is none.
    """
    response = SESSION.get(url, timeout=60)
    if response.status_code == 404:  # ruff: ignore[magic-value-comparison]
        logger.warning(f"URL not found: {url}")
        return None
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find_all("table")
    # There should be two html tables for this URL
    if len(table) == 1 and "jsocobs_info" in url:
        return None
    table = table[-1]
    rows = table.find_all("tr")
    year = match[1] if (match := re.search(r"jsocobs_info(\d{4})\.html", url)) else None
    # These HTML tables are by column and not by row
    if "hmi/cov2/" in url:
        new_rows = rows[0].text.split("\n\n")
        # Time is one single element whereas each event text is a separate element
        dates, text = new_rows[0].strip().split("\n"), new_rows[1:-1]
        if dates in [[""], [" "], []]:
            logger.warning(f"No data found for {url}")
            return None
        instrument = ["HMI" if "HMI" in new_row else "AIA" if "AIA" in new_row else "SDO" for new_row in text]
        comment = [new_row.replace("\n", " ") for new_row in text]
        new_dates = dates.copy()
        for date in dates:
            # Workaround for http://jsoc.stanford.edu/doc/data/hmi/cov2/cov202503.html
            # where the date just "multiple"
            if "multiple" in date:
                new_dates[new_dates.index(date)] = date.replace("multiple", new_dates[0])
        new_dates = [_clean_date(date) for date in new_dates]
        new_data = pd.DataFrame({
            "Start Time": [_format_date(date, year) for date in new_dates],
            "End Time": None,
            "Instrument": instrument,
            "Comment": comment,
            "Start Date Only": [not _has_time(date) for date in new_dates],
        })
    else:
        events = []
        extra_replace = "jsocobs_info" in url
        for row in rows[1:]:
            # The columns are: start time, end time (can be blank), event, AIA script, AIA description,
            # HMI script, HMI FTS and HMI description
            text = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])] + [""] * 8
            comment = text[2] or text[4] or text[7]
            instrument = "SDO" if text[2] else "AIA" if text[4] else "HMI"
            # A cell can list several dates, e.g., "12/30 21:45 12/22 21:15", which are separate events
            starts = DATE_PATTERN.findall(text[0]) or [text[0]]
            ends = DATE_PATTERN.findall(text[1])
            if len(ends) != len(starts):
                ends = [text[1]] if len(starts) == 1 else [""] * len(starts)
            for start_text, end_text in zip(starts, ends, strict=True):
                start_date = _clean_date(start_text, extra_replace=extra_replace)
                end_date = _clean_date(end_text, extra_replace=extra_replace) if len(end_text) > 1 else "NaT"
                start = _format_date(start_date, year)
                events.append({
                    "Start Time": start,
                    "End Time": _format_date(end_date, year, start),
                    "Instrument": instrument,
                    "Comment": comment,
                    "Start Date Only": not _has_time(start_date),
                    "End Date Only": not _has_time(end_date),
                })
        new_data = pd.DataFrame(events)
    new_data["Source"] = url.rsplit("/", maxsplit=1)[-1]
    return new_data


def scrape_url(url: str) -> list[str]:
    """
    Scrapes a URL for all the text files.

    Parameters
    ----------
    url : str
        URL to scrape.

    Returns
    -------
    list
        List of all the urls scraped.
    """
    soup = BeautifulSoup(_get(url), "html.parser")
    hrefs = [str(link["href"]) for link in soup.find_all("a", href=True)]
    return [urljoin(url, href) for href in hrefs if "txt" in href]


def _join(values: pd.Series) -> str:
    return " and ".join(dict.fromkeys(values.astype(str)))


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine events that start within 5 minutes of the first event of a group.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to deduplicate, sorted by start time.

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe.
    """
    groups, first = [], None
    for start in data["Start Time"]:
        if first is None or start - first > pd.Timedelta("5 minute"):
            first = start
        groups.append(first)
    return (
        data
        .groupby(groups, sort=False)
        .agg({
            "Start Time": "first",
            "End Time": "max",
            "Instrument": lambda x: x.iloc[0] if (x == x.iloc[0]).all() else "SDO",
            "Source": _join,
            "Comment": _join,
            "Start Date Only": "all",
            "End Date Only": "all",
        })
        .reset_index(drop=True)
    )


def format_times(times: pd.Series, date_only: pd.Series) -> pd.Series:
    """
    Format times as strings.

    Times are written without the time of day if the source did not have one.

    Parameters
    ----------
    times : pd.Series
        Times to format.
    date_only : pd.Series
        Which times have no time of day.

    Returns
    -------
    pd.Series
        Formatted times.
    """
    return times.dt.strftime("%Y-%m-%d %H:%M:%S").where(~date_only, times.dt.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    this_month = datetime.now(tz=UTC).strftime("%Y%m")
    frames = []
    for dataset_name, block in DATASETS.items():
        logger.info(f"Scraping {dataset_name}")
        if block.get("SCRAPE"):
            urls = scrape_url(block["URL"])
        elif block.get("MONTH_RANGE"):
            months = [f"20{i:02}{j:02}" for i, j in product(block["RANGE"], block["MONTH_RANGE"])]
            urls = [block["fURL"].format(month) for month in months if month <= this_month]
        elif block.get("RANGE"):
            urls = [block["fURL"].format(f"20{i:02}") for i in block["RANGE"]]
        else:
            urls = [block["URL"]]
        for url in sorted(urls):
            logger.info(f"Parsing {url}")
            if "sdo_spacecraft_events" in url:
                frames.append(process_events(url))
            elif url.endswith(".ics"):
                frames.append(process_ics(url))
            elif "txt" in url:
                frames.append(process_txt(url, block.get("SKIP_ROWS")))
            elif "html" in url:
                frames.append(process_html(url))
            else:
                msg = f"Unknown file type for {url}"
                raise ValueError(msg)

    final_timeline = pd.concat([frame for frame in frames if frame is not None], ignore_index=True)
    logger.info(f"{len(final_timeline.index)} rows in total")
    final_timeline["Start Time"] = pd.to_datetime(final_timeline["Start Time"])
    final_timeline["End Time"] = pd.to_datetime(final_timeline["End Time"])
    final_timeline["Start Date Only"] = final_timeline["Start Date Only"].isin([True])
    final_timeline["End Date Only"] = final_timeline["End Date Only"].isin([True])
    final_timeline["Instrument"] = final_timeline["Instrument"].fillna("SDO")
    final_timeline["Comment"] = final_timeline["Comment"].fillna("No Comment")
    final_timeline = drop_duplicates(final_timeline.sort_values("Start Time", ignore_index=True))
    logger.info(f"{len(final_timeline.index)} rows in after deduplication")
    final_timeline["Start Time"] = format_times(final_timeline["Start Time"], final_timeline["Start Date Only"])
    final_timeline["End Time"] = format_times(final_timeline["End Time"], final_timeline["End Date Only"])
    final_timeline["End Time"] = final_timeline["End Time"].fillna("Unknown")
    final_timeline = final_timeline[["Start Time", "End Time", "Instrument", "Source", "Comment"]]
    final_timeline.to_csv("timeline.csv", index=False)
    final_timeline.to_csv("timeline.txt", sep="\t", index=False)
    final_timeline.to_json("timeline.json", orient="records")
    logger.info(f"Files were saved to {Path.cwd()}")
