"""
Simple scraper for SDO event timeline data.
"""

import contextlib
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

from config import DATASETS, MAP_4, TIME_FORMATS

logger.remove()
logger.add(sys.stderr, level="INFO")


def _format_date(date: str, year: str | None = None, start_date: datetime | None = None) -> pd.Timestamp:
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
        # Deal with only times with a hack
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
        new_date = date.split(" ")
        if "25" in new_date[1]:
            # This is a hack for 25 hour time
            new_date[1] = new_date[1].replace("25", "01")
            new_date[0] = new_date[0].split("/")[0] + "/" + str(int(new_date[0].split("/")[1]) + 1).zfill(1)
        new_date = pd.Timestamp(new_date[0] + f"/{year} " + new_date[1])
    # Multiple times - e.g., '8/28 20:35 8/14 20:50'
    # TODO: For now, just take the first entry
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
        # TODO: Improve this
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
                lambda x, time_format=time_format: datetime.strptime(x, time_format)  # NOQA: DTZ007
            )
            return data  # NOQA: TRY300
        except Exception as e:  # NOQA: BLE001
            logger.debug(f"Time format {time_format} did not work for {data.iloc[0, column]} for column {column}: {e}")
    msg = f"Could not find a suitable time format: {data.iloc[0, column]} or failed assignment to DataFrame."
    raise ValueError(
        msg,
    )


def _process_end_time(data: pd.DataFrame, column: int = 1) -> pd.DataFrame:
    # Add date to end time
    data[data.columns[column]] = pd.to_datetime(
        pd.to_datetime(data.iloc[:, 0]).dt.strftime("%m/%d/%Y") + " " + data.iloc[:, column],
    )
    # Increment date if end time is before start time
    timedelta = [
        pd.Timedelta(days=1) if pd.Timestamp(x) < pd.Timestamp(y) else pd.Timedelta(days=0)
        for x, y in zip(data.iloc[:, 0], data.iloc[:, 1], strict=False)
    ]
    data[data.columns[column]] += pd.to_timedelta(timedelta)
    return data


def _process_data(data: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """
    Certain files have no comments or have a comment in the third column.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to process.
    filepath : str
        Path to the file.

    Returns
    -------
    pd.DataFrame
        Processed dataframe.
    """
    if "AIA" in filepath:
        data["Instrument"] = "AIA"
    elif "HMI" in filepath:
        data["Instrument"] = "HMI"
    else:
        data["Instrument"] = "SDO"
    if "Start Date/Time" in data.columns:
        data = data.rename(columns={"Start Date/Time": "Start Time"})
    if "FSN" in data.columns:
        data = data.rename(columns={"FSN": "Comment"})
    if "Unnamed: 2" in data.columns:
        data = data.rename(columns={"Unnamed: 2": "Comment"})
    if data.columns[-1] == "Comment":
        data["Comment"].fillna(pd.read_fwf(filepath).columns[0])
    else:
        # Assumption that the comment is the first row which pandas turns into a column
        data["Comment"] = pd.read_fwf(filepath).columns[0]
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
        data["Start Time"] = [None] * len(data)
        data["End Time"] = [None] * len(data)
        for i, row in enumerate(data[0].str.split()):
            data.iloc[i, data.columns.get_loc("Start Time")] = row[0]
            data.iloc[i, data.columns.get_loc("End Time")] = row[1]
        data = data.drop(columns=[0])
        data = data.iloc[:, [1, 2, 0]]
        data.columns = ["Start Time", "End Time", "Comment"]
    elif "_2" in filepath or "_3" in filepath:
        data.columns = ["Start Time", "Comment"]
    elif "_4" in filepath:
        data = data.iloc[:, [1, 0]]
        data.columns = ["Start Time", "Comment"]
        data["Comment"] = data["Comment"].apply(lambda x: MAP_4[x])
    return data


def process_txt(filepath: str, skip_rows: list | None, data: pd.DataFrame) -> pd.DataFrame:
    """
    Process a text file.

    Parameters
    ----------
    filepath : str
        File path of the text file.
    skip_rows : list, None
        What rows to skip.
    data : pd.DataFrame
        Dataframe to append to.

    Returns
    -------
    pd.DataFrame
        Dataframe with the data from the text file.
    """
    if "http" in filepath:
        new_data = pd.read_fwf(
            filepath,
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
            new_data = _process_data(new_data, filepath)
        elif len(new_data.columns) > 3:  # NOQA: PLR2004
            logger.debug(f"Unexpected number of columns for {filepath}, dropping all but first two")
            new_data = new_data.iloc[:, [0, 1]]
            new_data.columns = ["Start Time", "End Time"]
            with contextlib.suppress(Exception):
                new_data = _process_time(new_data, 1)
            new_data = _process_data(new_data, filepath)
    else:
        new_data = pd.read_csv(filepath, header=None, sep="    ", skiprows=skip_rows, engine="python")
        new_data = _reformat_data(new_data, filepath)
        new_data = _process_time(new_data)
        new_data["Instrument"] = new_data["Comment"].apply(lambda x: "AIA" if "AIA" in x else None)
        new_data["Instrument"] = new_data["Comment"].apply(lambda x: "HMI" if "HMI" in x else None)
    new_data["Source"] = filepath.rsplit("/", maxsplit=1)[-1]
    data = pd.concat([data, new_data], ignore_index=True)
    data = new_data if data.empty else pd.concat([data, new_data], ignore_index=True)
    new_data["Source"] = filepath.rsplit("/", maxsplit=1)[-1]
    return pd.concat([data, new_data], ignore_index=True)


def process_html(url: str, data: pd.DataFrame) -> pd.DataFrame:  # NOQA: PLR0914
    """
    Process an html file.

    Parameters
    ----------
    url : str
        URL of the html file.
    data : pd.DataFrame
        Dataframe to append to.

    Returns
    -------
    pd.DataFrame
        Dataframe with the data from the html file.
    """
    request = requests.get(url, timeout=60)
    if request.status_code == 404:  # NOQA: PLR2004
        logger.warning(f"URL not found: {url}")
        return data
    soup = BeautifulSoup(request.text, "html.parser")
    table = soup.find_all("table")
    # There should be two html tables for this URL
    if len(table) == 1 and "jsocobs_info" in url:
        return data
    table = table[-1]
    rows = table.find_all("tr")
    # TODO: Regex to get the year
    year = None
    if "jsocobs_info" in url:
        year = url.split("info")[1].split(".")[0]
    # These HTML tables are by column and not by row
    if "hmi/cov2/" in url:
        new_rows = rows[0].text.split("\n\n")
        # Time is one single element whereas each event text is a separate element
        dates, text = new_rows[0].strip().split("\n"), new_rows[1:-1]
        if dates in [[""], [" "], []]:
            logger.warning(f"No data found for {url}")
            return data
        instrument = ["HMI" if "HMI" in new_row else "AIA" if "AIA" in new_row else "SDO" for new_row in text]
        comment = [new_row.replace("\n", " ") for new_row in text]
        new_dates = dates.copy()
        for date in dates:
            # Hack workaround for http://jsoc.stanford.edu/doc/data/hmi/cov2/cov202503.html
            # where the date just "multiple"
            if "multiple" in date:
                new_dates[new_dates.index(date)] = date.replace("multiple", new_dates[0])
        start_dates = [(_format_date(_clean_date(date), year)) for date in new_dates]
        end_dates = [None] * len(new_dates)
        new_data = pd.DataFrame(
            {"Start Time": start_dates, "End Time": end_dates, "Instrument": instrument, "Comment": comment},
        )
        new_data["Source"] = url.rsplit("/", maxsplit=1)[-1]
        data = pd.concat([data, new_data])
    else:
        for row in rows[1:]:
            text = row.text.strip().split("\n")
            # First column is the start time
            #   Can have multiple times
            # Second column is the end time
            #   Can be be blank
            # Third column is the event
            # Fifth column is the AIA Description
            # Eighth column is the HMI Description
            comment = text[2].strip() or text[4].strip() or text[7].strip()
            instrument = "SDO" if text[2].strip() else "AIA" if text[4].strip() else "HMI"
            extra_replace = False
            if "jsocobs_info" in url:
                extra_replace = True
            start_date = _clean_date(text[0], extra_replace=extra_replace)
            end_date = _clean_date(text[1], extra_replace=extra_replace) if len(text[1]) > 1 else "NaT"
            start_date = _format_date(start_date, year)
            end_date = _format_date(end_date, year, start_date)
            new_data = pd.Series(
                {"Start Time": start_date, "End Time": end_date, "Instrument": instrument, "Comment": comment},
            )
            new_data["Source"] = url.rsplit("/", maxsplit=1)[-1]
            data = pd.concat([data, pd.DataFrame([new_data], columns=new_data.index)]).reset_index(drop=True)
    return data


def scrape_url(url: str) -> list:
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
    base_url = str(Path(url).parent).replace("https:/", "https://")
    request = requests.get(url, timeout=60)
    soup = BeautifulSoup(request.text, "html.parser")
    urls = []
    for link in soup.find_all("a"):
        a_url = link.get("href")
        if a_url and "txt" in a_url:
            urls.append(base_url + "/" + a_url)
    return urls


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicates rows in a dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to deduplicate.

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe.
    """
    first_row = {
        "Start Time": data["Start Time"][0],
        "End Time": data["End Time"][0],
        "Instrument": data["Instrument"][0],
        "Source": data["Source"][0],
        "Comment": data["Comment"][0],
    }
    updated_timeline = pd.DataFrame([first_row])
    for idx, row in data.iterrows():
        if idx == 0:
            continue
        # We want to combine events that <=5 minutes apart
        if row["Start Time"] - updated_timeline.iloc[-1]["Start Time"] <= pd.Timedelta("5 minute"):
            updated_timeline.loc[updated_timeline["Start Time"] == row["Start Time"], "End Time"] = row["End Time"]
            # Need to update the instrument and comment if they are different
            if updated_timeline.iloc[-1]["Instrument"] != row["Instrument"]:
                updated_timeline.loc[updated_timeline["Start Time"] == row["Start Time"], "Instrument"] = "SDO"
            if row["Comment"] not in updated_timeline.iloc[-1]["Comment"]:
                updated_timeline.loc[updated_timeline["Start Time"] == row["Start Time"], "Comment"] = (
                    updated_timeline.iloc[-1]["Comment"] + " and " + row["Comment"]
                )
            if row["Source"] not in updated_timeline.iloc[-1]["Source"]:
                updated_timeline.loc[updated_timeline["Start Time"] == row["Start Time"], "Source"] = (
                    updated_timeline.iloc[-1]["Source"] + " and " + row["Source"]
                )
            continue
        insert_row = {
            "Start Time": row["Start Time"],
            "End Time": row["End Time"],
            "Instrument": row["Instrument"],
            "Source": row["Source"],
            "Comment": row["Comment"],
        }
        updated_timeline = pd.concat([updated_timeline, pd.DataFrame([insert_row])])
    return updated_timeline


if __name__ == "__main__":
    final_timeline = pd.DataFrame(columns=["Start Time", "End Time", "Instrument", "Source", "Comment"])
    for dataset_name, block in DATASETS.items():
        logger.info(f"Scraping {dataset_name}")
        logger.info(f"{len(final_timeline.index)} rows so far")
        urls = [block.get("URL")]
        if block.get("SCRAPE"):
            urls = scrape_url(block["URL"])
        if block.get("RANGE"):
            if block.get("MONTH_RANGE"):
                urls = [
                    block["fURL"].format(f"20{i:02}{j:02}") for i, j in product(block["RANGE"], block["MONTH_RANGE"])
                ]
            else:
                urls = [block["fURL"].format(f"20{i:02}") for i in block["RANGE"]]
        for url in sorted(urls):
            logger.info(f"Parsing {url}")
            if "txt" in url:
                final_timeline = process_txt(url, block.get("SKIP_ROWS"), final_timeline)
            elif "html" in url:
                final_timeline = process_html(url, final_timeline)
            else:
                msg = f"Unknown file type for {url}"
                raise ValueError(msg)

    logger.info(f"{len(final_timeline.index)} rows in total")
    final_timeline = final_timeline.sort_values("Start Time")
    final_timeline = final_timeline.reset_index(drop=True)
    final_timeline["End Time"] = final_timeline["End Time"].fillna("Unknown")
    final_timeline["Instrument"] = final_timeline["Instrument"].fillna("SDO")
    final_timeline["Comment"] = final_timeline["Comment"].fillna("No Comment")
    final_timeline = drop_duplicates(final_timeline)
    logger.info(f"{len(final_timeline.index)} rows in after deduplication")
    today_date = pd.Timestamp("today").strftime("%Y%m%d")
    final_timeline.to_csv(f"timeline_{today_date}.csv", index=False)
    final_timeline.to_csv(f"timeline_{today_date}.txt", sep="\t", index=False)
    logger.info(f"Files were saved to {Path.cwd()}")
