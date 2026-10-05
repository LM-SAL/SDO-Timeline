# SDO Non-nominal Timeline

## Warning

This captures data from multiple sources and is not guaranteed to be accurate.
It is intended to be a rough guide to the non-nominal periods of SDO.

## Sources

- [JSOC observation logs](https://aia.lmsal.com/public/jsocobs_info.html) (one page per year)
- [SDO eclipse (spacecraft night) times](https://aia.lmsal.com/public/sdo_spacecraft_night.txt), including lunar transits
- [SDO spacecraft operations](https://aia.lmsal.com/public/sdo_spacecraft_events.txt)
- [AIA and HMI calibrations](https://aia.lmsal.com/public/jsocinst_calibrations.html)
- [HMI data coverage events](http://jsoc.stanford.edu/doc/data/hmi/cov2/) (one page per month)
- The SDO, AIA, HMI and GS Maintenance calendars from the [JSOC calendar](https://aia.lmsal.com/public/SDOcalendar.html),
  which includes events scheduled for the next 180 days ("TBD" placeholder events are skipped)
- `data_*.txt`, static lists of older events

## Requirements

Requirements are in `requirements.txt`, and the tools for the checks are in `requirements-dev.txt`.

Run `tox -e py314` to run the checks and create the files, and `tox -e codestyle` for the style, type and file checks.

## Notes

Things to note:

1. If there is no end date, it fills that in with "Unknown".
2. If the source has no time of day, the date is given without one.
3. Events of the same instrument that start within 5 minutes of each other are combined into one row.
4. Events labelled SDO whose comment only names AIA or only names HMI are given that instrument.

This runs daily on GitHub Actions to create `timeline.csv`, `timeline.json` and `timeline.txt` and update the single
`nightly` release with them, so the latest files are always at
<https://github.com/LM-SAL/SDO-Timeline/releases/latest/download/timeline.csv>.

[The files are also served by GitHub Pages](https://lm-sal.github.io/SDO-Timeline/), with a searchable table
([Tabulator](https://tabulator.info/)) and a timeline ([vis-timeline](https://visjs.github.io/vis-timeline/)) of the events.
