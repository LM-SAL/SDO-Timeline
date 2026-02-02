# SDO Non-nominal Timeline

## Warning

This captures data from multiple sources and is not guaranteed to be accurate.
It is intended to be a rough guide to the non-nominal periods of SDO.

## Requirements

Requirements are in `requirements.txt`.

## Notes

Things to note:

1. If there is no end date, it fills that in with "Unknown".
2. Any dates without hours are assumed to start/end at midnight of that day.

This runs on GitHub Actions to create the files and tag a release for them.
It will also render the dataframe to html and push it to the `gh-pages` branch.

[The files are then served by GitHub Pages](https://lm-sal.github.io/SDO-Timeline/).
