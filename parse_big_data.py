"""Parse the raw performance dumps in data/big_data/ into per-base-station CSVs.

Source files (Performance_{4G,5G}_{Weekday,Weekend}.txt) contain one
representative day per cell at 30-minute resolution:

    Base Station ID, Cell ID, Timestamp, PRB Usage Ratio (%),
    Traffic Volume (KByte), Number of Users, ... (energy columns, unused)

Output: one CSV per base station in the same shape the training pipeline
reads (time column + two auxiliary features + `users` target):

    time_hour, prb, bytes, users

Note: the source has no packet counts, so instead of the `packets` column of
the existing bs_<id>.csv files the parsed files carry `prb` (mean PRB usage
ratio, %). The data pipeline does not care about feature column names, only
that `users` is present and enc_in matches the feature count (3).

Aggregation rules:
  - cells -> base station (per 30-min slot): users and traffic are summed,
    PRB usage is averaged
  - 30-min -> hourly: traffic is summed (volume), users and PRB are averaged
    (concurrent levels)
  - bytes = Traffic Volume (KByte) * 1024
  - time_hour gets a synthetic date matching the file type (the series only
    carries a time of day): Monday 2024-01-01 for Weekday files, Saturday
    2024-01-06 for Weekend files, so weekday-dependent time features stay
    consistent.

Besides the four per-day folders, a concatenated training variant is built in
parsed/concat/{4G,5G}/bs_<id>.csv: the weekday day re-dated to Friday
2024-01-05 followed by the weekend day (Saturday 2024-01-06), giving one
continuous 48-hour series per station. This is what `run.py -f big_data`
trains on.

Usage:
    python parse_big_data.py                 # parse all four files + concat
    from parse_big_data import parse_performance_file   # single file
"""

import os
import glob

import pandas as pd

SOURCE_COLUMNS = {
    'Base Station ID': 'bs',
    'Timestamp': 'timestamp',
    'PRB Usage Ratio (%)': 'prb',
    'Traffic Volume (KByte)': 'kbyte',
    'Number of Users': 'users',
}

WEEKDAY_DATE = '2024-01-01'   # a Monday
WEEKEND_DATE = '2024-01-06'   # a Saturday

# for the concatenated 48h series: Friday followed by Saturday is continuous
CONCAT_WEEKDAY_DATE = '2024-01-05'
CONCAT_WEEKEND_DATE = '2024-01-06'


def parse_performance_file(input_path, output_dir, date=None):
    """Parse one Performance_*.txt file into per-base-station hourly CSVs.

    Args:
        input_path: path to a Performance_{4G,5G}_{Weekday,Weekend}.txt file.
        output_dir: directory that receives one bs_<id>.csv per base station.
        date: 'YYYY-MM-DD' used to build time_hour. Defaults by filename:
              Weekend files get WEEKEND_DATE, everything else WEEKDAY_DATE.

    Returns:
        Number of base-station files written.
    """
    if date is None:
        name = os.path.basename(input_path).lower()
        date = WEEKEND_DATE if 'weekend' in name else WEEKDAY_DATE

    df = pd.read_csv(input_path, usecols=list(SOURCE_COLUMNS))
    df = df.rename(columns=SOURCE_COLUMNS)

    # cells -> base station, per 30-minute slot
    per_slot = df.groupby(['bs', 'timestamp'], sort=True).agg(
        prb=('prb', 'mean'),
        kbyte=('kbyte', 'sum'),
        users=('users', 'sum'),
    ).reset_index()

    # 30-minute slots -> hours
    per_slot['hour'] = per_slot['timestamp'].str.slice(0, 2).astype(int)
    hourly = per_slot.groupby(['bs', 'hour'], sort=True).agg(
        prb=('prb', 'mean'),
        kbyte=('kbyte', 'sum'),
        users=('users', 'mean'),
    ).reset_index()

    hourly['time_hour'] = pd.Timestamp(date) + pd.to_timedelta(hourly['hour'], unit='h')
    hourly['bytes'] = (hourly['kbyte'] * 1024).round(2)
    hourly['prb'] = hourly['prb'].round(2)
    hourly['users'] = hourly['users'].round(2)

    os.makedirs(output_dir, exist_ok=True)
    written = 0
    for bs, group in hourly.groupby('bs'):
        out_path = os.path.join(output_dir, f'bs_{bs}.csv')
        group[['time_hour', 'prb', 'bytes', 'users']].to_csv(out_path, index=False)
        written += 1

    print(f'{os.path.basename(input_path)}: wrote {written} base-station files '
          f'({len(hourly)} hourly rows) to {output_dir}')
    return written


def _redate(df, date):
    """Replace the date part of time_hour, keeping the hour of day."""
    hours = pd.to_datetime(df['time_hour']).dt.hour
    df = df.copy()
    df['time_hour'] = pd.Timestamp(date) + pd.to_timedelta(hours, unit='h')
    return df


def concat_weekday_weekend(parsed_base='data/big_data/parsed'):
    """Build the 48-hour training series: weekday day + weekend day.

    For every station present in both <tech>_weekday and <tech>_weekend, the
    weekday day is re-dated to Friday and the weekend day to the following
    Saturday, so the concatenation is one continuous hourly series. Output:
    <parsed_base>/concat/<tech>/bs_<id>.csv (48 rows each).
    """
    total = 0
    for tech in ['4G', '5G']:
        weekday_dir = os.path.join(parsed_base, f'{tech}_weekday')
        weekend_dir = os.path.join(parsed_base, f'{tech}_weekend')
        out_dir = os.path.join(parsed_base, 'concat', tech)
        os.makedirs(out_dir, exist_ok=True)

        weekday_files = {os.path.basename(p) for p in glob.glob(os.path.join(weekday_dir, 'bs_*.csv'))}
        weekend_files = {os.path.basename(p) for p in glob.glob(os.path.join(weekend_dir, 'bs_*.csv'))}
        both = sorted(weekday_files & weekend_files)
        missing = (weekday_files | weekend_files) - set(both)
        if missing:
            print(f'{tech}: skipping {len(missing)} stations missing one of the two days')

        for name in both:
            weekday = _redate(pd.read_csv(os.path.join(weekday_dir, name)), CONCAT_WEEKDAY_DATE)
            weekend = _redate(pd.read_csv(os.path.join(weekend_dir, name)), CONCAT_WEEKEND_DATE)
            combined = pd.concat([weekday, weekend], ignore_index=True)
            combined.to_csv(os.path.join(out_dir, name), index=False)
            total += 1

        print(f'{tech}: wrote {len(both)} concatenated 48h files to {out_dir}')

    print(f'Concat done: {total} files under {os.path.join(parsed_base, "concat")}')
    return total


def parse_all(base_dir='data/big_data', output_base=None):
    """Parse every Performance_*.txt in base_dir, then build the concatenated
    weekday+weekend training series.

    Per-day output goes to <output_base>/<technology>_<daytype>/bs_<id>.csv,
    e.g. data/big_data/parsed/4G_weekday/bs_10055.csv.
    """
    if output_base is None:
        output_base = os.path.join(base_dir, 'parsed')

    sources = sorted(glob.glob(os.path.join(base_dir, 'Performance_*.txt')))
    if not sources:
        raise FileNotFoundError(f'No Performance_*.txt files found in {base_dir}')

    total = 0
    for src in sources:
        stem = os.path.splitext(os.path.basename(src))[0]     # Performance_4G_Weekday
        parts = stem.split('_')                               # [Performance, 4G, Weekday]
        subfolder = f'{parts[1]}_{parts[2].lower()}'          # 4G_weekday
        total += parse_performance_file(src, os.path.join(output_base, subfolder))

    print(f'Done: {total} base-station files under {output_base}')

    concat_weekday_weekend(output_base)
    return total


if __name__ == '__main__':
    parse_all()
