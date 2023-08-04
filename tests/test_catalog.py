import pandas as pd

from xscen import catalog


def test_subset_file_coverage():
    df = pd.DataFrame.from_records(
        [
            {"id": "A", "date_start": "1950-01-01", "date_end": "1959-12-31"},
            {"id": "A", "date_start": "1960-01-01", "date_end": "1969-12-31"},
        ]
    )
    df["date_start"] = pd.to_datetime(df.date_start)
    df["date_end"] = pd.to_datetime(df.date_end)

    # Ok
    pd.testing.assert_frame_equal(
        catalog.subset_file_coverage(df, [1951, 1970], coverage=0.8), df
    )

    # Insufficient coverage (no files)
    pd.testing.assert_frame_equal(
        catalog.subset_file_coverage(df, [1975, 1986], coverage=0.8),
        pd.DataFrame(columns=df.columns),
    )

    # Insufficient coverage (partial files)
    pd.testing.assert_frame_equal(
        catalog.subset_file_coverage(df, [1951, 1976], coverage=0.8),
        pd.DataFrame(columns=df.columns),
    )
