"""Validate and serialize a BED-like dataframe against the BED specification.

The validators in this module are designed to make a dataframe "ready-to-write"
to a compliant BED file.

The BED specification is defined here: https://github.com/samtools/hts-specs/blob/master/BEDv1.pdf

Some facts about BED files
--------------------------
* Intervals in BED are 0-based, half-open.
* The first 3 fields are mandatory, last 9 are optional
* Fields use 7-bit printable ASCII, including spaces but excluding tabs,
  newlines, and other control characters.
* Schemas: ``BED{3,4,5,6,7,8,9,12}+m``: ``m`` corresponds to custom fields, you
  can also do ``BEDn+`` for an unspecified number of custom fields.
* BED10 and BED11 are illegal.
* Order is "binding": if an optional field is filled then all previous ones
  must also be filled.
* Standard BED fields can never be empty - one must use a special null or
  "uninformative" placeholder value.
* Custom BED fields can be empty when tab is used as delimiter.

Information supplied out-of-band
--------------------------------
* Assembly/chromsizes: a dictionary or pandas Series mapping chromosome names
  to lengths to validate interval bounds [optional].

* Custom fields in positions 4-12: which of the first 4 to 12 fields are
  provided as standard BED fields and which are custom fields. This can be
  specified using a ``BEDn+m`` schema specifier described above.

Information we are agnostic to
-------------------------------
* Delimiters: Fortunately, delimiters do not concern the validation here.
  We work with data that is already in dataframe form, e.g. already parsed from
  a file.

* Comments and blank lines: We assume that the dataframe contains only data
  lines.

* Names, dtypes, and values of custom fields.

.. warning::
    Note that the BED spec allows for the use of either spaces or tabs as
    delimiters, even permitting a mixture in the same file!

    We will not validate any of the whitespace constraints imposed on fields in
    the spec to deal with the possibility of delimiters being improperly
    parsed upon reading. We recommend for sanity's sake that any BED-like
    dataframe will be written to a file using a **single tab** as the sole
    delimiter.

Relaxations
-----------
The spec is overly strict in some ways. Here are some conditions we either
don't check or make optional:

* We don't enforce limiting the ``name`` field to 7-bit printable ascii.

* Non-alphanumeric characters in the ``chrom`` field: the spec requires that
  the chrom field contain only alphanumeric characters and underscores. Many
  sequences contain non-alphanumeric characters, such as NCBI identifiers
  like ``gb|accession|locus``. TODO: make this optional.

* Non-integer score field: the spec requires that the score be an integer
  between 0-1000. We provide some lenience by allowing arbitrary integers or
  floats by setting `strict_score=False`. Many tools use floating point scores
  in practice.

.. warning::
    Many BED files in the wild will use "." as an uninformative score value
    for all features. However, the spec defines the uninformative fill value
    as 0.
"""
from typing import Optional, Union
import pathlib
import re
import warnings

import pandas as pd
import numpy as np


UINT64_MAX = np.iinfo(np.uint64).max

# Custom BED fields should contain either one of these data types or a
# comma-separated list of Integer, Unsigned, or Float.
BED_DTYPE_MAP = {
    "Integer": np.int64,
    "Unsigned": np.uint64,
    "Float": np.float64,
    "Character": object,
    "String": object,
}

BED_FIELD_NAMES = [
    "chrom",
    "start",
    "end",
    "name",
    "score",
    "strand",
    "thickStart",
    "thickEnd",
    "itemRgb",
    "blockCount",
    "blockSizes",
    "blockStarts",
]

BED_FIELD_KINDS = {
    "chrom": "OU",
    "start": "iu",
    "end": "iu",
    "name": "OU",
    "score": "iuf",
    "strand": "OU",
    "thickStart": "iu",
    "thickEnd": "iu",
    "itemRgb": "OU",
    "blockCount": "iu",
    "blockSizes": "OU",
    "blockStarts": "OU",
}

BED_FIELD_FILLVALUES = {
    "chrom": "_",
    "start": 0,
    "end": 0,
    "name": ".",
    "score": 0,
    "strand": ".",
    "itemRgb": "0",
}

BED_FIELD_VALIDATORS = {}


def validator(col: str) -> callable:
    def decorator(func: callable) -> callable:
        BED_FIELD_VALIDATORS[col] = func
        return func

    return decorator


@validator("chrom")
def check_chrom(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the chromosome names of a BED dataframe.

    The chrom column is limited to non-whitespace word characters only
    (alphanumeric characters and underscores). Each name must be between 1 and
    255 characters in length, inclusive.
    """
    # Check that the chrom column contains only alphanumeric characters
    is_alnum = df["chrom"].str.match(r"^[A-Za-z0-9_]+$").all()

    # Check that the name column is no longer than 255 characters
    lengths = df["chrom"].str.len()
    is_len_ok = ((lengths >= 1) & (lengths <= 255)).all()

    return {
        "chrom.is_alnum": is_alnum,
        "chrom.is_len_ok": is_len_ok,
    }


@validator("start")
def check_start(
    df: pd.DataFrame, chromsizes: Optional[dict | pd.Series] = None
) -> dict[bool]:
    """
    Validate the start coordinates of a BED dataframe.

    Start must be an integer greater than or equal to 0 and less than or equal
    to the total number of bases of the chromosome to which it belongs.

    If the size of the chromosome is unknown, then start must be less than or
    equal to 2**64 - 1, which is the maximum size of an unsigned 64-bit integer.
    """
    # Check that the start column contains only non-negative integers
    is_nonneg = (df["start"] >= 0).all()

    # Check that the start column contains only integers less than 2**64 - 1
    is_le_64 = (df["start"] <= UINT64_MAX).all()

    out = {
        "start.is_nonneg": is_nonneg,
        "start.is_le_64": is_le_64,
    }

    # Check that the start column contains only integers < the chromosome size
    if chromsizes is not None:
        chromsizes = pd.Series(chromsizes)
        is_lt_chrom = (df["end"] < chromsizes[df["chrom"]]).all()
        out["start.is_lt_chrom"] = is_lt_chrom

    return out


@validator("end")
def check_end(
    df: pd.DataFrame, chromsizes: Optional[dict | pd.Series] = None
) -> dict[bool]:
    """
    Validate the end coordinates of a BED dataframe.

    End must be an integer greater than or equal to the value of start and
    less than or equal to the total number of bases in the chromosome to
    which it belongs.

    If the size of the chromosome is unknown, then end must be less than or
    equal to 2**64 - 1, the maximum size of an unsigned 64-bit integer.
    """
    # Check that the end column contains only non-negative integers
    is_nonneg = (df["end"] >= 0).all()

    # Check that the end column contains only integers less than 2**64 - 1
    is_le_64 = (df["end"] <= UINT64_MAX).all()

    is_end_ge_start = (df["end"] >= df["start"]).all()

    out = {
        "end.is_nonneg": is_nonneg,
        "end.is_le_64": is_le_64,
        "end.is_end_ge_start": is_end_ge_start,
    }

    # Check that the end column contains only integers <= the chromosome size
    if chromsizes is not None:
        chromsizes = pd.Series(chromsizes)
        is_le_chrom = (df["end"] <= chromsizes[df["chrom"]]).all()
        out["end.is_le_chrom"] = is_le_chrom

    return out


@validator("name")
def check_name(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the name column of a BED dataframe.

    Name must be 1 to 255 non-tab characters. Multiple data lines may share
    the same name. If all features have uninformative names, dot (.) may be
    used as a name on every data line.
    """
    # Check that the name column is no longer than 255 characters
    lengths = df["name"].str.len()
    is_len_ok = ((lengths >= 1) & (lengths <= 255)).all()

    return {
        "name.is_len_ok": is_len_ok,
    }


@validator("score")
def check_score(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the score column of a BED dataframe.

    Integer between 0 and 1000, inclusive. When all features have uninformative
    scores, 0 should be used as the score on every data line.

    Note: Using "." is illegal in the spec, but is used in practice. 0 is the
    the uninformative score used in the spec.
    """
    # Check that the score column contains only integers between 0 and 1000, inclusive
    is_in_range = ((df["score"] >= 0) & (df["score"] <= 1000)).all()

    return {
        "score.is_in_range": is_in_range,
    }


@validator("strand")
def check_strand(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the strand column of a BED dataframe.

    Strand must be one of +, -, . (no strand), or ? (unknown strand).
    When parsing files that are not BED6+, strand should be treated as ".".
    """
    # Check that the strand column contains only valid strand characters
    is_pattern_ok = df["strand"].str.match(r"^[+\-.?]$").all()

    return {
        "strand.is_pattern_ok": is_pattern_ok,
    }


@validator("thickStart")
def check_thickStart(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the thickStart column of a BED dataframe.

    Must be an integer between start and end, inclusive. When all features
    have uninformative thickStarts, the value of start should be used.
    """
    # Check that the thickStart column contains only integers between start and end, inclusive
    is_ge_start = (df["thickStart"] >= df["start"]).all()
    is_le_end = (df["thickStart"] <= df["end"]).all()

    return {
        "thickStart.is_ge_start": is_ge_start,
        "thickStart.is_le_end": is_le_end,
    }


@validator("thickEnd")
def check_thickEnd(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the thickEnd column of a BED dataframe.

    Must be an integer greater than or equal to start and less than or equal
    to end, inclusive. When all features have uninformative thickEnds, the
    value of end should be used.
    """
    # Check that the thickEnd column contains only integers between start and end, inclusive
    is_ge_start = (df["thickEnd"] >= df["start"]).all()
    is_le_end = (df["thickEnd"] <= df["end"]).all()

    return {
        "thickEnd.is_ge_start": is_ge_start,
        "thickEnd.is_le_end": is_le_end,
    }


@validator("itemRgb")
def check_itemRgb(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the itemRgb column of a BED dataframe.

    A triple of 3 integers separated by commas. Each integer is between 0 and
    255, inclusive. To make a feature black, itemRgb may be a single 0, as a
    shorthand for 0,0,0. When all features have uninformative itemRgb values,
    0 should be used.
    """
    # Check that the itemRgb is a triple of integers separated by commas
    # or a single 0
    is_pattern_ok = (
        df["itemRgb"].str.match(r"^(\d{1,3},){2}\d{1,3}$") | (df["itemRgb"] == "0")
    ).all()

    # Check that the itemRgb column contains only integers between 0 and 255, inclusive
    is_in_range = (
        df["itemRgb"]
        .str.split(",")
        .apply(lambda x: all([int(i) >= 0 and int(i) <= 255 for i in x]))
    ).all()

    return {
        "itemRgb.is_pattern_ok": is_pattern_ok,
        "itemRgb.is_in_range": is_in_range,
    }


@validator("blockCount")
def check_blockCount(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the blockCount column of a BED dataframe.

    Must be an integer greater than 0.

    Note: mandatory in BED12+ files.
    """
    # Check that the blockCount column contains only integers greater than 0
    is_gt_0 = (df["blockCount"] > 0).all()

    return {
        "blockCount.is_gt_0": is_gt_0,
    }


@validator("blockSizes")
def check_blockSizes(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the blockSizes column of a BED dataframe.

    Comma-separated list of length blockCount containing the size of each
    block. There must be no spaces before or after commas.

    There may be a trailing comma after the last element of the list.

    Note: mandatory in BED12+ files.
    """
    # Check that the blockSizes column contains only comma-separated lists of integers
    is_pattern_ok = df["blockSizes"].str.match(r"^(\d+,)*\d+(,)?$").all()

    # Check that the number of block sizes matches the blockCount
    n_blocks = df["blockSizes"].str.rstrip(",").str.count(",") + 1
    is_n_blocks_ok = (n_blocks == df["blockCount"]).all()

    return {
        "blockSizes.is_pattern_ok": is_pattern_ok,
        "blockSizes.is_n_blocks_ok": is_n_blocks_ok,
    }


@validator("blockStarts")
def check_blockStarts(df: pd.DataFrame) -> dict[bool]:
    """
    Validate the blockStarts column of a BED dataframe.

    Comma-separated list of length blockCount containing each block's start
    position, relative to start. There must not be spaces before or after the
    commas. There may be a trailing comma after the last element of the list.
    Each element in blockStarts is paired with the corresponding element in
    blockSizes.

    Each blockStarts element must be an integer between 0 and end - start,
    inclusive.

    Each block must be contained within the feature. That means that for each
    couple i of (blockStart, blockSize), the quantity start +
    blockStart + blockSize must be less or equal to end.

    The first block must start at start and the last block must end at end.

    The blockStarts must be sorted in ascending order.

    The blocks must not overlap.

    Note: mandatory in BED12+ files.
    """
    # Check that the blockStarts column contains only comma-separated lists of integers
    is_pattern_ok = df["blockStarts"].str.match(r"^(\d+,)*\d+(,)?$").all()

    block_starts = (
        df["blockStarts"]
        .str.rstrip(",")
        .str.split(",")
        .apply(lambda x: [int(i) for i in x])
    )
    block_sizes = (
        df["blockSizes"]
        .str.rstrip(",")
        .str.split(",")
        .apply(lambda x: [int(i) for i in x])
    )
    bs_start_end = pd.concat(
        [block_starts, block_sizes, df["start"], df["end"]], axis=1
    )

    # Check that the number of block starts matches the blockCount
    is_n_blocks_ok = (block_starts.apply(len) == df["blockCount"]).all()

    # Check that the blockStarts are in range
    is_in_range = bs_start_end.apply(
        lambda x: all(
            [
                x["blockStarts"][i] >= 0 and x["blockStarts"][i] <= x["end"]
                for i in range(len(x["blockStarts"]))
            ]
        ),
        axis=1,
    ).all()

    # Check that the first block begins at start
    is_first_block_start = bs_start_end.apply(
        (lambda x: x["blockStarts"][0] == 0), axis=1
    ).all()

    # Check that the last block stops at end
    is_last_block_end = bs_start_end.apply(
        (lambda x: x["blockStarts"][-1] + x["blockSizes"][-1] == x["end"] - x["start"]),
        axis=1,
    ).all()

    # Check that the blockStarts are in ascending order
    is_sorted = block_starts.apply(lambda x: x == sorted(x)).all()

    # Check that the blocks do not overlap
    is_no_overlap = True
    for row_block_starts, row_block_sizes in zip(
        block_starts.values, block_sizes.values
    ):
        for i in range(len(row_block_starts) - 1):
            if row_block_starts[i] + row_block_sizes[i] > row_block_starts[i + 1]:
                is_no_overlap = False
                break

    return {
        "blockStarts.is_pattern_ok": is_pattern_ok,
        "blockStarts.is_n_blocks_ok": is_n_blocks_ok,
        "blockStarts.is_in_range": is_in_range,
        "blockStarts.is_first_block_start": is_first_block_start,
        "blockStarts.is_last_block_end": is_last_block_end,
        "blockStarts.is_sorted": is_sorted,
        "blockStarts.is_no_overlap": is_no_overlap,
    }


def validate_bed_fields(
    df: pd.DataFrame,
    fields: list[str],
    chromsizes: Optional[dict | pd.Series] = None,
    strict_score: bool = False,
) -> tuple[set[str], set[str], set[str]]:
    """
    Validate the fields of a BED dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Bedframe to validate.
    fields : list[str]
        List of fields to validate.
    chromsizes : dict or pd.Series, optional
        Assembly/chromsizes to validate against.
    strict_score : bool, optional
        Whether to enforce the score field to be integer in the range 0-1000.

    Returns
    -------
    set[str], set[str], set[str]
        (1) set of names of fields having an invalid dtype
        (2) set of names of fields containing at least one null value
        (3) set of properties that failed validation.
    """
    # Find all fields with an invalid dtype
    dtype_failed = set()
    for col in fields:
        kind = df[col].dtype.kind
        if strict_score and col == "score":
            allowed_kinds = "iu"
        else:
            allowed_kinds = BED_FIELD_KINDS[col]
        if kind not in allowed_kinds:
            dtype_failed.add(col)

    # Find all fields containing at least one null value
    notnull = {}
    for col in fields:
        if col not in dtype_failed:
            if col == "score" and not strict_score:
                continue
            notnull[col] = df[col].notnull().all()
    notnull = pd.Series(notnull)
    notnull_failed = set(notnull.loc[~notnull].index)

    # Find all field properties that failed validation
    props = {}
    for col in fields:
        if col not in dtype_failed:
            if col == "score" and not strict_score:
                continue
            if col in ("start", "end"):
                props.update(BED_FIELD_VALIDATORS[col](df, chromsizes))
            else:
                props.update(BED_FIELD_VALIDATORS[col](df))
    props = pd.Series(props)
    prop_failed = set(props.loc[~props].index)

    return dtype_failed, notnull_failed, prop_failed


def check_is_sorted(df: pd.DataFrame) -> dict[bool]:
    """
    Validate that a BED dataframe is sorted.

    BED dataframes should have all records belonging to the same ``chrom``
    appear consecutively, sorted by start and then by end.

    Parameters
    ----------
    df : pd.DataFrame
        Bedframe to validate.

    Returns
    -------
    dict[bool]
        Sorting criteria checks, True for pass and False for fail.

    Notes
    -----
    The scheme for ordering the chromosomes doesn't matter, as long as all
    rows with the same chrom value occur consecutively.
    """
    # Check that all rows with the same chrom value are grouped together
    run_starts = np.r_[
        0, np.flatnonzero(df["chrom"].values[1:] != df["chrom"].values[:-1]) + 1
    ]
    run_values = df["chrom"].to_numpy()[run_starts]
    is_chrom_consecutive = len(run_values) == len(np.unique(run_values))

    # Check that that within chromosomes the rows are sorted by start, then by end
    is_sorted_start_end = True
    for _, group in df.groupby("chrom", sort=False):
        starts = group["start"].to_numpy()
        ends = group["end"].to_numpy()
        indices = np.lexsort((ends, starts))
        if not (
            np.array_equal(starts[indices], starts)
            and np.array_equal(ends[indices], ends)
        ):
            is_sorted_start_end = False
            break

    return {
        "sorted.is_chrom_consecutive": is_chrom_consecutive,
        "sorted.is_sorted_start_end": is_sorted_start_end,
    }


def _infer_bed_schema(df: pd.DataFrame) -> tuple[int, bool]:
    for i in [12, 9, 8, 7, 6, 5, 4, 3]:
        if BED_FIELD_NAMES[i - 1] in df.columns:
            n = i
            break
    else:
        raise ValueError("Could not infer a BED schema.")
    extended = len(df.columns) > n
    return n, extended


def _parse_bed_schema_name(schema: str) -> tuple[int, bool]:
    pattern = r"^bed(3|4|5|6|7|8|9|12)?(\+(\d+)?)?$"
    match = re.match(pattern, schema.lower())
    if not match:
        raise ValueError(f"Invalid BED schema name: {schema}")
    n = int(match.group(1)) if match.group(1) else 6
    extended = match.group(2) is not None
    return n, extended


def to_bed(
    df: pd.DataFrame,
    path: Optional[Union[str, pathlib.Path]] = None,
    *,
    schema: str = "infer",
    validate_fields: bool = True,
    require_sorted: bool = False,
    chromsizes: Optional[Union[dict, pd.Series]] = None,
    strict_score: bool = False,
    replace_na: bool = True,
    na_rep: str = "nan",
) -> Union[str, None]:
    """Write a DataFrame to a compliant BED file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write.
    path : str or Path, optional
        Path to write the BED file to. If not provided, the dataframe will be
        serialized to a string and returned.
    schema : str, optional [default: "infer"]
        BED schema name, e.g. "bed9" or "bed4+". If "infer", the schema will
        be inferred from the name in df's columns that matches the highest
        standard BED field name.
    validate_fields : bool, optional [default: True]
        If True, validate the standard BED fields for the required schema.
    require_sorted : bool, optional [default: False]
        If True, require that the DataFrame is sorted by chrom, start, end.
    chromsizes : dict or Series, optional
        Chromosome sizes to use for validating interval bounds. If not provided,
        this validation will be skipped.
    strict_score : bool, optional [default: False]
        If True, raise an error if any score values are not integers in the
        range [0, 1000].
    replace_na : bool, optional [default: True]
        If True, replace NA or NaN values in strict BED fields before writing
        to prevent validation failure.
    na_rep : str, optional [default: "nan"]
        String to use to print NA or NaN values in non-standard or relaxed
        fields, e.g. floating point ``score`` field.

    Returns
    -------
    If `path` is `None`, returns the serialized BED file as a string.
    Otherwise writes the file to `path` and returns `None`.

    Notes
    -----
    BED schema names are case-insensitive. "bed" is a synonym for "bed6".
    By default, this function will only write the standard BED fields and fill
    in any missing fields with uninformative values.

    To write custom fields, use the "bedN+" syntax, where N is the number
    of standard fields: all additional columns in the input dataframe will be
    appended to the output, following the standard fields, in the order they
    appear in the input.
    """
    if schema == "infer":
        n, extended = _infer_bed_schema(df)
    else:
        n, extended = _parse_bed_schema_name(schema)

    if (
        "chrom" not in df.columns
        or "start" not in df.columns
        or "end" not in df.columns
    ):
        raise ValueError(
            "BED dataframe must have at least 3 fields: chrom, start, end."
        )

    if n == 12 and (
        "blockCount" not in df.columns
        or "blockSizes" not in df.columns
        or "blockStarts" not in df.columns
    ):
        raise ValueError(
            "Informative blockCount, blockSizes, and blockStarts fields are "
            "mandatory in BED12+ files."
        )

    standard_cols = BED_FIELD_NAMES[:n]
    fill_cols = list(set(standard_cols) - set(df.columns))
    data_cols = list(set(standard_cols) - set(fill_cols))
    custom_cols = list(set(df.columns) - set(standard_cols)) if extended else []

    fields_with_nulls = set()
    if validate_fields:
        dtypes_failed, fields_with_nulls, props_failed = validate_bed_fields(
            df, data_cols, chromsizes=chromsizes, strict_score=strict_score
        )
        if dtypes_failed:
            raise TypeError(f"Fields contain invalid dtypes: {dtypes_failed}.")
        if fields_with_nulls and not replace_na:
            raise ValueError(f"Fields contain null values: {fields_with_nulls}.")
        if props_failed:
            raise ValueError(f"Properties that failed validation: {props_failed}.")

    if require_sorted:
        props = pd.Series(check_is_sorted(df))
        props_failed = props.index[~props].tolist()
        if props_failed:
            raise ValueError(f"DataFrame isn't properly sorted: {props_failed}.")

    bed = pd.DataFrame(index=df.index)
    for col in standard_cols:
        if col in fill_cols:
            if col == "thickStart":
                bed[col] = df["start"]
            elif col == "thickEnd":
                bed[col] = df["end"]
            else:
                bed[col] = BED_FIELD_FILLVALUES[col]
        elif col in fields_with_nulls:
            warnings.warn(
                f"Standard column {col} contains null values. "
                "These will be replaced with the uninformative value "
                f"{BED_FIELD_FILLVALUES[col]}."
            )
            bed[col] = df[col].fillna(BED_FIELD_FILLVALUES[col])
        else:
            bed[col] = df[col]

    for col in custom_cols:
        bed[col] = df[col]

    return bed.to_csv(path, sep="\t", na_rep=na_rep, index=False, header=False)
