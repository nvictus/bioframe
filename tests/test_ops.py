import pandas as pd
import bioframe
import pyranges as pr
import numpy as np
from io import StringIO


def bioframe_to_pyranges(df):
    pydf = df.copy()
    pydf.rename(
        {"chrom": "Chromosome", "start": "Start", "end": "End"},
        axis="columns",
        inplace=True,
    )
    return pr.PyRanges(pydf)


def pyranges_to_bioframe(pydf):
    df = pydf.df
    df.rename(
        {"Chromosome": "chrom", "Start": "start", "End": "end", "Count": "n_intervals"},
        axis="columns",
        inplace=True,
    )
    return df


def pyranges_overlap_to_bioframe(pydf):
    ## convert the df output by pyranges join into a bioframe-compatible format
    df = pydf.df.copy()
    df.rename(
        {
            "Chromosome": "chrom_1",
            "Start": "start_1",
            "End": "end_1",
            "Start_b": "start_2",
            "End_b": "end_2",
        },
        axis="columns",
        inplace=True,
    )
    df["chrom_1"] = df["chrom_1"].values.astype("object")  # to remove categories
    df["chrom_2"] = df["chrom_1"].values
    return df


chroms = ["chr12", "chrX"]


def mock_bioframe(num_entries=100):
    pos = np.random.randint(1, 1e7, size=(num_entries, 2))
    df = pd.DataFrame()
    df["chrom"] = np.random.choice(chroms, num_entries)
    df["start"] = np.min(pos, axis=1)
    df["end"] = np.max(pos, axis=1)
    df.sort_values(["chrom", "start"], inplace=True)
    return df


############# tests #####################
def test_select():
    df1 = pd.DataFrame(
        [["chrX", 3, 8], ["chr1", 4, 5], ["chrX", 1, 5]],
        columns=["chrom", "start", "end"],
    )

    region1 = "chr1:4-10"
    df_result = pd.DataFrame([["chr1", 4, 5]], columns=["chrom", "start", "end"])
    pd.testing.assert_frame_equal(
        df_result, bioframe.select(df1, region1).reset_index(drop=True)
    )

    region1 = "chrX"
    df_result = pd.DataFrame(
        [["chrX", 3, 8], ["chrX", 1, 5]], columns=["chrom", "start", "end"]
    )
    pd.testing.assert_frame_equal(
        df_result, bioframe.select(df1, region1).reset_index(drop=True)
    )

    region1 = "chrX:4-6"
    df_result = pd.DataFrame(
        [["chrX", 3, 8], ["chrX", 1, 5]], columns=["chrom", "start", "end"]
    )
    pd.testing.assert_frame_equal(
        df_result, bioframe.select(df1, region1).reset_index(drop=True)
    )


def test_expand():
    fake_bioframe = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chr2"], "start": [1, 50, 100], "end": [5, 55, 200]}
    )
    fake_chromsizes = {"chr1": 60, "chr2": 300}
    expand_bp = 10
    fake_expanded = bioframe.expand(fake_bioframe.copy(), expand_bp, fake_chromsizes)
    print(fake_expanded)
    assert fake_expanded.iloc[0].start == 0  # don't expand below zero
    assert (
        fake_expanded.iloc[1].end == fake_chromsizes["chr1"]
    )  # don't expand above chromsize
    assert (
        fake_expanded.iloc[2].end == fake_bioframe.iloc[2].end + expand_bp
    )  # expand end normally
    assert (
        fake_expanded.iloc[2].start == fake_bioframe.iloc[2].start - expand_bp
    )  # expand start normally


def test_overlap():

    ### test consistency of overlap(how='inner') with pyranges.join ###
    ### note does not test overlap_start or overlap_end columns of bioframe.overlap
    df1 = mock_bioframe()
    df2 = mock_bioframe()
    assert df1.equals(df2) == False
    p1 = bioframe_to_pyranges(df1)
    p2 = bioframe_to_pyranges(df2)
    pp = pyranges_overlap_to_bioframe(p1.join(p2, how=None))[
        ["chrom_1", "start_1", "end_1", "chrom_2", "start_2", "end_2"]
    ]
    bb = bioframe.overlap(df1, df2, how="inner")[
        ["chrom_1", "start_1", "end_1", "chrom_2", "start_2", "end_2"]
    ]
    pp = pp.sort_values(
        ["chrom_1", "start_1", "end_1", "chrom_2", "start_2", "end_2"],
        ignore_index=True)
    bb = bb.sort_values(
        ["chrom_1", "start_1", "end_1", "chrom_2", "start_2", "end_2"],
        ignore_index=True)
    pd.testing.assert_frame_equal(bb, pp, check_dtype=False, check_exact=False)
    print("overlap elements agree")

    ### test overlap on= [] ###
    df1 = pd.DataFrame(
        [
            ["chr1", 8, 12, "+", "cat"],
            ["chr1", 8, 12, "-", "cat"],
            ["chrX", 1, 8, "+", "cat"],
        ],
        columns=["chrom1", "start", "end", "strand", "animal"],
    )

    df2 = pd.DataFrame(
        [["chr1", 6, 10, "+", "dog"], ["chrX", 7, 10, "-", "dog"]],
        columns=["chrom2", "start2", "end2", "strand", "animal"],
    )

    b = bioframe.overlap(
        df1,
        df2,
        on=["animal"],
        how="left",
        cols1=("chrom1", "start", "end"),
        cols2=("chrom2", "start2", "end2"),
        return_index=True,
        return_input=False,
    )
    assert np.sum(pd.isna(b["index_2"].values)) == 3

    b = bioframe.overlap(
        df1,
        df2,
        on=["strand"],
        how="left",
        cols1=("chrom1", "start", "end"),
        cols2=("chrom2", "start2", "end2"),
        return_index=True,
        return_input=False,
    )
    assert np.sum(pd.isna(b["index_2"].values)) == 2

    b = bioframe.overlap(
        df1,
        df2,
        on=None,
        how="left",
        cols1=("chrom1", "start", "end"),
        cols2=("chrom2", "start2", "end2"),
        return_index=True,
        return_input=False,
    )
    assert np.sum(pd.isna(b["index_2"].values)) == 0

    ### test overlap 'left', 'outer', and 'right'
    b = bioframe.overlap(
        df1,
        df2,
        on=None,
        how="outer",
        cols1=("chrom1", "start", "end"),
        cols2=("chrom2", "start2", "end2"),
    )
    assert len(b) == 3

    b = bioframe.overlap(
        df1,
        df2,
        on=["animal"],
        how="outer",
        cols1=("chrom1", "start", "end"),
        cols2=("chrom2", "start2", "end2"),
    )
    assert len(b) == 5

    b = bioframe.overlap(
        df1,
        df2,
        on=["animal"],
        how="inner",
        cols1=("chrom1", "start", "end"),
        cols2=("chrom2", "start2", "end2"),
    )
    assert len(b) == 0

    b = bioframe.overlap(
        df1,
        df2,
        on=["animal"],
        how="right",
        cols1=("chrom1", "start", "end"),
        cols2=("chrom2", "start2", "end2"),
    )
    assert len(b) == 2

    b = bioframe.overlap(
        df1,
        df2,
        on=["animal"],
        how="left",
        cols1=("chrom1", "start", "end"),
        cols2=("chrom2", "start2", "end2"),
    )
    assert len(b) == 3


def test_cluster():
    df1 = pd.DataFrame(
        [["chr1", 1, 5], ["chr1", 3, 8], ["chr1", 8, 10], ["chr1", 12, 14],],
        columns=["chrom", "start", "end"],
    )
    df_annotated = bioframe.cluster(df1)
    assert (
        df_annotated["cluster"].values == np.array([0, 0, 0, 1])
    ).all()  # the last interval does not overlap the first three
    df_annotated = bioframe.cluster(df1, min_dist=2)
    assert (
        df_annotated["cluster"].values == np.array([0, 0, 0, 0])
    ).all()  # all intervals part of the same cluster

    df_annotated = bioframe.cluster(df1, min_dist=None)
    assert (
        df_annotated["cluster"].values == np.array([0, 0, 1, 2])
    ).all()  # adjacent intervals not clustered

    df1.iloc[0, 0] = "chrX"
    df_annotated = bioframe.cluster(df1)
    assert (
        df_annotated["cluster"].values == np.array([2, 0, 0, 1])
    ).all()  # do not cluster intervals across chromosomes

    # test consistency with pyranges (which automatically sorts df upon creation and uses 1-based indexing for clusters)
    assert (
        (bioframe_to_pyranges(df1).cluster(count=True).df["Cluster"].values - 1)
        == bioframe.cluster(df1.sort_values(["chrom", "start"]))["cluster"].values
    ).all()

    # test on=[] argument
    df1 = pd.DataFrame(
        [
            ["chr1", 3, 8, "+", "cat", 5.5],
            ["chr1", 3, 8, "-", "dog", 6.5],
            ["chr1", 6, 10, "-", "cat", 6.5],
            ["chrX", 6, 10, "-", "cat", 6.5],
        ],
        columns=["chrom", "start", "end", "strand", "animal", "location"],
    )
    assert (
        bioframe.cluster(df1, on=["animal"])["cluster"].values == np.array([0, 1, 0, 2])
    ).all()
    assert (
        bioframe.cluster(df1, on=["strand"])["cluster"].values == np.array([0, 1, 1, 2])
    ).all()
    assert (
        bioframe.cluster(df1, on=["location", "animal"])["cluster"].values
        == np.array([0, 2, 1, 3])
    ).all()


def test_merge():
    df1 = pd.DataFrame(
        [["chr1", 1, 5], ["chr1", 3, 8], ["chr1", 8, 10], ["chr1", 12, 14],],
        columns=["chrom", "start", "end"],
    )

    # the last interval does not overlap the first three with default min_dist=0
    assert (bioframe.merge(df1)["n_intervals"].values == np.array([3, 1])).all()

    # adjacent intervals are not clustered with min_dist=none
    assert (
        bioframe.merge(df1, min_dist=None)["n_intervals"].values == np.array([2, 1, 1])
    ).all()

    # all intervals part of one cluster
    assert (
        bioframe.merge(df1, min_dist=2)["n_intervals"].values == np.array([4])
    ).all()

    df1.iloc[0, 0] = "chrX"
    assert (
        bioframe.merge(df1, min_dist=None)["n_intervals"].values
        == np.array([1, 1, 1, 1])
    ).all()
    assert (
        bioframe.merge(df1, min_dist=0)["n_intervals"].values == np.array([2, 1, 1])
    ).all()

    # total number of intervals should equal length of original dataframe
    mock_df = mock_bioframe()
    assert np.sum(bioframe.merge(mock_df, min_dist=0)["n_intervals"].values) == len(
        mock_df
    )

    # test consistency with pyranges
    pd.testing.assert_frame_equal(
        pyranges_to_bioframe(bioframe_to_pyranges(df1).merge(count=True)),
        bioframe.merge(df1),
        check_dtype=False,
        check_exact=False,
    )

    # test on=['chrom',...] argument
    df1 = pd.DataFrame(
        [
            ["chr1", 3, 8, "+", "cat", 5.5],
            ["chr1", 3, 8, "-", "dog", 6.5],
            ["chr1", 6, 10, "-", "cat", 6.5],
            ["chrX", 6, 10, "-", "cat", 6.5],
        ],
        columns=["chrom", "start", "end", "strand", "animal", "location"],
    )
    assert len(bioframe.merge(df1, on=None)) == 2
    assert len(bioframe.merge(df1, on=["strand"])) == 3
    assert len(bioframe.merge(df1, on=["strand", "location"])) == 3
    assert len(bioframe.merge(df1, on=["strand", "location", "animal"])) == 4
    d = """ chrom   start   end animal  n_intervals
        0   chr1    3   10  cat 2
        1   chr1    3   8   dog 1
        2   chrX    6   10  cat 1"""
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    pd.testing.assert_frame_equal(
        df, bioframe.merge(df1, on=["animal"]), check_dtype=False,
    )


def test_complement():
    df1 = pd.DataFrame(
        [["chr1", 1, 5], ["chr1", 3, 8], ["chr1", 8, 10], ["chr1", 12, 14]],
        columns=["chrom", "start", "end"],
    )
    df1_chromsizes = {"chr1": 100, "chrX": 100}

    df1_complement = pd.DataFrame(
        [["chr1", 0, 1], ["chr1", 10, 12], ["chr1", 14, 100], ['chrX', 0, 100]],
        columns=["chrom", "start", "end"],
    )

    pd.testing.assert_frame_equal(
        bioframe.complement(df1, chromsizes=df1_chromsizes), df1_complement
    )

    ### test complement with two chromosomes ###
    df1.iloc[0, 0] = "chrX"
    df1_complement = pd.DataFrame(
        [
            ["chr1", 0, 3],
            ["chr1", 10, 12],
            ["chr1", 14, 100],
            ["chrX", 0, 1],
            ["chrX", 5, 100],
        ],
        columns=["chrom", "start", "end"],
    )
    pd.testing.assert_frame_equal(
        bioframe.complement(df1, chromsizes=df1_chromsizes), df1_complement
    )


def test_closest():
    df1 = pd.DataFrame([["chr1", 1, 5],], columns=["chrom", "start", "end"])

    df2 = pd.DataFrame(
        [["chr1", 4, 8], ["chr1", 10, 11]], columns=["chrom", "start", "end"]
    )

    ### closest(df1,df2,k=1) ###
    d = """chrom_1  start_1  end_1 chrom_2  start_2  end_2  distance
        0    chr1        1      5    chr1        4      8         0"""
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    pd.testing.assert_frame_equal(df, bioframe.closest(df1, df2, k=1))

    ### closest(df1,df2, ignore_overlaps=True)) ###
    d = """chrom_1 start_1 end_1   chrom_2 start_2 end_2   distance
        0   chr1    1   5   chr1    10  11  5"""
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    pd.testing.assert_frame_equal(df, bioframe.closest(df1, df2, ignore_overlaps=True))

    ### closest(df1,df2,k=2) ###
    d = """chrom_1 start_1 end_1   chrom_2 start_2 end_2   distance
            0   chr1    1   5   chr1    4   8   0
            1   chr1    1   5   chr1    10  11  5"""
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    pd.testing.assert_frame_equal(df, bioframe.closest(df1, df2, k=2))

    ### closest(df2,df1) ###
    d = """chrom_1  start_1 end_1   chrom_2 start_2 end_2   distance
            0   chr1    4   8   chr1    1   5   0
            1   chr1    10  11  chr1    1   5   5 """
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    pd.testing.assert_frame_equal(df, bioframe.closest(df2, df1))

    ### change first interval to new chrom ###
    df2.iloc[0, 0] = "chrA"
    d = """chrom_1 start_1 end_1   chrom_2 start_2 end_2   distance
              0   chr1    1   5   chr1    10  11  5"""
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    pd.testing.assert_frame_equal(df, bioframe.closest(df1, df2, k=1))

    ### test other return arguments ###
    df2.iloc[0, 0] = "chr1"
    d = """
        index_1 index_2 have_overlap    overlap_start   overlap_end distance
        0   0   0   True    4   5   0
        1   0   1   False   <NA>    <NA>    5
        """
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    pd.testing.assert_frame_equal(
        df,
        bioframe.closest(
            df1,
            df2,
            k=2,
            return_overlap=True,
            return_index=True,
            return_input=False,
            return_distance=True,
        ),
        check_dtype=False,
    )


def test_coverage():

    #### coverage does not exceed length of original interval
    df1 = pd.DataFrame([["chr1", 3, 8]], columns=["chrom", "start", "end"])
    df2 = pd.DataFrame([["chr1", 2, 10]], columns=["chrom", "start", "end"])
    d = """chrom    start   end coverage
         0  chr1    3   8   5"""
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    print(df)
    print(bioframe.coverage(df1, df2))
    pd.testing.assert_frame_equal(df, bioframe.coverage(df1, df2))

    ### coverage of interval on different chrom returns zero for coverage and n_overlaps
    df1 = pd.DataFrame([["chr1", 3, 8]], columns=["chrom", "start", "end"])
    df2 = pd.DataFrame([["chrX", 3, 8]], columns=["chrom", "start", "end"])
    d = """chrom    start   end coverage
        0  chr1      3       8     0   """
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    pd.testing.assert_frame_equal(df, bioframe.coverage(df1, df2))

    ### when a second overlap starts within the first
    df1 = pd.DataFrame([["chr1", 3, 8]], columns=["chrom", "start", "end"])
    df2 = pd.DataFrame(
        [["chr1", 3, 6], ["chr1", 5, 8]], columns=["chrom", "start", "end"]
    )

    d = """chrom    start   end coverage
         0  chr1     3       8     5"""
    df = pd.read_csv(StringIO(d), sep=r"\s+")
    pd.testing.assert_frame_equal(df, bioframe.coverage(df1, df2))


def test_subtract():
    df1 = pd.DataFrame(
        [["chrX", 3, 8], ["chr1", 4, 7], ["chrX", 1, 5]],
        columns=["chrom", "start", "end"],
    )
    assert len(bioframe.subtract(df1, df1)) == 0

    df2 = pd.DataFrame(
        [["chrX", 0, 18], ["chr1", 5, 6],], columns=["chrom", "start", "end"]
    )

    df1["animal"] = "sea-creature"
    df_result = pd.DataFrame(
        [["chr1", 4, 5, "sea-creature"], ["chr1", 6, 7, "sea-creature"]],
        columns=["chrom", "start", "end", "animal"],
    )
    pd.testing.assert_frame_equal(
        df_result, 
        bioframe.subtract(df1, df2)
            .sort_values(['chrom','start','end'])
            .reset_index(drop=True)
    )

    df2 = pd.DataFrame(
        [["chrX", 0, 4], ["chr1", 6, 6], ["chrX", 4, 9]],
        columns=["chrom", "start", "end"],
    )

    df1["animal"] = "sea-creature"
    df_result = pd.DataFrame(
        [["chr1", 4, 6, "sea-creature"], ["chr1", 6, 7, "sea-creature"]],
        columns=["chrom", "start", "end", "animal"],
    )
    print(bioframe.subtract(df1, df2))
    pd.testing.assert_frame_equal(
        df_result, 
        bioframe.subtract(df1, df2)
            .sort_values(['chrom','start','end'])
            .reset_index(drop=True)
    )


def test_setdiff():

    df1 = pd.DataFrame(
        [
            ["chr1", 8, 12, "+", "cat"],
            ["chr1", 8, 12, "-", "cat"],
            ["chrX", 1, 8, "+", "cat"],
        ],
        columns=["chrom1", "start", "end", "strand", "animal"],
    )

    df2 = pd.DataFrame(
        [
            ["chrX", 7, 10, "-", "dog"],
            ["chr1", 6, 10, "-", "cat"],
            ["chr1", 6, 10, "-", "cat"],
        ],
        columns=["chrom2", "start", "end", "strand", "animal"],
    )

    assert (
        len(
            bioframe.setdiff(
                df1,
                df2,
                cols1=("chrom1", "start", "end"),
                cols2=("chrom2", "start", "end"),
                on=None,
            )
        )
        == 0
    )  # everything overlaps

    assert (
        len(
            bioframe.setdiff(
                df1,
                df2,
                cols1=("chrom1", "start", "end"),
                cols2=("chrom2", "start", "end"),
                on=["animal"],
            )
        )
        == 1
    )  # two overlap, one remains

    assert (
        len(
            bioframe.setdiff(
                df1,
                df2,
                cols1=("chrom1", "start", "end"),
                cols2=("chrom2", "start", "end"),
                on=["strand"],
            )
        )
        == 2
    )  # one overlaps, two remain


def test_split():
    df1 = pd.DataFrame(
        [["chrX", 3, 8], 
         ["chr1", 4, 7], 
         ["chrX", 1, 5]
        ],
        columns=["chrom", "start", "end"],
    )
   
    df2 = pd.DataFrame(
        [["chrX", 4], 
         ["chr1", 5],], 
         columns=["chrom", "pos"]
    )

    df_result = pd.DataFrame(
        [["chrX", 1, 4],
         ["chrX", 3, 4],
         ["chrX", 4, 5],
         ["chrX", 4, 8], 
         ["chr1", 5, 7],
         ["chr1", 4, 5]
        ],
        columns=["chrom", "start", "end"],
    ).sort_values(['chrom','start','end']).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        df_result, 
        bioframe.split(df1, df2)
            .sort_values(['chrom','start','end'])
            .reset_index(drop=True)
    )

    # Test the case when a chromosome is missing from points.
    df1 = pd.DataFrame(
        [["chrX", 3, 8], 
         ["chr1", 4, 7], 
        ],
        columns=["chrom", "start", "end"],
    )
   
    df2 = pd.DataFrame(
        [["chrX", 4]], 
         columns=["chrom", "pos"]
    )

    df_result = pd.DataFrame(
        [["chrX", 3, 4],
         ["chrX", 4, 8],
         ["chr1", 4, 7],
        ],
        columns=["chrom", "start", "end"],
    ).sort_values(['chrom','start','end']).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        df_result, 
        bioframe.split(df1, df2)
            .sort_values(['chrom','start','end'])
            .reset_index(drop=True)
    )

    df1 = pd.DataFrame(
        [["chrX", 3, 8]],
        columns=["chromosome", "lo", "hi"],
    )
   
    df2 = pd.DataFrame(
        [["chrX", 4]], 
         columns=["chromosome", "loc"]
    )

    df_result = pd.DataFrame(
        [["chrX", 3, 4],
         ["chrX", 4, 8],
        ],
        columns=["chrom", "start", "end"],
    ).sort_values(['chrom','start','end']).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        df_result, 
        bioframe.split(
            df1, df2, 
            cols=['chromosome', 'lo', 'hi'],
            cols_points=['chromosome', 'loc'],)
            .sort_values(['chrom','start','end'])
            .reset_index(drop=True)
    )



def test_count_overlaps():
    df1 = pd.DataFrame(
        [
            ["chr1", 8, 12, "+", "cat"],
            ["chr1", 8, 12, "-", "cat"],
            ["chrX", 1, 8, "+", "cat"],
        ],
        columns=["chrom1", "start", "end", "strand", "animal"],
    )

    df2 = pd.DataFrame(
        [
            ["chr1", 6, 10, "+", "dog"],
            ["chr1", 6, 10, "+", "dog"],
            ["chrX", 7, 10, "+", "dog"],
            ["chrX", 7, 10, "+", "dog"],
        ],
        columns=["chrom2", "start2", "end2", "strand", "animal"],
    )

    assert (
        bioframe.count_overlaps(
            df1,
            df2,
            on=None,
            cols1=("chrom1", "start", "end"),
            cols2=("chrom2", "start2", "end2"),
        )["count"].values
        == np.array([2, 2, 2])
    ).all()

    assert (
        bioframe.count_overlaps(
            df1,
            df2,
            on=["strand"],
            cols1=("chrom1", "start", "end"),
            cols2=("chrom2", "start2", "end2"),
        )["count"].values
        == np.array([2, 0, 2])
    ).all()

    assert (
        bioframe.count_overlaps(
            df1,
            df2,
            on=["strand", "animal"],
            cols1=("chrom1", "start", "end"),
            cols2=("chrom2", "start2", "end2"),
        )["count"].values
        == np.array([0, 0, 0])
    ).all()
