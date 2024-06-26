class GeneInfo:
    """A simple objects that stores basic info on a gene.
    Parsed from the .gtf file and used to build the row_attrs of the loom file"""

    __slots__ = ["chrom", "end", "geneid", "genename", "start", "strand"]

    def __init__(self, genename: str, geneid: str, chromstrand: str, start: int, end: int) -> None:
        self.genename = genename
        self.geneid = geneid
        self.chrom = chromstrand[:-1]
        self.strand = chromstrand[-1]
        self.start = start
        self.end = end
