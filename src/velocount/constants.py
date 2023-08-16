# Should this all be dumped into a JSON file?
MATCH_INSIDE = 1
MATCH_OVER5END = 2
MATCH_OVER3END = 4

MIN_FLANK = 5
PATCH_INDELS = 3
SPLIC_INACUR = 6
MIN_POLYT = 8
MAX_USHORT = 2**16 - 1

LOOM_NUMERIC_DTYPE = "uint16"

EXTENSION5_LEN = 0  # basepairs to extend 5' ends of models
EXTENSION3_LEN = 0  # basepairs to extend 3' ends of models

BINSIZE_BP = 100000  # binsize to look up for other transcripts used during interval determination
LONGEST_INTRON_ALLOWED = 1000000
BAM_COMPRESSION = 7

EXON = 1
ALT_EXON = 2  # From another transcript of same gene
OTHER_EXON = 4  # From an unrelated gene
INTRON = 8
ALT_INTRON = 16
OTHER_INTRON = 32
COMPETING_INTRON = ALT_INTRON | OTHER_INTRON
COMPETING_EXON = ALT_EXON | OTHER_EXON

PLACEHOLDER_UMI_LEN = (
    12  # the length of the placeholder random umi added if -U is set (complexity is 36 per unit length)
)

CIGAR = {
    0: "BAM_CMATCH",
    1: "BAM_CINS",
    2: "BAM_CDEL",
    3: "BAM_CREF_SKIP",
    4: "BAM_CSOFT_CLIP",
    5: "BAM_CHARD_CLIP",
    6: "BAM_CPAD",
    7: "BAM_CEQUAL",
    8: "BAM_CDIFF",
    9: "BAM_CBACK",
}  # type: dict  # currently hard coded for speed
