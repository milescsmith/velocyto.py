
# from memory_profiler import profile, LogFile
import sys
from pathlib import Path

from loguru import logger
from velocyto.commands._run import _run
from velocyto.commands.common import LogicType, init_logger

sys.path[0] = str(Path(sys.path[0]).parent)

# from memory_profiler import profile, LogFile


# run10x(
#     samplefolder=Path("/mnt/vault/PAM/PAM1"),
#     gtffile=Path("/mnt/vault/Homo_sapiens.GRCh38.94.chr_patch_hapl_scaff.gtf.gz"),
#     mask=Path("/mnt/group/references/genomic/homo_sapiens/sequences/grch3810_repeat_mask.gtf")
# )

def main():
    logger.remove()
    init_logger(3, msg_format="<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    # sys.stdout = LogFile("memory_profile_log", reportIncrementFlag=False)

    samplefolder = Path("/mnt/vault/workspace/analysis/pam")

    _run(
            bam_input=samplefolder.joinpath("subsample_alignments.bam"),
            gtffile=samplefolder.joinpath("genes.gtf"),
            bcfile=samplefolder.joinpath("barcodes.tsv.gz"),
            outputfolder=samplefolder.joinpath("velocyto"),
            sampleid="PAM1",
            metadatatable=None,
            repmask=samplefolder.joinpath("grch3810_repeat_mask.gtf"),
            onefilepercell=False,
            logic=LogicType.Permissive10X,
            without_umi=False,
            umi_extension="no",
            multimap=False,
            test=False,
            samtools_threads=4,
            samtools_memory="4G",
            dump="0",
            loom_numeric_dtype="uint32",
            verbose=True,
            bughunting=True,
            is_10X=True,
            samplefolder=samplefolder,
        )

main()
