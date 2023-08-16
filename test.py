
from velocount.commands._run import _run
# from velocount.commands.run10x import run10x
from velocount.commands.common import logicType, init_logger
from pathlib import Path
from loguru import logger

import sys
import better_exceptions
sys.path[0] = str(Path(sys.path[0]).parent)


better_exceptions.hook()
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
            outputfolder=samplefolder.joinpath("velocount"),
            sampleid="PAM1",
            metadatatable=None,
            repmask=samplefolder.joinpath("grch3810_repeat_mask.gtf"),
            onefilepercell=False,
            logic=logicType.Permissive10X,
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
