# from velocyto.commands._run import _run
from velocyto.commands.run10x import run10x
from velocyto.commands.common import logicType, init_logger
from pathlib import Path
import typer
from loguru import logger
import numpy as np
# from memory_profiler import profile, LogFile
import sys

project_root = Path().home().joinpath("workspace", "datasets", "ana_multiome")

logger.remove()
init_logger(3, msg_format="<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
# sys.stdout = LogFile('memory_profile_log', reportIncrementFlag=False)

samplefolder = project_root.joinpath('data', 'counts', "04_pam_33-48", "PAM33")

umap_file = next(samplefolder.joinpath("outs", "per_sample_outs").rglob("umap/gene_expression_2_components/projection.csv"))
umap = np.loadtxt(umap_file, usecols=(1, 2), delimiter=",", skiprows=1)

clusters_file = next(samplefolder.joinpath("outs", "per_sample_outs").rglob("gene_expression_graphclust/clusters.csv"))
labels = np.loadtxt(clusters_file, usecols=(1,), delimiter=",", skiprows=1)


additional_ca = {
    "_X": umap[:, 0].astype("float32"),
    "_Y": umap[:, 1].astype("float32"),
    "Clusters": labels.astype("int") - 1
}
typer.run(
    run10x(
        samplefolder=samplefolder,
        gtffile=project_root.joinpath("Homo_sapiens.GRCh38.94.chr_patch_hapl_scaff.gtf.gz"),
        mask=project_root.joinpath("grch3810_repeat_mask.gtf"),
        logic=logicType.Permissive10X,
        dump="0",
        # multimap=False,
        samtools_threads=16,
        samtools_memory="4G",
        dtype='uint32',
        verbose=True,
    )
)
# _run(
#         bamfile=(samplefolder.joinpath('outs','per_sample_outs','PAM33','count','sample_alignments.bam'),),
#         gtffile=project_root.joinpath("ana_multiome","Homo_sapiens.GRCh38.94.chr_patch_hapl_scaff.gtf.gz"),
#         bcfile=Path(samplefolder.joinpath('outs','per_sample_outs','PAM33','count','sample_filtered_feature_bc_matrix','barcodes.tsv.gz')),
#         outputfolder=project_root.joinpath("data","velocity","PAM33"),
#         sampleid="PAM33",
#         metadatatable=None,
#         repmask=project_root.joinpath("ana_multiome", "grch3810_repeat_mask.gtf"),
#         onefilepercell=False,
#         logic=logicType.Permissive10X,
#         without_umi=False,
#         umi_extension="no",
#         multimap=False,
#         test=True,
#         samtools_threads=16,
#         samtools_memory=1024,
#         dump="0",
#         loom_numeric_dtype='uint32',
#         verbose=True,
#         bughunting=True,
#         additional_ca=additional_ca,
#         is_10X=True,
#         samplefolder=samplefolder,
#     )
