import gzip
import multiprocessing
import subprocess
import sys
from distutils.spawn import find_executable
from functools import reduce
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy as sp
from loguru import logger

from velocyto.commands.common import LogicType, choose_dtype, choose_logic, id_generator
from velocyto.constants import BAM_COMPRESSION
from velocyto.counter import ExInCounter
from velocyto.metadata import MetadataCollection


def _run(
    *,
    bam_input: tuple[Path],
    gtffile: Path,
    sampleid: str,
    metadatatable: str,
    repmask: str,
    onefilepercell: bool,
    logic: str,
    without_umi: str,
    umi_extension: str,
    multimap: bool,
    samtools_threads: int,
    samtools_memory: str,
    loom_numeric_dtype: str,
    dump: str,
    verbose: int,
    bcfile: Path | None = None,
    outputfolder: Path | None = None,
    samtools_path: Path | None = None,
    **kwargs,
) -> None:
    """Runs the velocity analysis outputing a loom file

    BAMFILE or [BAMFILES] one or several bam files with position-sorted

    GTFFILE annotation file

    NOTE: it is keyword only argument function
    """

    ########################
    #    Resolve Inputs    #
    ########################
    if samtools_path is None:
        samtools = find_executable("samtools")
        if samtools is None:
            logger.error("samtools was not found")
            msg = "Samtools was not found. Make sure that it is both installed and on the system path"
            raise FileNotFoundError(msg)

    loom_numeric_dtype = choose_dtype(loom_numeric_dtype)

    bamfile, multi = parse_bam_input(bam_input)

    if onefilepercell and multi:
        if bcfile:
            msg = "Inputs incompatibility. --bcfile/-b option was used together with --onefilepercell/-c option."
            raise ValueError(msg)
        else:
            logger.warning("Each bam file will be interpreted as a DIFFERENT cell")
    elif not onefilepercell and multi:
        logger.warning(
            "Several input files but --onefilepercell is False. Each bam file will be interpreted as containing a SET of cells!"
        )

    sampleid = resolve_sampleid(sampleid, metadatatable, onefilepercell, bamfile, multi)

    # Create an output folder inside the cell ranger output folder
    if outputfolder is None:
        outputfolder = bamfile[0].parent.joinpath("velocyto")
    if not outputfolder.exists():
        outputfolder.mkdir(parents=True, exist_ok=False)

    if logic not in LogicType:
        msg = f"{logic} is not a valid logic. Choose one among {', '.join([_.value for _ in LogicType])}"
        raise ValueError(msg)
    logger.debug(f"Using logic: {logic}")
    logic = choose_logic(logic)
    logic_obj = logic()

    if bcfile is None:
        logger.debug("Cell barcodes will be determined while reading the .bam file")
        valid_bcset = None
        gem_grp = "x"
    else:
        valid_bcset, gem_grp = get_barcodes(bcfile, sampleid)

    # Initialize Exon-Intron Counter with the logic and valid barcodes (need to do it now to peek)
    if without_umi:
        if umi_extension != "no":
            logger.warning("--umi-extension was specified but incompatible with --without-umi, it will be ignored!")
        umi_extension = "without_umi"

    exincounter = ExInCounter(
        sampleid=sampleid,
        logic=logic,
        valid_bcset=valid_bcset,
        umi_extension=umi_extension,
        onefilepercell=onefilepercell,
        dump_option=dump,
        outputfolder=outputfolder,
    )

    # TODO: if we can, should check to see if the bamfile is already sorted.
    if multi and onefilepercell:
        bamfile_cellsorted = list(bamfile)
    elif onefilepercell:
        bamfile_cellsorted = [bamfile[0]]
    else:
        bamfile_cellsorted = [f"{bmf.parent.joinpath(f'cellsorted_{bmf.name}')}" for bmf in bamfile]

    # I need to peek into the bam file to know wich cell barcode flag should be used
    if onefilepercell and without_umi:
        tagname = "NOTAG"
    elif onefilepercell:
        logger.debug("The multi input option ")
        tagname = "NOTAG"
        exincounter.peek_umi_only(bamfile[0])
    else:
        exincounter.peek(bamfile[0])
        tagname = exincounter.cellbarcode_str

    sort_bamfiles(
        bamfile,
        bamfile_cellsorted,
        tagname,
        samtools_memory,
        samtools_threads,
    )

    load_annotations(gtffile, exincounter)

    logger.info(f"Scan {' '.join(str(_) for _ in bamfile)} to validate intron intervals")
    if repmask is not None:
        logger.info(f"Load the repeat masking annotation from {repmask}")
        exincounter.read_repeats(repmask)

    exincounter.mark_up_introns(bamfile=bamfile, multimap=multimap)

    # Do the actual counting
    logger.debug("Start molecule counting!")
    results = exincounter.count(
        bamfile_cellsorted, multimap=multimap
    )  # NOTE: we would avoid some millions of if statements evaluations if we write two function count and count_with output
    dict_list_arrays, cell_bcs_order = results

    # Prepare the loom file output
    if not exincounter.filter_mode:
        valid_bcset = exincounter.valid_bcset  # without -1
        valid_bcs_list = list(valid_bcset)  # without -1
        gem_grp = ""
        valid_cellid_list = np.array([f"{sampleid}:{v_bc}" for v_bc in valid_bcs_list])  # with sampleid and with -1
        logger.debug(f"Example of barcode: {valid_bcs_list[0]} and cell_id: {valid_cellid_list[0]}")

    obs_df = pd.DataFrame(index=np.array([f"{sampleid}:{v_bc}{gem_grp}" for v_bc in cell_bcs_order]))

    # var dataframe attributes
    logger.debug("Collecting row attributes")
    var_df = pd.DataFrame(
        {
            "Gene": [exincounter.genes[k].genename for k in exincounter.genes],
            "Accession": [exincounter.genes[k].geneid for k in exincounter.genes],
            "Chromosome": [exincounter.genes[k].chrom for k in exincounter.genes],
            "Strand": [exincounter.genes[k].strand for k in exincounter.genes],
            "Start": [exincounter.genes[k].start for k in exincounter.genes],
            "End": [exincounter.genes[k].end for k in exincounter.genes],
        },
        index=[exincounter.genes[k].genename for k in exincounter.genes],
    )

    logger.debug("Generating data table")
    layers: dict[str, np.ndarray] = {
        layer_name: sp.sparse.hstack(dict_list_arrays[layer_name], dtype=loom_numeric_dtype).transpose().tocsc()
        for layer_name in logic_obj.layers
    }
    total = reduce(np.add, layers.items())
    outfile = outputfolder.joinpath(f"{sampleid}.h5ad")
    logger.debug(f"Creating anndata object at {outfile}")

    try:
        adata = ad.AnnData(
            X=total[1],
            obs=obs_df,
            var=var_df,
            layers=layers,
        )
        adata.write_h5ad(str(outfile))

    except TypeError as e:
        logger.error(e)
        sys.exit()

    logger.debug("Terminated Succesfully!")


def get_barcodes(bcfile, sampleid):
    # Get valid cell barcodes
    valid_bcs_list = (
        (gzip.open(bcfile).read().decode() if bcfile.suffix == ".gz" else open(bcfile).read()).rstrip().split()
    )
    valid_cellid_list = np.array([f"{sampleid}:{v_bc}" for v_bc in valid_bcs_list])  # with sample id and with -1
    if len({bc.split("-")[0] for bc in valid_bcs_list}) == 1:
        gem_grp = f"-{valid_bcs_list[0].split('-')[-1]}"
    else:
        gem_grp = "x"
    # need to peek at the barcodes - newer "sample_filtered_barcodes.csv" style files from cellranger 7.1+ multi
    # have a "GRCh38," prefix
    if "," in valid_bcs_list[0]:
        valid_bcset = {bc.split(",")[1].split("-")[0] for bc in valid_bcs_list}  # without -1
    else:
        valid_bcset = {bc.split("-")[0] for bc in valid_bcs_list}  # without -1
    logger.info(f"Read {len(valid_bcs_list)} cell barcodes from {bcfile}")
    logger.debug(f"Example of barcode: {valid_bcs_list[0].split('-')[0]} and cell_id: {valid_cellid_list[0]}")
    return valid_bcset, gem_grp


def get_metadata(sampleid, metadatatable):
    if metadatatable:
        try:
            sample_metadata = MetadataCollection(metadatatable)
            sample = sample_metadata.where("SampleID", sampleid)
            if len(sample) == 0:
                logger.error(f"Sample ID {sampleid} not found in sample sheet")
                # schema = []  # type: list
                sample = {}
            elif len(sample) > 1:
                logger.error(f"Sample ID {sampleid} has multiple lines in sample sheet")
                sys.exit(1)
            else:
                # schema = sample[0].types
                sample = sample[0].dict
            logger.debug(f"Collecting column attributes from {metadatatable}")
        except (NameError, TypeError):
            logger.warn("SAMPLEFILE was not specified. add -s SAMPLEFILE to add metadata.")
            sample = {}
    else:
        sample = {}
    return sample


def sort_bamfiles(
    bamfile: list[Path],
    bamfile_cellsorted: list[Path],
    tagname: str,
    samtools_memory: str,
    samtools_threads: int,
):
    threads_to_use = min(samtools_threads, multiprocessing.cpu_count())
    sorting_process: dict[int, Any] = {}
    for ni, bmf_cellsorted in enumerate(bamfile_cellsorted):
        # Start a subprocess that sorts the bam file
        command = f"samtools sort -l {BAM_COMPRESSION} -m {samtools_memory} -t {tagname} -O BAM -@ {threads_to_use} -o {bmf_cellsorted} {bamfile[ni]}"
        if Path(bmf_cellsorted).exists():
            # This should skip sorting in smartseq2
            logger.warning(
                f"The file {bmf_cellsorted} already exists. The sorting step will be skipped and the existing file will be used."
            )
            check_end_process = False
        else:
            sorting_process[ni] = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            logger.info(f"Starting the sorting process of {bamfile[ni]} the output will be at: {bmf_cellsorted}")
            logger.info(f"Command being run is: {command}")
            check_end_process = True

    # Wait for child process to terminate
    if check_end_process:
        logger.info("Now just waiting that the bam sorting process terminates")
        for k in sorting_process:
            returncode = sorting_process[k].wait()
            if returncode == 0:
                logger.info(f"bam file #{k} has been sorted")
            else:
                msg = f"bam file #{k} could not be sorted by cells.\n\tThis is probably related to an old version of samtools, please install samtools >= 1.6.                In alternative this could be a memory error, try to set the --samtools_memory option to a value compatible with your system.                 Otherwise sort manually by samtools ``sort -l [compression] -m [mb_to_use]M -t [tagname] -O BAM -@ [threads_to_use] -o cellsorted_[bamfile] [bamfile]``"
                raise MemoryError(msg)


def load_annotations(gtffile, exincounter):
    logger.info(f"Load the annotation from {gtffile}")
    features, chrm_strands = exincounter.read_transcriptmodels(gtffile)
    logger.debug(f"Generated {features} features corresponding to {chrm_strands} chromosome strands from {gtffile}")


def resolve_sampleid(sampleid, metadatatable, onefilepercell, bamfile, multi):
    if sampleid is None:
        if metadatatable is None:
            msg = "--metadatatable was specified but cannot fetch sample metadata without valid sampleid"
            logger.exception(msg)
        if multi:
            logger.warning(
                "When using mutliple files you may want to use --sampleid option to specify the name of the output file"
            )
        if multi and not onefilepercell:
            full_name = "_".join([bamfile[i].name.split(".")[0] for i in range(len(bamfile))])
            FILENAME_IS_TOO_LONG = 50
            if len(full_name) > FILENAME_IS_TOO_LONG:
                sampleid = f'multi_input_{bamfile[0].name.split(".")[0]}_{id_generator(5)}'
            else:
                sampleid = f"multi_input_{full_name}_and_others_{id_generator(5)}"
        elif multi:
            sampleid = f'onefilepercell_{bamfile[0].name.split(".")[0]}_and_others_{id_generator(5)}'
        else:
            sampleid = f'{bamfile[0].name.split(".")[0]}_{id_generator(5)}'
        logger.info(
            f"No SAMPLEID specified, the sample will be called {sampleid} (last 5 digits are a random-id to avoid overwriting some other file by mistake)"
        )

    return sampleid


def parse_bam_input(bamfile):
    if isinstance(bamfile, str | Path):
        bamfile = (bamfile,)
    if isinstance(bamfile, tuple) and len(bamfile) > 1 and bamfile.suffix in [".bam", ".sam"]:
        multi = True
    elif isinstance(bamfile, tuple) and len(bamfile) == 1:
        multi = False
    else:
        msg = f"Something went wrong in the argument parsing. You passed as bamfile: {bamfile}"
        raise OSError(msg)
    return bamfile, multi


# def extract_cellranger_umap(umap_df, cell_bcs_order):
#     result = pd.read_csv(umap_df)
#     result["Barcode"] = [_[:-2] for _ in result["Barcode"]]
#     result = result[result["Barcode"].isin(cell_bcs_order)]
#     return result
