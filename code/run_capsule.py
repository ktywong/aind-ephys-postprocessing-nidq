import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import shutil
import json
import argparse
import time
import logging
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc

from spikeinterface.core.core_tools import check_json

# AIND
from aind_data_schema.core.processing import DataProcess

try:
    from aind_log_utils import log

    HAVE_AIND_LOG_UTILS = True
except ImportError:
    HAVE_AIND_LOG_UTILS = False

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-postprocessing"
VERSION = "1.0"

data_folder = Path("../data/")
scratch_folder = Path("../scratch")
results_folder = Path("../results/")

# Define argument parser
parser = argparse.ArgumentParser(description="Postprocess ecephys data")

use_motion_corrected_group = parser.add_mutually_exclusive_group()
use_motion_corrected_help = (
    "If True and motion corrected has been computed (and not applied), "
    "it applies motion interpolation to the recording prior to postprocessing."
)
use_motion_corrected_group.add_argument("static_use_motion_corrected", nargs="?", default="false", help=use_motion_corrected_help)
use_motion_corrected_group.add_argument("--use-motion-corrected", action="store_true", help=use_motion_corrected_help)

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is 0.8 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default="-1", help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

params_group = parser.add_mutually_exclusive_group()
params_file_help = "Optional json file with parameters"
params_group.add_argument("static_params_file", nargs="?", default=None, help=params_file_help)
params_group.add_argument("--params-file", default=None, help=params_file_help)
params_group.add_argument("--params-str", default=None, help="Optional json string with parameters")


if __name__ == "__main__":
    args = parser.parse_args()

    USE_MOTION_CORRECTED = args.use_motion_corrected or args.static_use_motion_corrected == "true"
    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS_FILE = args.static_params_file or args.params_file
    PARAMS_STR = args.params_str

    # Use CO_CPUS env variable if available
    N_JOBS_CO = os.getenv("CO_CPUS")
    if N_JOBS_CO is not None:
        if isinstance(N_JOBS, float):
            N_JOBS = int(N_JOBS * int(N_JOBS_CO))
        elif N_JOBS == -1:
            N_JOBS = int(N_JOBS_CO)
        elif int(N_JOBS_CO) < N_JOBS:
            N_JOBS = int(N_JOBS_CO)

    # setup AIND logging before any other logging call
    ecephys_session_folders = [
        p for p in data_folder.iterdir() if "ecephys" in p.name.lower() or "behavior" in p.name.lower()
    ]
    ecephys_session_folder = None
    aind_log_setup = False
    if len(ecephys_session_folders) == 1:
        ecephys_session_folder = ecephys_session_folders[0]
        if HAVE_AIND_LOG_UTILS:
            # look for subject.json and data_description.json files
            subject_json = ecephys_session_folder / "subject.json"
            subject_id = "undefined"
            if subject_json.is_file():
                subject_data = json.load(open(subject_json, "r"))
                subject_id = subject_data["subject_id"]

            data_description_json = ecephys_session_folder / "data_description.json"
            session_name = "undefined"
            if data_description_json.is_file():
                data_description = json.load(open(data_description_json, "r"))
                session_name = data_description["name"]

            log.setup_logging(
                "Preprocess Ecephys",
                mouse_id=subject_id,
                session_name=session_name,
            )
            aind_log_setup = True

    if not aind_log_setup:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info(f"Running postprocessing with the following parameters:")
    logging.info(f"\tUSE_MOTION_CORRECTED: {USE_MOTION_CORRECTED}")
    logging.info(f"\tN_JOBS: {N_JOBS}")

    if PARAMS_FILE is not None:
        logging.info(f"\nUsing custom parameter file: {PARAMS_FILE}")
        with open(PARAMS_FILE, "r") as f:
            processing_params = json.load(f)
    elif PARAMS_STR is not None:
        processing_params = json.loads(PARAMS_STR)
    else:
        with open("params.json", "r") as f:
            processing_params = json.load(f)

    data_process_prefix = "data_process_postprocessing"

    job_kwargs = processing_params["job_kwargs"]
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    postprocessing_params = processing_params["postprocessing"]
    sparsity_params = processing_params["sparsity"]
    quality_metrics_names = processing_params["quality_metrics_names"]
    quality_metrics_params = processing_params["quality_metrics"]

    ####### POSTPROCESSING ########
    logging.info("\nPOSTPROCESSING")
    t_postprocessing_start_all = time.perf_counter()

    # check if test
    if (data_folder / "preprocessing_pipeline_output_test").is_dir():
        logging.info("\n*******************\n**** TEST MODE ****\n*******************\n")
        preprocessed_folder = data_folder / "preprocessing_pipeline_output_test"
        spikesorted_folder = data_folder / "spikesorting_pipeline_output_test"
    else:
        preprocessed_folder = data_folder
        spikesorted_folder = data_folder

    preprocessed_folders = [p for p in preprocessed_folder.iterdir() if p.is_dir() and "preprocessed_" in p.name]

    # load job json files
    job_config_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    logging.info(f"Found {len(job_config_json_files)} json configurations")

    if len(job_config_json_files) > 0:
        recording_names = []
        for json_file in job_config_json_files:
            with open(json_file, "r") as f:
                config = json.load(f)
            recording_name = config["recording_name"]
            assert (
                preprocessed_folder / f"preprocessed_{recording_name}"
            ).is_dir(), f"Preprocessed folder for {recording_name} not found!"
            recording_names.append(recording_name)
    else:
        recording_names = [("_").join(p.name.split("_")[1:]) for p in preprocessed_folders]

    for recording_name in recording_names:
        datetime_start_postprocessing = datetime.now()
        t_postprocessing_start = time.perf_counter()
        postprocessing_notes = ""
        binary_json_file = preprocessed_folder / f"binary_{recording_name}.json"
        preprocessed_json_file = preprocessed_folder / f"preprocessed_{recording_name}.json"
        motion_corrected_folder = preprocessed_folder / f"motion_{recording_name}"

        logging.info(f"\tProcessing {recording_name}")
        postprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
        postprocessing_output_folder = results_folder / f"postprocessed_{recording_name}.zarr"

        try:
            recording_bin = None
            if binary_json_file.is_file():
                recording_bin = si.load_extractor(binary_json_file, base_folder=preprocessed_folder)
                logging.info(f"\tLoaded binary recording from JSON")
            else:
                recording_bin = si.load_extractor(preprocessed_folder / f"preprocessed_{recording_name}")
            recording_lazy = None
            try:
                if preprocessed_json_file.is_file():
                    logging.info(f"\tLoading lazy recording from JSON")
                    recording_lazy = si.load_extractor(preprocessed_json_file, base_folder=data_folder)
            except:
                logging.info("Could not load lazy preprocessed recording")
        except ValueError as e:
            logging.info(f"Spike sorting skipped on {recording_name}. Skipping postprocessing")
            # create an empty result file (needed for pipeline)
            postprocessing_output_folder.mkdir()
            mock_array = np.array([], dtype=bool)
            np.save(postprocessing_output_folder / "placeholder.npy", mock_array)
            continue

        if USE_MOTION_CORRECTED and motion_corrected_folder is not None:
            from spikeinterface.sortingcomponents.motion import (
                InterpolateMotionRecording,
                interpolate_motion,
            )

            logging.info("\tCorrecting for motion prior to postprocessing")
            if not isinstance(recording, InterpolateMotionRecording):
                logging.info("\t\tApplying motion interpolation")
                motion_info = spre.load_motion_info(motion_corrected_folder)
                interpolate_motion_kwargs = motion_info["parameters"]["interpolate_motion_kwargs"]
                recording_bin_f = spre.astype(recording_bin, "float32")

                recording_bin = interpolate_motion(
                    recording_bin_f,
                    motion=motion_info["motion"],
                    **interpolate_motion_kwargs
                )
                # InterpolateMotion is not compatible with times.
                # Removing time info for postprocessing
                for rec_segment in recording_bin._recording_segments:
                    if rec_segment.time_vector is not None:
                        rec_segment.time_vector = None

                if recording_lazy is not None:
                    recording_lazy_f = spre.astype(recording_bin, "float32")
                    recording_lazy = interpolate_motion(
                        recording_lazy_f,
                        motion=motion_info["motion"],
                        **interpolate_motion_kwargs
                    )
                    # InterpolateMotion is not compatible with times.
                    # Removing time info for postprocessing
                    for rec_segment in recording_lazy._recording_segments:
                        if rec_segment.time_vector is not None:
                            rec_segment.time_vector = None
            else:
                logging.info("\tRecording is already interpolated")

        # make sure we have spikesorted output for the block-stream
        sorted_folder = spikesorted_folder / f"spikesorted_{recording_name}"
        if not sorted_folder.is_dir():
            raise FileNotFoundError(f"Spike sorted data for {recording_name} not found!")

        try:
            sorting = si.load_extractor(sorted_folder)
        except ValueError as e:
            logging.info(f"Spike sorting failed on {recording_name}. Skipping postprocessing")
            # create an empty result file (needed for pipeline)
            postprocessing_output_folder.mkdir()
            mock_array = np.array([], dtype=bool)
            np.save(postprocessing_output_folder / "placeholder.npy", mock_array)
            continue

        logging.info(f"\tCreating sorting analyzer")
        sorting_analyzer_full = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording_bin,
            sparse=True,
            return_scaled=postprocessing_params["return_scaled"],
            **sparsity_params
        )
        # compute templates for de-duplication
        # now postprocess
        analyzer_dict = postprocessing_params.copy()
        analyzer_dict.pop("duplicate_threshold")
        analyzer_dict.pop("return_scaled")
        sorting_analyzer_full.compute("random_spikes", **analyzer_dict["random_spikes"])
        sorting_analyzer_full.compute("templates")
        # de-duplication
        sorting_deduplicated = sc.remove_redundant_units(
            sorting_analyzer_full, duplicate_threshold=postprocessing_params["duplicate_threshold"]
        )
        logging.info(
            f"\tNumber of original units: {len(sorting.unit_ids)} -- Number of units after de-duplication: {len(sorting_deduplicated.unit_ids)}"
        )
        n_duplicated = int(len(sorting.unit_ids) - len(sorting_deduplicated.unit_ids))
        postprocessing_notes += f"\n- Removed {n_duplicated} duplicated units.\n"
        deduplicated_unit_ids = sorting_deduplicated.unit_ids

        sorting_analyzer_dedup = sorting_analyzer_full.select_units(sorting_deduplicated.unit_ids)

        if recording_lazy is not None:
            recording = recording_lazy
            recording_tmp = recording_bin
        else:
            recording = recording_bin
            recording_tmp = None

        
        # Build the analyzer with the *persisted* recording as the canonical one
        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting_deduplicated,
            recording=recording_bin,                 # <- this is what the analyzer will remember
            format="binary_folder",
            folder=scratch_folder / "tmp_analyzer",  # temp workspace for compute
            sparse=True,
            return_scaled=postprocessing_params["return_scaled"],
            sparsity=sorting_analyzer_dedup.sparsity)
        
        # now compute all extensions
        logging.info(f"\tComputing all postprocessing extensions")
        sorting_analyzer.compute(analyzer_dict)


        logging.info("\tComputing quality metrics")
        qm = sorting_analyzer.compute(
            "quality_metrics",
            metric_names=quality_metrics_names,
            qm_params=quality_metrics_params
        )

        # save
        logging.info("\tSaving SortingAnalyzer to zarr")
        sorting_analyzer = sorting_analyzer.save_as(
            format="zarr",
            folder=postprocessing_output_folder
        )

        t_postprocessing_end = time.perf_counter()
        elapsed_time_postprocessing = np.round(t_postprocessing_end - t_postprocessing_start, 2)

        # save params in output
        postprocessing_params["recording_name"] = recording_name
        postprocessing_outputs = dict(duplicated_units=n_duplicated)
        postprocessing_process = DataProcess(
            name="Ephys postprocessing",
            software_version=VERSION,  # either release or git commit
            start_date_time=datetime_start_postprocessing,
            end_date_time=datetime_start_postprocessing + timedelta(seconds=np.floor(elapsed_time_postprocessing)),
            input_location=str(data_folder),
            output_location=str(results_folder),
            code_url=URL,
            parameters=postprocessing_params,
            outputs=postprocessing_outputs,
            notes=postprocessing_notes,
        )
        with open(postprocessing_output_process_json, "w") as f:
            f.write(postprocessing_process.model_dump_json(indent=3))

        # copy data_description and subject json
        if ecephys_session_folder is not None:
            metadata_json_files = [p for p in ecephys_session_folder.iterdir() if p.suffix == ".json"]
            for metadata_file in metadata_json_files:
                if "data_description" in metadata_file.name or "subject" in metadata_file.name:
                    shutil.copy(metadata_file, results_folder / f"postprocessing_{recording_name}_{metadata_file.name}")

    t_postprocessing_end_all = time.perf_counter()
    elapsed_time_postprocessing_all = np.round(t_postprocessing_end_all - t_postprocessing_start_all, 2)
    logging.info(f"POSTPROCESSING time: {elapsed_time_postprocessing_all}s")
