"""
Runner script for the data processing pipeline.

Usage:
    python scripts/run_data_pipeline.py [--dev|--full] [--validate-only]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Ensure legacy Keras before any TF import
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    MOVIELENS_DIR, PROCESSED_DATA_DIR, DEV_MODE, DEV_SAMPLE_SIZE,
    DEV_RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SPARK_APP_NAME, SPARK_DRIVER_MEMORY, SPARK_LOCAL_DIR,
)
from src.data.validation import validate_dataset
from src.data.processing import create_spark_session, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="MovieLens data processing pipeline")
    parser.add_argument("--dev", action="store_true", default=None,
                        help="Run in development mode (sample ~1M ratings)")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Run in full-data mode (all 25M ratings)")
    parser.add_argument("--validate-only", action="store_true", default=False,
                        help="Only validate the dataset, don't process")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Override dev mode sample size")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override MovieLens data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Determine mode
    if args.full:
        dev_mode = False
    elif args.dev is not None:
        dev_mode = True
    else:
        dev_mode = DEV_MODE

    data_dir = Path(args.data_dir) if args.data_dir else MOVIELENS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR
    sample_size = args.sample_size if args.sample_size else DEV_SAMPLE_SIZE

    logger.info("=" * 60)
    logger.info("MovieLens Data Processing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data directory:  {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mode:            {'DEV' if dev_mode else 'FULL'}")
    if dev_mode:
        logger.info(f"Sample size:     {sample_size:,}")
    logger.info("=" * 60)

    # Create Spark session
    spark = create_spark_session(
        app_name=SPARK_APP_NAME,
        driver_memory=SPARK_DRIVER_MEMORY,
        local_dir=SPARK_LOCAL_DIR,
    )

    try:
        # Validate
        logger.info("Validating dataset...")
        val_result = validate_dataset(data_dir, spark=spark)
        print("\n" + val_result.summary() + "\n")

        if not val_result.passed:
            logger.error("Validation FAILED. Fix errors before processing.")
            sys.exit(1)

        if args.validate_only:
            logger.info("Validation-only mode. Exiting.")
            return

        # Run pipeline
        result = run_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            dev_mode=dev_mode,
            dev_sample_size=sample_size,
            dev_seed=DEV_RANDOM_SEED,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            spark=spark,
        )

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE RESULTS")
        logger.info("=" * 60)
        logger.info(f"Status:        {result['status']}")
        logger.info(f"Mode:          {result['dataset_mode']}")
        logger.info(f"Total ratings: {result['total_ratings']:,}")
        logger.info(f"Used ratings:  {result['sampled_ratings']:,}")
        logger.info(f"Duration:      {result['duration_sec']:.1f}s")
        
        sm = result["split_metadata"]
        logger.info(f"Train:         {sm['train_count']:,} ({sm['train_actual_ratio']:.1%})")
        logger.info(f"Validation:    {sm['val_count']:,} ({sm['val_actual_ratio']:.1%})")
        logger.info(f"Test:          {sm['test_count']:,} ({sm['test_actual_ratio']:.1%})")
        logger.info(f"Train period:  {sm['train_start_date']} → {sm['train_end_date']}")
        logger.info(f"Val period:    {sm['val_start_date']} → {sm['val_end_date']}")
        logger.info(f"Test period:   {sm['test_start_date']} → {sm['test_end_date']}")
        logger.info(f"Output:        {result['output_dir']}")
        logger.info("=" * 60)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
