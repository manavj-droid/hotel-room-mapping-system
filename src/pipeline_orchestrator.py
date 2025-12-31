import logging
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
from dataclasses import asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================================================
# CONFIGURATION
# ========================================================================

class PipelineConfig:
    """Configuration for the entire pipeline."""
    # Input paths
    HOTELBEDS_INPUT = Path("data/inputs/hotelbeds_hotels.json")
    AXISDATA_INPUT = Path("data/inputs/axisdata_hotels.json")

    # Room types catalog (for enrichment)
    ROOM_TYPES_INPUT = Path("data/inputs/hotelbeds_room_types.json")

    # Intermediate output paths
    ENRICHED_HOTELBEDS = Path("data/outputs/enriched_hotelbeds_hotels.json")
    NORMALIZED_HOTELBEDS = Path("data/outputs/hotelbed_norm_data.json")
    NORMALIZED_AXISDATA = Path("data/outputs/axisdata_norm_data.json")

    # Embeddings output paths (Sentence Transformers)
    EMBEDDINGS_DIR = Path("data/embeddings")
    HB_HOTEL_EMB = EMBEDDINGS_DIR / "hotelbeds_hotel_embeddings.json"
    HB_ROOM_EMB = EMBEDDINGS_DIR / "hotelbeds_room_embeddings.json"
    AD_HOTEL_EMB = EMBEDDINGS_DIR / "axisdata_hotel_embeddings.json"
    AD_ROOM_EMB = EMBEDDINGS_DIR / "axisdata_room_embeddings.json"

    # Google embeddings output paths
    GOOGLE_EMBEDDINGS_DIR = Path("data/google_embedings")
    GOOGLE_HB_HOTEL_EMB = GOOGLE_EMBEDDINGS_DIR / "hotelbeds_hotel_embeddings_google.json"
    GOOGLE_HB_ROOM_EMB = GOOGLE_EMBEDDINGS_DIR / "hotelbeds_room_embeddings_google.json"
    GOOGLE_AD_HOTEL_EMB = GOOGLE_EMBEDDINGS_DIR / "axisdata_hotel_embeddings_google.json"
    GOOGLE_AD_ROOM_EMB = GOOGLE_EMBEDDINGS_DIR / "axisdata_room_embeddings_google.json"

    # Final output paths
    MAPPED_OUTPUT_DIR = Path("data/mapped")
    FINAL_MAPPED_DATA = MAPPED_OUTPUT_DIR / "mapped_data.json"

    # Google output paths
    GOOGLE_MAPPED_DATA = Path("data/mapped/mapped_data_google.json")

    # Logs
    LOGS_DIR = Path("logs")

# ========================================================================
# PIPELINE ORCHESTRATOR
# ========================================================================

class PipelineOrchestrator:
    """Orchestrates the complete hotel room mapping pipeline."""
    def __init__(
        self,
        embedding_method: str = "sentence-transformers",
        skip_enrichment: bool = False,
        skip_normalization: bool = False,
        skip_embeddings: bool = False,
        output_dir: Optional[Path] = None
    ):
        self.config = PipelineConfig()
        self.embedding_method = embedding_method
        self.skip_enrichment = skip_enrichment
        self.skip_normalization = skip_normalization
        self.skip_embeddings = skip_embeddings

        if output_dir:
            self.config.FINAL_MAPPED_DATA = output_dir / "mapped_data.json"
            self.config.GOOGLE_MAPPED_DATA = output_dir / "mapped_data_google.json"

        self.stats = {
            "start_time": datetime.now(),
            "steps_completed": [],
            "steps_failed": [],
            "total_hotels_processed": 0,
            "total_rooms_matched": 0,
            "embedding_method": embedding_method
        }

        logger.info("=" * 80)
        logger.info("PIPELINE ORCHESTRATOR INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Embedding method: {embedding_method}")
        logger.info(f"Skip enrichment: {skip_enrichment}")
        logger.info(f"Skip normalization: {skip_normalization}")
        logger.info(f"Skip embeddings: {skip_embeddings}")

    def run_pipeline(self) -> bool:
        try:
            if not self._step_1_load_data():
                return False
            if not self.skip_enrichment:
                if not self._step_2_enrich_data():
                    logger.warning("Enrichment failed, continuing with raw data...")
            if not self.skip_normalization:
                if not self._step_3_normalize_data():
                    return False
            if not self.skip_embeddings:
                if not self._step_4_generate_embeddings():
                    return False
            if not self._step_5_match_hotels_rooms():
                return False
            self._step_6_generate_report()
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            return True
        except Exception as e:
            logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
            return False

    def _step_1_load_data(self) -> bool:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: LOADING RAW DATA")
        logger.info("=" * 80)
        try:
            from data_loader import load_all_data
            hotelbeds_data, axisdata_data, _ = load_all_data(
                hotelbeds_path=self.config.HOTELBEDS_INPUT,
                axisdata_path=self.config.AXISDATA_INPUT,
                strict_mode=False
            )
            logger.info(f"✓ Loaded {len(hotelbeds_data)} Hotelbeds hotels")
            logger.info(f"✓ Loaded {len(axisdata_data)} AxisData hotels")
            self.stats["steps_completed"].append("load_data")
            self.stats["total_hotels_processed"] = len(hotelbeds_data) + len(axisdata_data)
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.stats["steps_failed"].append("load_data")
            return False

    def _step_2_enrich_data(self) -> bool:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: ENRICHING WITH ROOM TYPE DESCRIPTIONS")
        logger.info("=" * 80)
        try:
            from room_enricher import RoomEnricher
            if self.config.ENRICHED_HOTELBEDS.exists():
                logger.info(f"✓ Enriched data already exists: {self.config.ENRICHED_HOTELBEDS}")
                self.stats["steps_completed"].append("enrich_data_skipped")
                return True
            if not self.config.ROOM_TYPES_INPUT.exists():
                logger.warning(f"⚠ Room types catalog not found: {self.config.ROOM_TYPES_INPUT}")
                logger.warning("  Skipping enrichment step...")
                self.stats["steps_completed"].append("enrich_data_skipped")
                return True
            from data_loader import load_all_data
            hotelbeds_data, _, _ = load_all_data(
                hotelbeds_path=self.config.HOTELBEDS_INPUT,
                axisdata_path=self.config.AXISDATA_INPUT,
                strict_mode=False
            )
            enricher = RoomEnricher(self.config.ROOM_TYPES_INPUT)
            enriched_hotels, errors = enricher.enrich_hotelbeds_hotels(hotelbeds_data)
            self.config.ENRICHED_HOTELBEDS.parent.mkdir(parents=True, exist_ok=True)
            # Save as dict, not as __dict__ or as string
            enriched_hotels_data = [asdict(hotel) if hasattr(hotel, '__dataclass_fields__') else hotel for hotel in enriched_hotels]
            with open(self.config.ENRICHED_HOTELBEDS, 'w') as f:
                json.dump(enriched_hotels_data, f, indent=2, default=str)
            logger.info(f"✓ Enriched {len(enriched_hotels)} Hotelbeds hotels")
            logger.info(f"✓ Enriched data saved to {self.config.ENRICHED_HOTELBEDS}")
            if errors:
                logger.warning(f"⚠ Enrichment completed with {len(errors)} errors")
            self.stats["steps_completed"].append("enrich_data")
            return True
        except Exception as e:
            logger.error(f"Failed to enrich data: {e}")
            self.stats["steps_failed"].append("enrich_data")
            return True

    def _step_3_normalize_data(self) -> bool:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: NORMALIZING DATA")
        logger.info("=" * 80)
        try:
            from data_normalizer import DataNormalizer
            from data_loader import load_all_data
            if self.config.NORMALIZED_HOTELBEDS.exists() and self.config.NORMALIZED_AXISDATA.exists():
                logger.info(f"✓ Normalized data already exists")
                self.stats["steps_completed"].append("normalize_data_skipped")
                return True
            hotelbeds_input = self.config.ENRICHED_HOTELBEDS if self.config.ENRICHED_HOTELBEDS.exists() else self.config.HOTELBEDS_INPUT
            hotelbeds_data, axisdata_data, _ = load_all_data(
                hotelbeds_path=hotelbeds_input,
                axisdata_path=self.config.AXISDATA_INPUT,
                strict_mode=False
            )
            normalizer = DataNormalizer()
            logger.info("Normalizing Hotelbeds data...")
            hb_normalized, hb_stats = normalizer.normalize_hotelbeds_hotels(hotelbeds_data)
            logger.info("Normalizing AxisData data...")
            ad_normalized, ad_stats = normalizer.normalize_axisdata_hotels(axisdata_data)
            self.config.NORMALIZED_HOTELBEDS.parent.mkdir(parents=True, exist_ok=True)
            # Use asdict for dataclasses so as to output full nested dicts
            hb_dicts = [asdict(h) if hasattr(h, '__dataclass_fields__') else h for h in hb_normalized]
            ad_dicts = [asdict(h) if hasattr(h, '__dataclass_fields__') else h for h in ad_normalized]
            with open(self.config.NORMALIZED_HOTELBEDS, 'w') as f:
                json.dump(hb_dicts, f, indent=2, default=str)
            with open(self.config.NORMALIZED_AXISDATA, 'w') as f:
                json.dump(ad_dicts, f, indent=2, default=str)
            logger.info(f"✓ Normalized Hotelbeds data saved to {self.config.NORMALIZED_HOTELBEDS}")
            logger.info(f"✓ Normalized AxisData data saved to {self.config.NORMALIZED_AXISDATA}")
            self.stats["steps_completed"].append("normalize_data")
            return True
        except Exception as e:
            logger.error(f"Failed to normalize data: {e}", exc_info=True)
            self.stats["steps_failed"].append("normalize_data")
            return False

    def _step_4_generate_embeddings(self) -> bool:
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP 4: GENERATING EMBEDDINGS ({self.embedding_method.upper()})")
        logger.info("=" * 80)
        try:
            if self.embedding_method == "sentence-transformers":
                return self._generate_sentence_transformer_embeddings()
            elif self.embedding_method == "google":
                return self._generate_google_embeddings()
            else:
                logger.error(f"Unknown embedding method: {self.embedding_method}")
                return False
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            self.stats["steps_failed"].append("generate_embeddings")
            return False

    def _generate_sentence_transformer_embeddings(self) -> bool:
        if all([
            self.config.HB_HOTEL_EMB.exists(),
            self.config.HB_ROOM_EMB.exists(),
            self.config.AD_HOTEL_EMB.exists(),
            self.config.AD_ROOM_EMB.exists()
        ]):
            logger.info("✓ Sentence-transformer embeddings already exist")
            self.stats["steps_completed"].append("generate_embeddings_skipped")
            return True
        logger.info("Running embedding_service.py...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "embedding_service.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✓ Sentence-transformer embeddings generated successfully")
            self.stats["steps_completed"].append("generate_embeddings_st")
            return True
        else:
            logger.error(f"embedding_service.py failed: {result.stderr}")
            return False

    def _generate_google_embeddings(self) -> bool:
        if all([
            self.config.GOOGLE_HB_HOTEL_EMB.exists(),
            self.config.GOOGLE_HB_ROOM_EMB.exists(),
            self.config.GOOGLE_AD_HOTEL_EMB.exists(),
            self.config.GOOGLE_AD_ROOM_EMB.exists()
        ]):
            logger.info("✓ Google embeddings already exist")
            self.stats["steps_completed"].append("generate_embeddings_skipped")
            return True
        logger.info("Running google_embeddings.py...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "google_embeddings.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✓ Google embeddings generated successfully")
            self.stats["steps_completed"].append("generate_embeddings_google")
            return True
        else:
            logger.error(f"google_embeddings.py failed: {result.stderr}")
            return False

    def _step_5_match_hotels_rooms(self) -> bool:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: MATCHING HOTELS AND ROOMS")
        logger.info("=" * 80)
        try:
            # Choose correct matcher
            if self.embedding_method == "sentence-transformers":
                matcher_script = "hotel_matcher.py"
                output_file = self.config.FINAL_MAPPED_DATA
            elif self.embedding_method == "google":
                matcher_script = "hotel_matcher_using_google_embedings.py"
                output_file = self.config.GOOGLE_MAPPED_DATA
            else:
                logger.error(f"Unknown embedding method: {self.embedding_method}")
                return False
            logger.info(f"Running {matcher_script}...")
            import subprocess
            result = subprocess.run(
                [sys.executable, matcher_script],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"✓ Hotels and rooms matched successfully using {matcher_script}")
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        mapped_data = json.load(f)
                        self.stats["total_rooms_matched"] = mapped_data.get("summary", {}).get("total_room_matches", 0)
                        self.stats["output_file"] = str(output_file)
                self.stats["steps_completed"].append(f"match_hotels_rooms_{self.embedding_method}")
                return True
            else:
                logger.error(f"{matcher_script} failed: {result.stderr}")
                self.stats["steps_failed"].append(f"match_hotels_rooms_{self.embedding_method}")
                return False
        except Exception as e:
            logger.error(f"Failed to match hotels and rooms: {e}", exc_info=True)
            self.stats["steps_failed"].append("match_hotels_rooms")
            return False

    def _step_6_generate_report(self):
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: GENERATING FINAL REPORT")
        logger.info("=" * 80)
        self.stats["end_time"] = datetime.now()
        self.stats["total_duration"] = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        output_file = self.config.GOOGLE_MAPPED_DATA if self.embedding_method == "google" else self.config.FINAL_MAPPED_DATA
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"\nStart time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {self.stats['total_duration']:.2f} seconds")
        print(f"\nEmbedding method: {self.embedding_method}")
        print(f"Total hotels processed: {self.stats['total_hotels_processed']}")
        print(f"Total rooms matched: {self.stats['total_rooms_matched']}")
        print(f"\nSteps completed: {len(self.stats['steps_completed'])}")
        for step in self.stats['steps_completed']:
            print(f"  ✓ {step}")
        if self.stats['steps_failed']:
            print(f"\nSteps failed: {len(self.stats['steps_failed'])}")
            for step in self.stats['steps_failed']:
                print(f"  ✗ {step}")
        print(f"\nFinal output: {output_file}")
        print("=" * 80)
        report_path = self.config.LOGS_DIR / f"pipeline_report_{self.embedding_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        logger.info(f"✓ Report saved to {report_path}")

# ========================================================================
# CLI INTERFACE
# ========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hotel Room Mapping Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with sentence-transformers (default)
  python pipeline_orchestrator.py

  # Run with Google embeddings
  python pipeline_orchestrator.py --embedding-method google

  # Skip enrichment (if already done)
  python pipeline_orchestrator.py --skip-enrichment

  # Skip embeddings generation (use existing)
  python pipeline_orchestrator.py --skip-embeddings

  # Quick run (skip all intermediate steps, only match)
  python pipeline_orchestrator.py --skip-enrichment --skip-normalization --skip-embeddings

  # Compare both methods (run twice)
  python pipeline_orchestrator.py --embedding-method sentence-transformers
  python pipeline_orchestrator.py --embedding-method google --skip-enrichment --skip-normalization
        """
    )
    parser.add_argument(
        "--embedding-method",
        choices=["sentence-transformers", "google"],
        default="sentence-transformers",
        help="Embedding generation method (default: sentence-transformers)"
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip Google Places enrichment step"
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip data normalization step (use existing normalized data)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embeddings generation step (use existing embeddings)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for final mapped data"
    )
    return parser.parse_args()

# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    args = parse_args()
    logger.info("=" * 80)
    logger.info("HOTEL ROOM MAPPING PIPELINE")
    logger.info("=" * 80)
    orchestrator = PipelineOrchestrator(
        embedding_method=args.embedding_method,
        skip_enrichment=args.skip_enrichment,
        skip_normalization=args.skip_normalization,
        skip_embeddings=args.skip_embeddings,
        output_dir=args.output_dir
    )
    success = orchestrator.run_pipeline()
    if success:
        logger.info("\n✓ PIPELINE COMPLETED SUCCESSFULLY!")
        return 0
    else:
        logger.error("\n✗ PIPELINE FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
