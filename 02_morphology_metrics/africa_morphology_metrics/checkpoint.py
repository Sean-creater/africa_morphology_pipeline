"""
Checkpoint tracking for phased city processing.

Records which cities have completed each processing phase so a batch run
can be resumed after an interruption without reprocessing cities that
already succeeded. All writes are atomic (write to a temp file, then
rename) so a crash mid-write cannot corrupt the checkpoint file.

Checkpoint file format:
{
    "metadata": {
        "created_at": "...",
        "last_updated": "...",
        "total_cities_processed": 125,
        "current_phase": 2,
        "checkpoint_version": "1.0"
    },
    "phase1_completed": ["Lagos", "Accra", "Nairobi", ...],
    "phase2_completed": ["Apam", "Cape Coast", ...],
    "phase3_completed": [],
    "phase1_failed": [],
    "phase2_failed": [],
    "phase3_failed": [],
    "city_details": {
        "Lagos": {
            "phase": 1,
            "completed_at": "...",
            "iso3": "NGA",
            "status": "success"
        },
        ...
    }
}
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = "1.0"


class ProcessingCheckpoint:
    """Tracks and persists progress through phased city processing."""

    def __init__(self, checkpoint_file: Path):
        """
        Initialize the checkpoint manager.

        Parameters
        ----------
        checkpoint_file : Path
            Path to the checkpoint JSON file.
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.data = self._load_or_initialize()

    def _load_or_initialize(self) -> Dict:
        """Load an existing checkpoint, or initialize a new one."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                logger.info(f"✓ Loaded checkpoint: {self.checkpoint_file}")
                logger.info(f"  Total processed: {data['metadata']['total_cities_processed']}")
                logger.info(f"  Current phase: {data['metadata']['current_phase']}")

                if data['metadata'].get('checkpoint_version') != CHECKPOINT_VERSION:
                    logger.warning(f"Checkpoint version mismatch: {data['metadata'].get('checkpoint_version')} != {CHECKPOINT_VERSION}")

                return data

            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                logger.warning("Initializing new checkpoint...")
                return self._initialize_new_checkpoint()
        else:
            logger.info("No existing checkpoint found. Initializing new checkpoint...")
            return self._initialize_new_checkpoint()

    def _initialize_new_checkpoint(self) -> Dict:
        """Create a fresh checkpoint structure."""
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_cities_processed": 0,
                "current_phase": 1,
                "checkpoint_version": CHECKPOINT_VERSION
            },
            "phase1_completed": [],
            "phase2_completed": [],
            "phase3_completed": [],
            "phase1_failed": [],
            "phase2_failed": [],
            "phase3_failed": [],
            "city_details": {}
        }

    def save(self):
        """Save the checkpoint to disk via an atomic write-then-rename."""
        try:
            self.data['metadata']['last_updated'] = datetime.now().isoformat()
            self.data['metadata']['total_cities_processed'] = (
                len(self.data['phase1_completed']) +
                len(self.data['phase2_completed']) +
                len(self.data['phase3_completed'])
            )

            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)

            temp_file.replace(self.checkpoint_file)

            logger.debug(f"✓ Checkpoint saved: {self.checkpoint_file}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def mark_city_completed(self, city_name: str, phase: int, iso3: str):
        """
        Mark a city as successfully completed.

        Parameters
        ----------
        city_name : str
            Name of the city.
        phase : int
            Processing phase (1, 2, or 3).
        iso3 : str
            ISO3 country code.
        """
        phase_key = f"phase{phase}_completed"

        if city_name not in self.data[phase_key]:
            self.data[phase_key].append(city_name)

        self.data['city_details'][city_name] = {
            "phase": phase,
            "completed_at": datetime.now().isoformat(),
            "iso3": iso3,
            "status": "success"
        }

        self.data['metadata']['current_phase'] = phase

        self.save()

        logger.info(f"✓ Checkpoint: {city_name} (Phase {phase}) marked as completed")

    def mark_city_failed(self, city_name: str, phase: int, iso3: str, error_msg: str):
        """
        Mark a city as failed.

        Provided for logging/debugging; in the pipeline's strict processing
        mode a failure halts the batch rather than being silently recorded
        and skipped.

        Parameters
        ----------
        city_name : str
            Name of the city.
        phase : int
            Processing phase (1, 2, or 3).
        iso3 : str
            ISO3 country code.
        error_msg : str
            Error message.
        """
        phase_key = f"phase{phase}_failed"

        if city_name not in self.data[phase_key]:
            self.data[phase_key].append(city_name)

        self.data['city_details'][city_name] = {
            "phase": phase,
            "failed_at": datetime.now().isoformat(),
            "iso3": iso3,
            "status": "failed",
            "error": error_msg
        }

        self.save()

        logger.error(f"✗ Checkpoint: {city_name} (Phase {phase}) marked as FAILED: {error_msg}")

    def is_city_completed(self, city_name: str, phase: Optional[int] = None) -> bool:
        """
        Check whether a city has been completed.

        Parameters
        ----------
        city_name : str
            Name of the city.
        phase : int, optional
            Check a specific phase (if None, check all phases).

        Returns
        -------
        bool
            True if the city is completed.
        """
        if phase is not None:
            phase_key = f"phase{phase}_completed"
            return city_name in self.data[phase_key]
        else:
            return (
                city_name in self.data['phase1_completed'] or
                city_name in self.data['phase2_completed'] or
                city_name in self.data['phase3_completed']
            )

    def get_completed_cities(self, phase: Optional[int] = None) -> List[str]:
        """
        Get the list of completed cities.

        Parameters
        ----------
        phase : int, optional
            Get cities for a specific phase (if None, get all phases).

        Returns
        -------
        List[str]
            List of completed city names.
        """
        if phase is not None:
            phase_key = f"phase{phase}_completed"
            return self.data[phase_key].copy()
        else:
            return (
                self.data['phase1_completed'] +
                self.data['phase2_completed'] +
                self.data['phase3_completed']
            )

    def get_pending_cities(self, all_cities: List[str], phase: int) -> List[str]:
        """
        Get cities pending processing for a phase.

        Determined from each city's recorded status in ``city_details``
        rather than from the ``phaseX_completed`` lists alone, since a name
        with trailing whitespace differences would otherwise fail an exact
        list-membership check even though its status is recorded correctly.

        Parameters
        ----------
        all_cities : List[str]
            All cities that should be processed in this phase.
        phase : int
            Phase number.

        Returns
        -------
        List[str]
            List of pending city names.
        """
        pending = []
        for city in all_cities:
            if city in self.data['city_details']:
                status = self.data['city_details'][city].get('status', 'pending')
                # Anything not explicitly 'success' is pending, including
                # 'failed' cities (eligible for retry).
                if status != 'success':
                    pending.append(city)
            else:
                pending.append(city)

        return pending

    def get_city_status(self, city_name: str, phase: Optional[int] = None) -> Optional[str]:
        """
        Get the recorded status of a city.

        Parameters
        ----------
        city_name : str
            Name of the city.
        phase : int, optional
            Check a specific phase (if None, check all phases).

        Returns
        -------
        Optional[str]
            'success', 'failed', or None if not in the checkpoint.
        """
        if city_name in self.data['city_details']:
            return self.data['city_details'][city_name].get('status')
        return None

    def get_statistics(self) -> Dict:
        """
        Get checkpoint statistics.

        Returns
        -------
        Dict
            Statistics about processing progress.
        """
        return {
            "total_completed": len(self.get_completed_cities()),
            "phase1_completed": len(self.data['phase1_completed']),
            "phase2_completed": len(self.data['phase2_completed']),
            "phase3_completed": len(self.data['phase3_completed']),
            "phase1_failed": len(self.data['phase1_failed']),
            "phase2_failed": len(self.data['phase2_failed']),
            "phase3_failed": len(self.data['phase3_failed']),
            "current_phase": self.data['metadata']['current_phase'],
            "last_updated": self.data['metadata']['last_updated'],
        }

    def print_statistics(self):
        """Print checkpoint statistics to the console."""
        stats = self.get_statistics()

        print("\n" + "="*80)
        print("CHECKPOINT STATISTICS")
        print("="*80)
        print(f"Current Phase: {stats['current_phase']}")
        print(f"Last Updated: {stats['last_updated']}")
        print(f"\nTotal Completed: {stats['total_completed']}")
        print(f"\nPhase 1 (Sample Cities):")
        print(f"  Completed: {stats['phase1_completed']}")
        print(f"  Failed:    {stats['phase1_failed']}")
        print(f"\nPhase 2 (Ghana Remaining):")
        print(f"  Completed: {stats['phase2_completed']}")
        print(f"  Failed:    {stats['phase2_failed']}")
        print(f"\nPhase 3 (All Remaining):")
        print(f"  Completed: {stats['phase3_completed']}")
        print(f"  Failed:    {stats['phase3_failed']}")
        print("="*80)

    def reset(self):
        """
        Reset the checkpoint, clearing all recorded progress.

        WARNING: This deletes all checkpoint data.
        """
        logger.warning("Resetting checkpoint - all progress will be lost!")
        self.data = self._initialize_new_checkpoint()
        self.save()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_checkpoint_path(output_base: Path) -> Path:
    """
    Get the standard checkpoint file path for an output directory.

    Parameters
    ----------
    output_base : Path
        Base output directory.

    Returns
    -------
    Path
        Path to the checkpoint file, under a dedicated ``checkpoint/``
        subdirectory of ``output_base``.
    """
    checkpoint_dir = output_base / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / "processing_checkpoint.json"


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test checkpoint functionality."""
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*80)
    print("TESTING CHECKPOINT MANAGER")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_checkpoint_file = Path(f.name)

    try:
        print("\n[Test 1] Initializing checkpoint...")
        checkpoint = ProcessingCheckpoint(test_checkpoint_file)
        checkpoint.print_statistics()

        print("\n[Test 2] Marking cities completed...")
        checkpoint.mark_city_completed("Lagos", phase=1, iso3="NGA")
        checkpoint.mark_city_completed("Accra", phase=1, iso3="GHA")
        checkpoint.mark_city_completed("Nairobi", phase=1, iso3="KEN")

        print("\n[Test 3] Checking completion status...")
        assert checkpoint.is_city_completed("Lagos", phase=1), "Lagos should be completed"
        assert not checkpoint.is_city_completed("Apam", phase=1), "Apam should not be completed"
        print("✓ Completion checks working")

        print("\n[Test 4] Getting pending cities...")
        all_phase1_cities = ["Lagos", "Accra", "Nairobi", "Apam", "Kumasi"]
        pending = checkpoint.get_pending_cities(all_phase1_cities, phase=1)
        print(f"Pending cities: {pending}")
        assert "Apam" in pending, "Apam should be pending"
        assert "Lagos" not in pending, "Lagos should not be pending"
        print("✓ Pending city detection working")

        print("\n[Test 5] Getting statistics...")
        checkpoint.print_statistics()

        print("\n[Test 6] Testing persistence...")
        checkpoint2 = ProcessingCheckpoint(test_checkpoint_file)
        assert checkpoint2.is_city_completed("Lagos", phase=1), "Lagos should still be completed after reload"
        print("✓ Checkpoint persistence working")

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)

    finally:
        test_checkpoint_file.unlink(missing_ok=True)
