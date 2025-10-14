# ACORN-Edu Configuration
# Centralized configuration for all simulation parameters and constants

# Simulation Parameters
NUM_SIMULATION_RUNS = 30
RANDOM_SEED_BASE = 42

# Output Directories
OUTPUT_DATA_DIR = "./output/data"
OUTPUT_FIGURES_DIR = "./output/figures"

# BAP (Bandwidth-Aware Packaging) Configuration
AVG_CHUNK_SIZE = 128 * 1024  # 128 KB

# Asset Definitions
ASSETS = [
    {
        "id": "Lecture 1 Video",
        "size": 50 * 1024 * 1024,
        "deadline_hours": 24,
        "reuse_score": 0.8,
    },
    {
        "id": "Reading 1 PDF",
        "size": 2 * 1024 * 1024,
        "deadline_hours": 48,
        "reuse_score": 0.2,
    },
    {"id": "Quiz 1 Data", "size": 512 * 1024, "deadline_hours": 72, "reuse_score": 0.1},
    {
        "id": "Lecture 2 Video",
        "size": 65 * 1024 * 1024,
        "deadline_hours": 96,
        "reuse_score": 0.8,
    },
    {
        "id": "Project Spec PDF",
        "size": 5 * 1024 * 1024,
        "deadline_hours": 120,
        "reuse_score": 0.3,
    },
]

# Network Profiles
NETWORK_PROFILES = ["nightly_wifi", "spotty_cellular"]

# Scheduler Types
SCHEDULERS_TO_TEST = ["DeadlineFIFO_Downloader", "AcornScheduler"]

# Ablation Study Configurations
ABLATION_CONFIGS = {
    "Full ACORN Scheduler": (0.5, 0.2, 0.2, 0.1),
    "No Deadline (alpha=0)": (0.0, 0.3, 0.3, 0.1),
    "No Reuse (beta=0)": (0.6, 0.0, 0.3, 0.1),
    "No Size (gamma=0)": (0.6, 0.3, 0.0, 0.1),
    "No Network (delta=0)": (0.6, 0.2, 0.2, 0.0),
}

# Delta Sync Analysis Configuration
DELTA_MAP = {"Video": (0.10, 0.07), "PDF": (0.08, 0.05), "Data": (0.15, 0.10)}

# OAEC (Offline Authentication and Event Chain) Configuration
OAEC_NUM_CHAINS = 100
OAEC_TIMESTAMP_BASE = 1_725_000_000

# CAG (Curriculum Audit and Gap) Configuration
CAG_STANDARDS = {
    "CS101": {},
    "CS102": {"p": ["CS101"]},
    "CS201": {"p": ["CS102"]},
    "SE101": {},
    "DB101": {"p": ["CS101"]},
}

CAG_LIBRARY = {
    "M1": {"c": "CS101", "a": 1},
    "M2": {"c": "CS102", "a": 1},
    "M3": {"c": "CS201", "a": 1},
    "M4": {"c": "DB101", "a": 0},
}

CAG_GROUND_TRUTH = {"CS101", "CS102", "CS201"}

# Statistical Analysis
CONFIDENCE_LEVEL = 0.95

# Mock Signer Configuration
MOCK_PRIVATE_KEY = "device_private_key_secret"
MOCK_PUBLIC_KEY = "device_public_key_registered"
