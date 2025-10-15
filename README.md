# ACORN-Edu: Research Simulation Pipeline

A gold-standard research simulation system for educational technology, implementing deadline-aware prefetch and tamper-evident offline exams for low-connectivity learning environments.

## Overview

This project provides a complete, runnable simulation of the core components described in the paper "Deadline-Aware Prefetch and Tamper-Evident Offline Exams for Low-Connectivity Learning." The system includes:

- **BAP (Bandwidth-Aware Packaging)**: Intelligent content prefetching with deadline awareness
- **OAEC (Offline Assessment & Evidence Chain)**: Tamper-evident exam authentication
- **CAG (Curriculum Audit and Gap)**: Automated curriculum coverage analysis

## Quick Start

### Prerequisites

- Python 3.11+
- Cross-platform support (Windows PowerShell, Linux/macOS bash)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ACORN-Edu
   ```

2. **Set up the environment**:
   ```bash
   make setup
   ```

3. **Activate the virtual environment**:
   
   **POSIX (Linux/macOS)**:
   ```bash
   source .venv/bin/activate
   ```
   
   **Windows PowerShell**:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

### Running the Pipeline

The project uses a Makefile-based workflow for automation:

```bash
# Run the complete research pipeline
make run

# Or run the full quality pipeline (format, lint, test, run)
make all
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make setup` | Create virtual environment and install dependencies |
| `make clean` | Delete output directory and Python cache files |
| `make format` | Format all Python code using black |
| `make lint` | Lint all Python code using flake8 |
| `make test` | Run all unit tests using pytest |
| `make run` | Execute the full research pipeline |
| `make all` | Run format, lint, test, and run in sequence |
| `make help` | Show help information |

## Project Structure

```
ACORN-Edu/
├── .github/
│   └── workflows/
│       └── ci.yml              # Continuous Integration
├── .flake8                     # Linting configuration
├── Makefile                    # Automation commands
├── pyproject.toml             # Black formatter configuration
├── README.md                   # This file
├── requirements.txt           # Python dependencies
├── simulation_harness.py      # Core simulation logic
├── scripts/
│   └── create_plots.py        # Plot generation
├── tests/
│   ├── test_pipeline.py       # Pipeline tests
│   └── test_oaec.py          # OAEC tests
├── data/                      # Output data (CSV files)
└── figures/                   # Output figures (PNG files)
```

## Configuration

All simulation parameters are defined in `simulation_harness.py`:

- **Simulation Parameters**: N=30 trials, random seeds, 95% Student-t confidence intervals
- **Asset Definitions**: Course content with deadlines and reuse scores
- **Network Profiles**: Wi-Fi and cellular connectivity patterns
- **Scheduler Configurations**: AcornScheduler algorithm parameters
- **Ablation Study**: Component importance analysis

## Output

The simulation generates:

- **Data Files**: CSV results with statistical analysis (KB units, 95% Student-t CIs)
- **Figures**: Performance comparison charts (300 DPI, ≥12pt fonts)
- **Metadata**: Run parameters and timestamps

### Statistics & Units

- **Units**: All size metrics in KB only (column names end with `_kb`)
- **Statistics**: N=30 trials with two-sided 95% Student-t confidence intervals
- **Output Location**: `data/` and `figures/` directories

### Expected Output Files

**Data Files:**
- `data/bap_network_scenario_results.csv` - Network scenario results
- `data/bap_ablation_study_results.csv` - Ablation study results
- `data/oaec_confusion_matrix.csv` - OAEC tamper detection results
- `data/run_metadata.json` - Run parameters and version info

**Figures:**
- `figures/network_bytes.png` - Network bandwidth comparison
- `figures/network_hitrate.png` - Network hit rate comparison
- `figures/ablation_hitrate.png` - Ablation hit rate study
- `figures/ablation_bytes.png` - Ablation bandwidth study

## Development

### Code Quality

The project enforces high code quality standards:

- **Formatting**: Black code formatter
- **Linting**: Flake8 with custom rules
- **Testing**: Comprehensive unit test suite
- **CI/CD**: Automated GitHub Actions workflow

### Adding New Features

1. Update configuration in `simulation_harness.py`
2. Implement logic in `simulation_harness.py`
3. Add tests in `tests/`
4. Update documentation

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
# POSIX:
source .venv/bin/activate && python -m pytest tests/ -v --cov=.
# Windows PowerShell:
.venv\Scripts\Activate.ps1; python -m pytest tests/ -v --cov=.
```

## Research Applications

This simulation system is designed for:

- **Academic Research**: Generating empirical results for journal submissions
- **Performance Analysis**: Comparing scheduling algorithms under various network conditions
- **Sensitivity Analysis**: Understanding component importance through ablation studies
- **Reproducible Science**: Standardized experimental protocols

## Citation

If you use this simulation system in your research, please cite:

```
@article{acorn-edu-2024,
  title={Deadline-Aware Prefetch and Tamper-Evident Offline Exams for Low-Connectivity Learning},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the quality pipeline: `make all`
5. Submit a pull request

## Support

For questions or issues, please open a GitHub issue or contact the research team.