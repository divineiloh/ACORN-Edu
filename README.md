# ACORN-Edu: Research Simulation Pipeline

A gold-standard research simulation system for educational technology, implementing deadline-aware prefetch and tamper-evident offline exams for low-connectivity learning environments.

## Overview

This project provides a complete, runnable simulation of the core components described in the paper "Deadline-Aware Prefetch and Tamper-Evident Offline Exams for Low-Connectivity Learning." The system includes:

- **BAP (Bandwidth-Aware Packaging)**: Intelligent content prefetching with deadline awareness
- **OAEC (Offline Authentication and Event Chain)**: Tamper-evident exam authentication
- **CAG (Curriculum Audit and Gap)**: Automated curriculum coverage analysis

## Quick Start

### Prerequisites

- Python 3.7+
- Windows PowerShell (for Makefile commands)

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
   ```bash
   venv\Scripts\activate
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
acorn-edu-simulation/
├── .github/
│   └── workflows/
│       └── ci.yml              # Continuous Integration
├── .flake8                     # Linting configuration
├── Makefile                    # Automation commands
├── pyproject.toml             # Black formatter configuration
├── README.md                   # This file
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── simulation_harness.py   # Core simulation logic
│   ├── analyze_results.py      # Results analysis
│   └── config.py               # Centralized configuration
└── tests/
    └── test_simulation.py      # Unit tests
```

## Configuration

All simulation parameters are centralized in `src/config.py`:

- **Simulation Parameters**: Number of runs, random seeds
- **Asset Definitions**: Course content with deadlines and reuse scores
- **Network Profiles**: Wi-Fi and cellular connectivity patterns
- **Scheduler Configurations**: BAP algorithm parameters
- **Ablation Study**: Component importance analysis

## Output

The simulation generates:

- **Data Files**: CSV results with statistical analysis
- **Figures**: Performance comparison charts
- **Metadata**: Run parameters and timestamps

Output is saved to `./output/data/` and `./output/figures/`.

## Development

### Code Quality

The project enforces high code quality standards:

- **Formatting**: Black code formatter
- **Linting**: Flake8 with custom rules
- **Testing**: Comprehensive unit test suite
- **CI/CD**: Automated GitHub Actions workflow

### Adding New Features

1. Update configuration in `src/config.py`
2. Implement logic in `src/simulation_harness.py`
3. Add tests in `tests/test_simulation.py`
4. Update documentation

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
venv\Scripts\activate && python -m pytest tests/ -v --cov=src
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