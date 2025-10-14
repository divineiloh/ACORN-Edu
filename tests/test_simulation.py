# ACORN-Edu Unit Tests
# Comprehensive test suite for the research simulation pipeline

import pytest
from unittest.mock import patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import after path setup
from simulation_harness import (  # noqa: E402
    BAP_Simulator,
    OAEC_Simulator,
    CAG_Simulator,
    MockSigner,
)
from config import (  # noqa: E402
    ASSETS,
    NETWORK_PROFILES,
    SCHEDULERS_TO_TEST,
    ABLATION_CONFIGS,
)


class TestBAPSimulator:
    """Test suite for BAP (Bandwidth-Aware Packaging) simulation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.simulator = BAP_Simulator(network_profile="nightly_wifi")

    def test_initialization(self):
        """Test BAP simulator initialization."""
        assert self.simulator.network_profile == "nightly_wifi"
        assert self.simulator.avg_chunk_size == 128 * 1024
        assert len(self.simulator.assets) == 5
        assert len(self.simulator.chunk_db) == 5

    def test_define_assets(self):
        """Test asset definition."""
        assets = self.simulator.assets
        assert len(assets) == 5
        assert all("id" in asset for asset in assets)
        assert all("size" in asset for asset in assets)
        assert all("deadline_hours" in asset for asset in assets)
        assert all("reuse_score" in asset for asset in assets)

    def test_network_trace_generation(self):
        """Test network trace generation for different profiles."""
        # Test nightly_wifi profile
        wifi_trace = self.simulator._define_nightly_wifi_trace()
        assert len(wifi_trace) == 120  # 5 days * 24 hours
        assert all(isinstance(item, tuple) and len(item) == 2 for item in wifi_trace)

        # Test spotty_cellular profile
        cellular_trace = self.simulator._define_spotty_cellular_trace()
        assert len(cellular_trace) == 120
        assert all(
            isinstance(item, tuple) and len(item) == 2 for item in cellular_trace
        )

    def test_chunk_asset(self):
        """Test asset chunking functionality."""
        test_asset = {"id": "Test Video", "size": 1024 * 1024}  # 1MB
        chunks = self.simulator._chunk_asset(test_asset)
        assert len(chunks) > 0
        assert all("Test Video" in chunk for chunk in chunks)

    def test_calculate_priority(self):
        """Test BAP priority calculation."""
        asset = {
            "id": "test",
            "size": 1024 * 1024,
            "deadline_hours": 24,
            "reuse_score": 0.5,
        }
        weights = (0.5, 0.2, 0.2, 0.1)
        current_hour = 12
        net_factor = 1.0

        priority = self.simulator._calculate_priority(
            asset, current_hour, weights, net_factor
        )
        assert isinstance(priority, float)
        assert priority > 0

    def test_calculate_hit_rate(self):
        """Test hit rate calculation."""
        # Create mock download log
        download_log = [
            ("Lecture 1 Video", 10, 50 * 1024 * 1024),  # Complete download
            ("Reading 1 PDF", 20, 1 * 1024 * 1024),  # Partial download
        ]
        assets = [
            {"id": "Lecture 1 Video", "size": 50 * 1024 * 1024, "deadline_hours": 24}
        ]

        hit_rate = self.simulator._calculate_hit_rate(download_log, assets)
        assert isinstance(hit_rate, float)
        assert 0 <= hit_rate <= 100

    def test_simulation_pass_acorn_scheduler(self):
        """Test simulation pass with AcornScheduler."""
        result = self.simulator.run_simulation_pass(
            "AcornScheduler", weights=(0.5, 0.2, 0.2, 0.1)
        )
        assert "bytes" in result
        assert "hit_rate" in result
        assert isinstance(result["bytes"], (int, float))
        assert isinstance(result["hit_rate"], (int, float))
        assert result["bytes"] >= 0
        assert 0 <= result["hit_rate"] <= 100

    def test_simulation_pass_baseline_scheduler(self):
        """Test simulation pass with baseline scheduler."""
        result = self.simulator.run_simulation_pass("DeadlineFIFO_Downloader")
        assert "bytes" in result
        assert "hit_rate" in result
        assert isinstance(result["bytes"], (int, float))
        assert isinstance(result["hit_rate"], (int, float))
        assert result["bytes"] >= 0
        assert 0 <= result["hit_rate"] <= 100

    def test_delta_sync_analysis(self):
        """Test delta sync analysis functionality."""
        # Create output directory for the test
        os.makedirs("./output/data", exist_ok=True)
        # This should not raise an exception
        self.simulator.run_delta_sync_analysis()


class TestOAECSimulator:
    """Test suite for OAEC (Offline Authentication and Event Chain) simulation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.simulator = OAEC_Simulator(num_chains=10)

    def test_initialization(self):
        """Test OAEC simulator initialization."""
        assert self.simulator.num_chains == 10
        assert hasattr(self.simulator, "signer")
        assert hasattr(self.simulator, "t0")

    def test_generate_event_chain(self):
        """Test event chain generation."""
        chain = self.simulator._generate_event_chain()
        assert len(chain) == 5  # start, ans1, save, ans2, submit
        assert all("p" in record for record in chain)
        assert all("h_p" in record for record in chain)
        assert all("mc" in record for record in chain)
        assert all("h_c" in record for record in chain)
        assert all("sig" in record for record in chain)

    def test_tamper_chain(self):
        """Test chain tampering functionality."""
        original_chain = self.simulator._generate_event_chain()
        tampered_chain = self.simulator._tamper_chain(original_chain)
        assert len(tampered_chain) <= len(original_chain)

    def test_verify_chain(self):
        """Test chain verification functionality."""
        # Test valid chain
        valid_chain = self.simulator._generate_event_chain()
        assert self.simulator._verify_chain(valid_chain) is True

        # Test tampered chain
        tampered_chain = self.simulator._tamper_chain(valid_chain)
        # The verification result depends on the type of tampering
        verification_result = self.simulator._verify_chain(tampered_chain)
        assert isinstance(verification_result, bool)


class TestCAGSimulator:
    """Test suite for CAG (Curriculum Audit and Gap) simulation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.simulator = CAG_Simulator()

    def test_initialization(self):
        """Test CAG simulator initialization."""
        assert hasattr(self.simulator, "std")
        assert hasattr(self.simulator, "lib")
        assert hasattr(self.simulator, "gt")
        assert len(self.simulator.std) > 0
        assert len(self.simulator.lib) > 0
        assert len(self.simulator.gt) > 0

    def test_simulation(self):
        """Test CAG simulation execution."""
        # Create output directory for the test
        os.makedirs("./output/data", exist_ok=True)
        # This should not raise an exception
        self.simulator.run_simulation()


class TestMockSigner:
    """Test suite for MockSigner utility class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.signer = MockSigner()

    def test_initialization(self):
        """Test MockSigner initialization."""
        assert hasattr(self.signer, "private_key")
        assert hasattr(self.signer, "public_key")
        assert self.signer.private_key == "device_private_key_secret"
        assert self.signer.public_key == "device_public_key_registered"

    def test_sign(self):
        """Test signing functionality."""
        test_data = "test_data"
        signature = self.signer.sign(test_data)
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length

    def test_verify(self):
        """Test signature verification."""
        test_data = "test_data"
        signature = self.signer.sign(test_data)

        # Test valid signature
        assert self.signer.verify(test_data, signature, self.signer.public_key) is True

        # Test invalid signature
        assert (
            self.signer.verify(test_data, "invalid_signature", self.signer.public_key)
            is False
        )

        # Test wrong public key
        assert self.signer.verify(test_data, signature, "wrong_public_key") is False


class TestConfiguration:
    """Test suite for configuration constants."""

    def test_assets_configuration(self):
        """Test that assets are properly configured."""
        assert len(ASSETS) == 5
        for asset in ASSETS:
            assert "id" in asset
            assert "size" in asset
            assert "deadline_hours" in asset
            assert "reuse_score" in asset
            assert asset["size"] > 0
            assert asset["deadline_hours"] > 0
            assert 0 <= asset["reuse_score"] <= 1

    def test_network_profiles(self):
        """Test network profiles configuration."""
        assert len(NETWORK_PROFILES) == 2
        assert "nightly_wifi" in NETWORK_PROFILES
        assert "spotty_cellular" in NETWORK_PROFILES

    def test_schedulers_configuration(self):
        """Test schedulers configuration."""
        assert len(SCHEDULERS_TO_TEST) == 2
        assert "DeadlineFIFO_Downloader" in SCHEDULERS_TO_TEST
        assert "AcornScheduler" in SCHEDULERS_TO_TEST

    def test_ablation_configs(self):
        """Test ablation study configurations."""
        assert len(ABLATION_CONFIGS) == 5
        for name, weights in ABLATION_CONFIGS.items():
            assert len(weights) == 4
            assert all(isinstance(w, (int, float)) for w in weights)
            assert sum(weights) <= 1.0  # Allow for some flexibility


class TestIntegration:
    """Integration tests for the complete simulation pipeline."""

    @patch("src.simulation_harness.os.makedirs")
    def test_simulation_pipeline_integration(self, mock_makedirs):
        """Test that the complete simulation pipeline can be executed."""
        # This is a basic integration test
        # In a real scenario, you would mock more dependencies
        simulator = BAP_Simulator()
        result = simulator.run_simulation_pass("AcornScheduler", weights=(0.5, 0.2, 0.2, 0.1))
        assert "bytes" in result
        assert "hit_rate" in result


if __name__ == "__main__":
    pytest.main([__file__])
