import unittest
import json
import pathlib
import os
from io import StringIO
from unittest.mock import patch, mock_open
from src.benchmark.cost_manager import CostManager, CostMetrics
from src.benchmark import cost_manager  # Import to get the module's directory

class TestCostManagerReporting(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        # Load test config data
        with open(pathlib.Path(__file__).parent / "test_model_config.json") as f:
            self.test_config = json.load(f)
        
        # Compute the default config path that CostManager uses
        default_config_dir = os.path.dirname(cost_manager.__file__)
        self.default_config_path = os.path.join(default_config_dir, "config.json")
        
        # Mock os.path.exists to return True for the default_config_path
        self.os_path_exists_patcher = patch('os.path.exists')
        self.mock_os_path_exists = self.os_path_exists_patcher.start()
        original_exists = os.path.exists
        self.mock_os_path_exists.side_effect = lambda path: True if path == self.default_config_path else original_exists(path)
        
        # Mock the config file operations
        self.open_patcher = patch(
            'builtins.open',
            mock_open(read_data=json.dumps(self.test_config))
        )
        self.mock_open = self.open_patcher.start()

    def tearDown(self):
        self.os_path_exists_patcher.stop()
        self.open_patcher.stop()

    def create_manager_with_metrics(self, costs: dict):
        # Create manager with mocked config
        manager = CostManager()  # Config path is default, but mocks handle it
        manager.metrics = CostMetrics()
        manager.metrics.costs.update(costs)
        return manager


    def test_single_group_model_reporting(self):
        costs = {
            "deepseek_reasoner": 0.123456,
            "total": 0.123456
        }
        manager = self.create_manager_with_metrics(costs)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            manager.print_final_report()
            output = fake_out.getvalue()

        expected = """
  DeepSeek Reasoner: $0.1235
  Total Cost: $0.1235
"""
        self.assertIn(expected, output)

    def test_grouped_model_reporting(self):
        costs = {
            "claude_sonnet_standalone": 1.2345,
            "claude_sonnet_with_reasoning": 2.3456,
            "total": 3.5801
        }
        manager = self.create_manager_with_metrics(costs)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            manager.print_final_report()
            output = fake_out.getvalue()

        expected = """
  Claude Sonnet (Standalone): $1.2345
  Claude Sonnet (With Reasoning): $2.3456
  Total Cost: $3.5801
"""
        self.assertIn(expected, output)

    def test_mixed_model_reporting(self):
        costs = {
            "deepseek_reasoner": 0.5,
            "claude_sonnet_standalone": 1.0,
            "claude_sonnet_with_reasoning": 1.5,
            "total": 3.0
        }
        manager = self.create_manager_with_metrics(costs)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            manager.print_final_report()
            output = fake_out.getvalue()

        expected = """
  DeepSeek Reasoner: $0.5000
  Claude Sonnet (Standalone): $1.0000
  Claude Sonnet (With Reasoning): $1.5000
  Total Cost: $3.0000
"""
        self.assertIn(expected, output)

    def test_zero_cost_handling(self):
        costs = {
            "claude_sonnet_standalone": 0.0,
            "total": 0.0
        }
        manager = self.create_manager_with_metrics(costs)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            manager.print_final_report()
            output = fake_out.getvalue()

        # Should NOT display zero-cost entries
        self.assertNotIn("Claude Sonnet (Standalone)", output)
        self.assertIn("Total Cost: $0.0000", output)

    def test_formatting_consistency(self):
        costs = {
            "deepseek_reasoner": 0.123,
            "claude_sonnet_standalone": 1.234567,
            "total": 1.357567
        }
        manager = self.create_manager_with_metrics(costs)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            manager.print_final_report()
            output = fake_out.getvalue()

        # Verify all costs show exactly 4 decimal places
        self.assertIn("DeepSeek Reasoner: $0.1230", output)  # Padded zero
        self.assertIn("Claude Sonnet (Standalone): $1.2346", output)  # Rounded
        self.assertIn("Total Cost: $1.3576", output)

    def test_missing_group_handling(self):
        costs = {
            "claude_sonnet_standalone": 1.0,
            "total": 1.0
        }
        manager = self.create_manager_with_metrics(costs)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            manager.print_final_report()
            output = fake_out.getvalue()

        # Should only show existing groups
        self.assertIn("Claude Sonnet (Standalone): $1.0000", output)
        self.assertNotIn("With Reasoning", output)

if __name__ == '__main__':
    unittest.main()