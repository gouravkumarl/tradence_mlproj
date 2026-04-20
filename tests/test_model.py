import unittest
import torch
from src.pruning import PrunableLinear, SelfPruningNN
from src.model import create_model


class TestPrunableLinear(unittest.TestCase):
    """Tests for custom PrunableLinear layer."""

    def setUp(self):
        self.layer = PrunableLinear(10, 5)

    def test_layer_initialization(self):
        """Test that PrunableLinear initializes correctly."""
        self.assertIsNotNone(self.layer.weight)
        self.assertIsNotNone(self.layer.bias)
        self.assertIsNotNone(self.layer.gate_scores)

    def test_forward_pass(self):
        """Test forward pass through PrunableLinear."""
        x = torch.randn(2, 10)
        output = self.layer(x)
        self.assertEqual(output.shape, (2, 5))

    def test_gates_sigmoid(self):
        """Test that gates are properly sigmoid-activated."""
        gates = self.layer.gates
        self.assertTrue((gates >= 0).all() and (gates <= 1).all())

    def test_sparsity_calculation(self):
        """Test sparsity calculation."""
        sparsity = self.layer.get_sparsity(threshold=1e-2)
        self.assertTrue(0 <= sparsity <= 100)

    def test_gate_loss(self):
        """Test gate loss computation."""
        loss = self.layer.get_gate_loss()
        self.assertTrue(loss.item() > 0)


class TestSelfPruningNN(unittest.TestCase):
    """Tests for SelfPruningNN model."""

    def setUp(self):
        self.model = SelfPruningNN(input_size=10, hidden_size=8, output_size=2)

    def test_model_initialization(self):
        """Test model initializes with correct layer structure."""
        self.assertIsNotNone(self.model.layer1)
        self.assertIsNotNone(self.model.layer2)
        self.assertIsInstance(self.model.layer1, PrunableLinear)
        self.assertIsInstance(self.model.layer2, PrunableLinear)

    def test_forward_pass(self):
        """Test forward pass through full model."""
        x = torch.randn(4, 10)
        output = self.model(x)
        self.assertEqual(output.shape, (4, 2))

    def test_sparsity_loss(self):
        """Test sparsity loss computation."""
        loss = self.model.get_sparsity_loss()
        self.assertTrue(loss.item() > 0)

    def test_average_sparsity(self):
        """Test average sparsity calculation."""
        sparsity = self.model.get_sparsity()
        self.assertTrue(0 <= sparsity <= 100)

    def test_network_state(self):
        """Test network state reporting."""
        state = self.model.get_network_state()
        self.assertIn('layer1_sparsity', state)
        self.assertIn('layer2_sparsity', state)
        self.assertIn('average_sparsity', state)
        self.assertIn('total_parameters', state)


class TestModelFactory(unittest.TestCase):
    """Tests for model creation factory."""

    def test_create_model(self):
        """Test that create_model returns SelfPruningNN."""
        model = create_model(input_size=10, hidden_size=8, output_size=2)
        self.assertIsInstance(model, SelfPruningNN)

    def test_model_output_shape(self):
        """Test output shape matches expected dimensions."""
        model = create_model(input_size=10, hidden_size=8, output_size=2)
        x = torch.randn(2, 10)
        output = model(x)
        self.assertEqual(output.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()