from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chi_2023_variance" / "src"))

from model import ModelConfig, NoPEModel


class NoPEModelTests(unittest.TestCase):
    def test_batch_first_forward_and_backward_have_finite_values(self):
        torch.manual_seed(7)
        config = ModelConfig(device="cpu", hidden_dim=16, num_heads=4, seq_length=8)
        model = NoPEModel(config)
        inputs = torch.randn(2, 8, 16, requires_grad=True)

        output = model(inputs)

        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_future_perturbation_cannot_change_prefix_outputs(self):
        torch.manual_seed(11)
        model = NoPEModel(ModelConfig(hidden_dim=16, num_heads=4)).eval()
        inputs = torch.randn(2, 8, 16)
        changed = inputs.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10

        with torch.no_grad():
            original_output = model(inputs)
            changed_output = model(changed)

        torch.testing.assert_close(
            original_output[:, :4], changed_output[:, :4], atol=1e-6, rtol=1e-5
        )
        self.assertFalse(torch.allclose(original_output[:, 4:], changed_output[:, 4:]))

    def test_unmasked_control_responds_to_future_perturbation(self):
        torch.manual_seed(11)
        model = NoPEModel(ModelConfig(hidden_dim=16, num_heads=4)).eval()
        inputs = torch.randn(2, 8, 16)
        changed = inputs.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10

        with torch.no_grad():
            original_output = model(inputs, causal=False)
            changed_output = model(changed, causal=False)

        self.assertGreater(
            (original_output[:, :4] - changed_output[:, :4]).abs().max().item(),
            1e-5,
        )

    def test_forward_requires_batch_sequence_and_hidden_axes(self):
        model = NoPEModel(ModelConfig(hidden_dim=16, num_heads=4))
        with self.assertRaisesRegex(ValueError, r"\[B, T, D\]"):
            model(torch.randn(8, 16))


if __name__ == "__main__":
    unittest.main()
