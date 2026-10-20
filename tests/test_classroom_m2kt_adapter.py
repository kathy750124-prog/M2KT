"""CPU smoke tests for the trainable classroom memory path."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classroom_m2kt_adapter import ClassroomM2KTAdapter


class ClassroomM2KTAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.adapter = ClassroomM2KTAdapter(
            hidden_size=32, d_model=16, num_memory_slots=4,
            num_heads=4, max_clips=2, dropout=0.0,
        )

    def test_shape_finite_and_unselected_tokens_unchanged(self):
        inputs = torch.randn(2, 5, 32)
        av_mask = torch.tensor([[False, True, True, False, False], [False] * 5])
        audio_mask = torch.tensor([[False, True, False, False, False], [False] * 5])
        video_mask = torch.tensor([[False, False, True, False, False], [False] * 5])
        clip_index = torch.tensor([[-1, 0, 1, -1, -1], [-1] * 5])

        output = self.adapter(inputs, av_mask, audio_mask, video_mask, clip_index)
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[~av_mask], inputs[~av_mask]))
        self.assertLess((output[av_mask] - inputs[av_mask]).abs().max().item(), 0.05)

    def test_memory_receives_gradient(self):
        inputs = torch.randn(1, 4, 32, requires_grad=True)
        mask = torch.tensor([[True, False, True, False]])
        output = self.adapter(inputs, mask)
        output.square().mean().backward()
        self.assertIsNotNone(self.adapter.memory_slots.grad)
        self.assertTrue(torch.isfinite(self.adapter.memory_slots.grad).all())
        self.assertGreater(self.adapter.memory_slots.grad.abs().sum().item(), 0)
        self.assertIsNotNone(inputs.grad)

    def test_no_audio_video_tokens_is_identity(self):
        inputs = torch.randn(2, 3, 32)
        output = self.adapter(inputs, torch.zeros(2, 3, dtype=torch.bool))
        self.assertTrue(torch.equal(output, inputs))


if __name__ == "__main__":
    unittest.main()
