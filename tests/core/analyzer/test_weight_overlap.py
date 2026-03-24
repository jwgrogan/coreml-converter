import pytest
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path
from coreml_converter.core.models import Conflict, Severity
from coreml_converter.core.analyzer.weight_overlap import detect_weight_overlap


class TestDetectWeightOverlap:
    @patch("coreml_converter.core.analyzer.weight_overlap.safe_open")
    @patch("coreml_converter.core.analyzer.weight_overlap.torch")
    def test_no_overlap(self, mock_torch, mock_safe_open):
        # LoRA A modifies layer 1, LoRA B modifies layer 2
        tensor_a = torch.randn(4, 320)
        tensor_b = torch.randn(4, 320)
        mock_torch.norm.side_effect = lambda t, **kw: torch.norm(t.float())

        mock_a = MagicMock()
        mock_a.keys.return_value = ["layer1.lora_down.weight"]
        mock_a.get_tensor.return_value = tensor_a

        mock_b = MagicMock()
        mock_b.keys.return_value = ["layer2.lora_down.weight"]
        mock_b.get_tensor.return_value = tensor_b

        mock_safe_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mock_a), __exit__=MagicMock(return_value=False)),
            MagicMock(__enter__=MagicMock(return_value=mock_b), __exit__=MagicMock(return_value=False)),
        ]

        conflicts = detect_weight_overlap(
            [("LoRA A", Path("/a.safetensors")), ("LoRA B", Path("/b.safetensors"))]
        )
        assert len(conflicts) == 0

    @patch("coreml_converter.core.analyzer.weight_overlap.safe_open")
    @patch("coreml_converter.core.analyzer.weight_overlap.torch")
    def test_high_overlap(self, mock_torch, mock_safe_open):
        # Both LoRAs modify the same layers
        keys = ["layer1.lora_down.weight", "layer1.lora_up.weight"]
        tensor = torch.randn(4, 320)

        mock_torch.norm.side_effect = lambda t, **kw: torch.norm(t.float())

        mock_a = MagicMock()
        mock_a.keys.return_value = keys
        mock_a.get_tensor.return_value = tensor

        mock_b = MagicMock()
        mock_b.keys.return_value = keys
        mock_b.get_tensor.return_value = tensor

        mock_safe_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mock_a), __exit__=MagicMock(return_value=False)),
            MagicMock(__enter__=MagicMock(return_value=mock_b), __exit__=MagicMock(return_value=False)),
        ]

        conflicts = detect_weight_overlap(
            [("LoRA A", Path("/a.safetensors")), ("LoRA B", Path("/b.safetensors"))]
        )
        assert len(conflicts) >= 1
        assert conflicts[0].severity == Severity.WARNING

    def test_single_lora_no_conflicts(self):
        conflicts = detect_weight_overlap([("LoRA A", Path("/a.safetensors"))])
        assert len(conflicts) == 0

    def test_empty_loras_no_conflicts(self):
        conflicts = detect_weight_overlap([])
        assert len(conflicts) == 0
