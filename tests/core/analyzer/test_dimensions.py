import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from coreml_converter.core.models import BaseArchitecture, DimensionResult
from coreml_converter.core.analyzer.dimensions import validate_lora_dimensions

class TestValidateloraDimensions:
    @patch("coreml_converter.core.analyzer.dimensions.safe_open")
    def test_sd15_compatible_lora(self, mock_safe_open):
        mock_ctx = MagicMock()
        mock_ctx.keys.return_value = ["lora_unet_attn1_to_q.lora_down.weight"]
        mock_ctx.get_tensor.return_value = MagicMock(shape=(4, 768))
        mock_safe_open.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_safe_open.return_value.__exit__ = MagicMock(return_value=False)
        result = validate_lora_dimensions(Path("/fake/lora.safetensors"), BaseArchitecture.SD15)
        assert result.compatible is True

    @patch("coreml_converter.core.analyzer.dimensions.safe_open")
    def test_sd20_incompatible_lora(self, mock_safe_open):
        mock_ctx = MagicMock()
        mock_ctx.keys.return_value = ["lora_unet_attn1_to_q.lora_down.weight"]
        mock_ctx.get_tensor.return_value = MagicMock(shape=(4, 768))
        mock_safe_open.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_safe_open.return_value.__exit__ = MagicMock(return_value=False)
        result = validate_lora_dimensions(Path("/fake/lora.safetensors"), BaseArchitecture.SD20)
        assert result.compatible is False
        assert result.expected == 1024
        assert result.actual == 768
