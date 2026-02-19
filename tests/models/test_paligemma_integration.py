
import pytest
import torch
from unittest.mock import MagicMock, patch
from kornia.models.paligemma import PaliGemma, PaliGemmaConfig

class TestPaliGemmaIntegration:
    @patch('transformers.PaliGemmaForConditionalGeneration.from_pretrained')
    def test_weight_mapping(self, mock_from_pretrained):
        # Create a tiny mock configuration
        mock_hf_model = MagicMock()
        mock_hf_model.config.text_config.vocab_size = 100
        mock_hf_model.config.text_config.hidden_size = 16
        mock_hf_model.config.text_config.num_hidden_layers = 1
        mock_hf_model.config.text_config.num_attention_heads = 2
        mock_hf_model.config.text_config.head_dim = 8
        mock_hf_model.config.text_config.intermediate_size = 32
        mock_hf_model.config.text_config.num_key_value_heads = 1
        
        mock_hf_model.config.vision_config.image_size = 32
        mock_hf_model.config.vision_config.patch_size = 16
        mock_hf_model.config.vision_config.hidden_size = 16
        mock_hf_model.config.vision_config.num_hidden_layers = 1
        mock_hf_model.config.vision_config.num_attention_heads = 2
        mock_hf_model.config.vision_config.intermediate_size = 32

        # Create a dummy state dict for HF model
        # We only populate a few keys to verify mapping
        hf_sd = {
            "model.embed_tokens.weight": torch.ones(100, 16),
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight": torch.ones(16, 3, 16, 16),
            "model.vision_tower.vision_model.post_layernorm.weight": torch.ones(16),
            "model.multi_modal_projector.linear.weight": torch.ones(64, 16), # projected features to hidden
            "lm_head.weight": torch.ones(100, 16),
        }
        mock_hf_model.state_dict.return_value = hf_sd
        mock_from_pretrained.return_value = mock_hf_model

        # Call Kornia's from_pretrained (this won't actually download anything since we mocked HF)
        model = PaliGemma.from_pretrained("mock-repo-id")

        # Verify that weights were correctly copied to Kornia keys
        # embed_tokens.weight -> model.embed_tokens.weight
        assert torch.allclose(model.embed_tokens.weight, hf_sd["model.embed_tokens.weight"])
        
        # vision_tower.embeddings.patch_embedding.weight -> model.vision_tower.vision_model.embeddings.patch_embedding.weight
        assert torch.allclose(
            model.vision_tower.embeddings.patch_embedding.weight, 
            hf_sd["model.vision_tower.vision_model.embeddings.patch_embedding.weight"]
        )
        
        # vision_tower.post_layernorm.weight -> model.vision_tower.vision_model.post_layernorm.weight
        assert torch.allclose(
            model.vision_tower.post_layernorm.weight, 
            hf_sd["model.vision_tower.vision_model.post_layernorm.weight"]
        )

    @patch('transformers.PaliGemmaForConditionalGeneration.from_pretrained')
    def test_gated_model_error(self, mock_from_pretrained):
        # Mock a 401 Unauthorized error from HF
        mock_from_pretrained.side_effect = Exception("401 Client Error: Unauthorized for url")

        with pytest.raises(RuntimeError) as excinfo:
            PaliGemma.from_pretrained("google/paligemma-3b-pt-224")
        
        assert "gated" in str(excinfo.value)
        assert "huggingface-cli login" in str(excinfo.value)
