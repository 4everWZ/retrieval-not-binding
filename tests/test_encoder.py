import types
import unittest
from unittest import mock
try:
    import torch
except ImportError:
    torch = None
from retrieval_not_binding.encoder import Siglip2AdapterError, encode_single_global

class Siglip2AdapterTests(unittest.TestCase):

    def test_rejects_non_tensor_without_model_call(self):
        processor = mock.Mock(return_value={'pixel_values': 'not-a-tensor'})
        model = mock.Mock()
        with self.assertRaises(Siglip2AdapterError):
            encode_single_global(model=model, processor=processor, image=object(), device='cpu')
        model.get_image_features.assert_not_called()

    @unittest.skipUnless(torch is not None, 'torch is exercised in the remote target environment')
    def test_uses_v5_pooler_output_and_normalizes(self):
        processor = mock.Mock(return_value={'pixel_values': torch.ones((1, 1))})
        raw = torch.arange(1, 769, dtype=torch.float32).reshape(1, 768)
        model = mock.Mock()
        model.get_image_features.return_value = types.SimpleNamespace(pooler_output=raw)
        _, observed, normalized = encode_single_global(model=model, processor=processor, image=object(), device='cpu')
        self.assertTrue(torch.equal(observed, raw))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(normalized, dim=-1), torch.ones((1,))))

    @unittest.skipUnless(torch is not None, 'torch is exercised in the remote target environment')
    def test_rejects_v4_direct_tensor_return(self):
        processor = mock.Mock(return_value={'pixel_values': torch.ones((1, 1))})
        model = mock.Mock()
        model.get_image_features.return_value = torch.ones((1, 768))
        with self.assertRaises(Siglip2AdapterError):
            encode_single_global(model=model, processor=processor, image=object(), device='cpu')
