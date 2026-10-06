"""Mock precision validation; no Torch import or hardware access."""
from types import SimpleNamespace
import unittest
from tools.probe_precision_runtime import precision_snapshot


class Parameter:
    def __init__(self,dtype,device='cuda:0'):
        self.dtype=dtype;self.device=device
    def is_floating_point(self):return True


def fake_engine(dtype,fp16,device='cuda:0'):
    native=SimpleNamespace(parameters=lambda:iter([Parameter(dtype,device)]))
    backend=SimpleNamespace(model=native,fp16=fp16)
    predictor=SimpleNamespace(model=SimpleNamespace(backend=backend),args=SimpleNamespace(quantize=16 if fp16 else 32))
    owned=SimpleNamespace(parameters=lambda:iter([Parameter('torch.float32',device)]))
    return SimpleNamespace(model=SimpleNamespace(model=owned,predictor=predictor))


class PrecisionTests(unittest.TestCase):
    def test_native_backend_is_checked_even_when_original_stays_fp32(self):
        result=precision_snapshot(fake_engine('torch.float16',True),16)
        self.assertTrue(result['precision_confirmed'])
        self.assertEqual(result['source_parameter_dtype'],'torch.float32')
        self.assertEqual(result['actual_parameter_dtype_counts'],{'torch.float16':1})
    def test_precision_or_device_fallback_is_rejected(self):
        for engine,bits in [(fake_engine('torch.float32',False),16),(fake_engine('torch.float16',True,'cpu'),16)]:
            with self.assertRaises(RuntimeError):precision_snapshot(engine,bits)


if __name__=='__main__':unittest.main()
