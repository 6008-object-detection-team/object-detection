"""Guard against guessing graph stride or accepting incompatible skip paths."""
import unittest

from tools.audit_p2_architecture import infer_head_outputs


class StrideTests(unittest.TestCase):
    def test_conv_upsample_and_skip_produce_explicit_head_strides(self):
        configuration = {"backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]]], "head": [
                [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
                [[-1, 1], 1, "Concat", [1]],
                [-1, 2, "C3k2", [32, True]],
                [[5, 2], 1, "Detect", [80]]]}
        result = infer_head_outputs(configuration)
        self.assertEqual(result["output_strides"], [4, 8])
        self.assertEqual(result["input_layer_indices"], [5, 2])
        self.assertEqual(result["layer_index"], 6)

    def test_incompatible_skip_concat_is_rejected(self):
        configuration = {"backbone": [[-1, 1, "Conv", [16, 3, 2]],
                                      [-1, 1, "Conv", [32, 3, 2]]],
                         "head": [[[0, 1], 1, "Concat", [1]],
                                  [[2], 1, "Detect", [80]]]}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            infer_head_outputs(configuration)

    def test_unknown_module_is_rejected_instead_of_assuming_same_stride(self):
        configuration = {"backbone": [[-1, 1, "UnknownConv", [16, 3, 2]]],
                         "head": [[[0], 1, "Detect", [80]]]}
        with self.assertRaisesRegex(ValueError, "Unsupported module"):
            infer_head_outputs(configuration)


if __name__ == "__main__":
    unittest.main()
