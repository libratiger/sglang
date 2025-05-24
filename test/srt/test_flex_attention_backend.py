import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch,
)


class TestFlexAttnBackend(unittest.TestCase):
    def test_latency(self):
        ret = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST, other_args=["--attention-backend", "flex"]
        )
        if is_in_ci():
            # TODO(LISA): This is a placeholder value.
            # The actual performance of flex attention should be profiled.
            assert ret["output_throughput"] > 50.0

    def test_mmlu(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST

        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "flex"],
        )

        try:
            # Run MMLU evaluation
            args = SimpleNamespace(
                host="127.0.0.1",
                port=int(base_url.split(":")[-1]),
                model_path=model,
                model_name=model, # Though not strictly necessary for run_eval with base_url
                eval_name="mmlu",
                num_examples=64, # Keep this small for CI
                num_threads=32,
                max_model_len=None, # Use default from model config
                seed=0,
                conv_template="vicuna_v1.1", # Or any suitable default
                data_path="dataset/mmlu", # Assuming this path exists or run_eval handles it
                result_file=None, # Don't write to file for this test
                upload_result=False,
                verbose=False,
                param_file=None,
                report_file=None,
                report_host=None,
                report_port=None,
                use_alternative_prompt=False,
                chat_template=None, # Let server use its default
                trust_remote_code=True, # Consistent with server launch
                api_key=None,
                auth_token=None,
                retry=3
            )
            metrics = run_eval(args)
            # Based on torch_native_attention_backend.py
            # Might need adjustment for different models or if flex attention affects numerics.
            assert metrics["score"] >= 0.65
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
