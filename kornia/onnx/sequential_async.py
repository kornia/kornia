import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort

from .module import ONNXModule
from .sequential import ONNXSequential
from .mixin import ONNXMixin, ONNXRuntimeMixin

__all__ = ["PipelineItem",  "ONNXSequentialAsync"]


@dataclass
class PipelineItem:
    """Represents an item to be processed through the pipeline.

    Attributes:
        input_data: The input data for the pipeline.
        future: A Future object to hold the result of the pipeline processing.
    """
    input_data: Any
    future: asyncio.Future


class ONNXSequentialAsync(ONNXMixin, ONNXRuntimeMixin):
    """An asynchronous pipeline for processing inputs through multiple ONNX model stages in parallel.

    The pipeline is designed to allow multiple inputs to be in different stages simultaneously,
    maximizing resource utilization and throughput.

    Args:
        workers (List[asyncio.Task]): A list of worker tasks for each pipeline stage.
        *args: A variable number of ONNX models (either ONNX ModelProto objects or file paths).
            For Hugging Face-hosted models, use the format 'hf://model_name'. Valid `model_name` can be found on
            https://huggingface.co/kornia/ONNX_models. Or a URL to the ONNX model.
        providers: Execution providers for ONNXRuntime (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        session_options: Optional ONNXRuntime session options for session configuration and optimizations.
        cache_dir: The directory where ONNX models are cached locally (only for downloading from HuggingFace).
            Defaults to None, which will use a default `{kornia_config.hub_onnx_dir}` directory.
        verbose: If True, prints the status of the pipeline. Defaults to True.
    """

    def __init__(
        self,
        *args: Union["onnx.ModelProto", str, ONNXModule, ONNXSequential],  # type:ignore
        providers: Optional[List[str]] = None,
        session_options: Optional["ort.SessionOptions"] = None,  # type:ignore
        cache_dir: Optional[str] = None,
        verbose: bool = True
    ) -> None:
        self.operators = self.load_ops_as_modules(*args, cache_dir=cache_dir)
        self.num_stages = len(self.operators)
        for op in self.operators:
            sess = op.create_session(op.op, providers=providers, session_options=session_options)
            op.set_session(sess)
        # Each pipeline stage will have its own asyncio.Queue to receive inputs from the previous stage
        self.queues = [asyncio.Queue() for _ in range(self.num_stages)]
        self.verbose = verbose
        self.running = False
        self.workers = []

    def _load_ops(
        self,
        *args: Union["onnx.ModelProto", str],  # type:ignore
        cache_dir: Optional[str] = None,
    ) -> List["onnx.ModelProto"]:  # type:ignore
        """Loads multiple ONNX models or operators and returns them as a list.

        Args:
            *args: A variable number of ONNX models (either ONNX ModelProto objects or file paths).
                For Hugging Face-hosted models, use the format 'hf://model_name'. Valid `model_name` can be found on
                https://huggingface.co/kornia/ONNX_models. Or a URL to the ONNX model.

        Returns:
            List[onnx.ModelProto]: The loaded ONNX models as a list of ONNX graphs.
        """
        op_list = []
        for arg in args:
            if isinstance(arg, (ONNXModule, ONNXSequential,)):
                op_list.append(arg)
            else:
                op_list.append(super()._load_op(arg, cache_dir=cache_dir))
        return op_list

    def load_ops_as_modules(
        self,
        *args: Union["onnx.ModelProto", str],  # type:ignore
        cache_dir: str | None = None
    ) -> List[ONNXModule]:  # type:ignore
        ops = self._load_ops(*args, cache_dir=cache_dir)
        return list([ONNXModule(op) for op in ops])

    def as_cpu(self, **kwargs: Any) -> None:
        """Set the session to run on CPU."""
        [op.as_cpu(**kwargs) for op in self.operators]

    def as_cuda(self, device_id: int = 0, **kwargs: Any) -> None:
        """Set the session to run on CUDA.

        We set the ONNX runtime session to use CUDAExecutionProvider. For other CUDAExecutionProvider configurations,
        or CUDA/cuDNN/ONNX version issues,
        you may refer to https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html.

        Args:
            device_id: Select GPU to execute.
        """
        [op.as_cuda(device_id=device_id, **kwargs) for op in self.operators]

    def as_tensorrt(self, device_id: int = 0, **kwargs: Any) -> None:
        """Set the session to run on TensorRT.

        We set the ONNX runtime session to use TensorrtExecutionProvider. For other TensorrtExecutionProvider configurations,
        or CUDA/cuDNN/ONNX/TensorRT version issues,
        you may refer to https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html.

        Args:
            device_id: select GPU to execute.
        """
        [op.as_tensorrt(device_id=device_id, **kwargs) for op in self.operators]

    def as_openvino(self, device_type: str = "GPU", **kwargs: Any) -> None:
        """Set the session to run on TensorRT.

        We set the ONNX runtime session to use OpenVINOExecutionProvider. For other OpenVINOExecutionProvider configurations,
        or CUDA/cuDNN/ONNX/TensorRT version issues,
        you may refer to https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html.

        Args:
            device_type: CPU, NPU, GPU, GPU.0, GPU.1 based on the avaialable GPUs, NPU, Any valid Hetero combination,
                Any valid Multi or Auto devices combination.
        """
        [op.as_openvino(device_type=device_type, **kwargs) for op in self.operators]

    def export(self, file_path: str, **kwargs: Any) -> None:
        raise RuntimeError("export is not supported for `ONNXSequentialAsync`")

    def add_metadata(
        self, additional_metadata: List[Tuple[str]] = []
    ) -> "onnx.ModelProto":  # type:ignore
        raise RuntimeError("export is not supported for `ONNXSequentialAsync`")

    async def _worker(self, stage_idx: int) -> None:
        """
        Worker coroutine responsible for processing items in a specific pipeline stage.

        Args:
            stage_idx: 
                The index of the stage this worker is responsible for.
        """
        current_stage = self.operators[stage_idx]
        input_queue = self.queues[stage_idx]
        next_queue = self.queues[stage_idx + 1] if stage_idx + 1 < self.num_stages else None

        while True:
            item: Optional[PipelineItem] = await input_queue.get()
            if item is None:
                # Pass the sentinel to the next stage if it exists
                if next_queue is not None:
                    await next_queue.put(None)
                input_queue.task_done()
                break  # Terminate the worker

            try:
                data = item.input_data
                # Run the current stage
                output = await current_stage.run(data)
                
                if next_queue is not None:
                    # Pass the output to the next stage
                    new_item = PipelineItem(input_data=output, future=item.future)
                    await next_queue.put(new_item)
                else:
                    # Last stage: set the result in the future
                    if not item.future.cancelled():
                        item.future.set_result(output)
            except Exception as e:
                if not item.future.cancelled():
                    item.future.set_exception(e)
            finally:
                input_queue.task_done()

    def start(self) -> None:
        """Starts the pipeline by launching worker coroutines for each stage.

        If the pipeline is already running, this method has no effect.
        """
        if not self.running:
            self.running = True
            # Start a worker for each stage
            for stage_idx in range(self.num_stages):
                worker_task = asyncio.create_task(self._worker(stage_idx))
                self.workers.append(worker_task)
            if self.verbose:
                print("ONNX Pipeline started.")

    async def stop(self) -> None:
        """Stops the pipeline by sending shutdown signals to all workers.

        Ensures that all queued items are processed before terminating workers.
        """
        if self.running:
            # Send sentinel to the first stage to initiate shutdown
            await self.queues[0].put(None)
            # Wait for all queues to be processed
            await asyncio.gather(*(queue.join() for queue in self.queues))
            # Await all workers to finish
            await asyncio.gather(*self.workers)
            self.workers = []
            self.running = False
            if self.verbose:
                print("ONNX Pipeline stopped.")

    async def pipeline(self, input_data: Any) -> Any:
        """
        Enqueues input data for processing through the pipeline and awaits the result.

        Args:
            input_data: 
                The input data to process through the pipeline. Typically a NumPy array matching the
                input shape expected by the first pipeline stage.
        """
        if not self.running:
            self.start()
        future = asyncio.get_event_loop().create_future()
        item = PipelineItem(input_data=input_data, future=future)
        await self.queues[0].put(item)
        return await future
