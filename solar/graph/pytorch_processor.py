# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch model processor for extracting and analyzing PyTorch models.

This module processes PyTorch model files (including kernelbench)
to extract computation graphs and model information.

Supports both 'Model' and 'ReferenceModel' class names, and uses
launch_reference_implementation to infer how to call the model.
"""

import ast
import contextlib
import gc
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchview
from torch import nn

from solar.common.types import ProcessingConfig
from solar.common.utils import (
    ensure_directory,
    load_module_from_file,
    setup_safe_environment,
)
from solar.graph._torchview_param_shim import (
    UnsupportedFrontendOperandError,
    ensure_parameter_capture,
)
from solar.graph.torchview_processor import TorchviewProcessor
from solar.graph import tensor_layouts as _tensor_layouts
from solar.graph import torchview_capability as _torchview_capability
from solar.graph.torchview_capability import TorchviewCapabilityError
import logging

logger = logging.getLogger(__name__)

# Learned nn.Parameter operands of function calls (e.g. torch.matmul(w, x))
# are dropped by unpatched torchview; capture them before any trace runs.
ensure_parameter_capture()


@contextlib.contextmanager
def _patch_torchview_collect_attributes():
    """Monkey-patch torchview to propagate collect_attributes correctly.

    torchview has two bugs that prevent collect_attributes from working
    when input_data is provided (rather than input_size):

    1. process_input() calls get_recorder_tensor without collect_attributes,
       so initial TensorNodes always have collect_attributes=False.
    2. RecorderTensor.__torch_function__() omits collect_attributes from
       attach_kwargs for output TensorNodes, breaking propagation through
       chained function calls.

    See patches/torchview-collect-attributes.patch for the equivalent
    source-level fix.
    """
    import torchview.torchview as tv_mod
    import torchview.recorder_tensor as rt_mod

    orig_process_input = tv_mod.process_input
    orig_torch_function = rt_mod.RecorderTensor.__torch_function__

    def _patched_process_input(input_data, input_size, kwargs, device,
                               dtypes=None, collect_attributes=False):
        from torchview.torchview import (
            set_device, traverse_data, get_recorder_tensor,
            get_correct_input_sizes, get_input_tensor,
            RecorderTensor, reduce_data_info, collect_tensor_node,
        )
        from torchview.recorder_tensor import NodeContainer
        from torchview.computation_node import TensorNode

        x = None
        correct_input_size = []
        kwargs_recorder_tensor = traverse_data(
            kwargs, lambda t: get_recorder_tensor(t, collect_attributes), type
        )
        if input_data is not None:
            x = set_device(input_data, device)
            x = traverse_data(
                x, lambda t: get_recorder_tensor(t, collect_attributes), type
            )
            if isinstance(x, RecorderTensor):
                x = [x]

        if input_size is not None:
            if dtypes is None:
                dtypes = [torch.float] * len(input_size)
            correct_input_size = get_correct_input_sizes(input_size)
            x = get_input_tensor(correct_input_size, dtypes, device,
                                 collect_attributes)

        input_data_node: NodeContainer[TensorNode] = (
            reduce_data_info(
                [x, kwargs_recorder_tensor], collect_tensor_node,
                NodeContainer()
            )
        )
        return x, kwargs_recorder_tensor, input_data_node

    @classmethod  # type: ignore[misc]
    def _patched_torch_function(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        from torchview.recorder_tensor import (
            RecorderTensor, reduce_data_info, collect_tensor_node,
        )
        from torchview.computation_node import TensorNode, NodeContainer

        # Read the inputs' flag BEFORE the call: for an in-place op the
        # operand's latest TensorNode after the call is the op's own result
        # node, whose default flag is False, and reading it there turned the
        # flag off for every later call on that tensor (R8 / INTEG-6).
        recorder_nodes: NodeContainer[TensorNode] = (
            reduce_data_info([args, kwargs], collect_tensor_node,
                             NodeContainer())
        )
        collect_attributes = (
            next(iter(recorder_nodes)).collect_attributes if recorder_nodes else None
        )

        # Call the original, then fix collect_attributes on output nodes
        out = orig_torch_function.__func__(cls, func, types, args, kwargs)
        if not recorder_nodes:
            return out

        # Propagate to output TensorNodes
        output_nodes: NodeContainer[TensorNode] = (
            reduce_data_info(out, collect_tensor_node, NodeContainer())
        )
        for node in output_nodes:
            node.collect_attributes = collect_attributes

        # Identity binding of this call's tensor arguments to the operands the
        # recorder attached (an earlier decision round 5): read while the graph is written.
        try:
            from solar.graph import tensor_layouts as _tl
            fn_nodes = [p for node in output_nodes for p in (getattr(node, "parents", None) or [])
                        if type(p).__name__ == "FunctionNode"]
            if fn_nodes:
                fn_nodes[0].call_operands = _tl.bind_call_operands(fn_nodes[0], args, kwargs)
        except Exception:
            pass

        return out

    tv_mod.process_input = _patched_process_input
    rt_mod.RecorderTensor.__torch_function__ = _patched_torch_function

    try:
        yield
    finally:
        tv_mod.process_input = orig_process_input
        rt_mod.RecorderTensor.__torch_function__ = orig_torch_function


def _import_torch_dynamo_before_capture() -> bool:
    """Import ``torch._dynamo`` BEFORE any model is traced.

    ``torch._dynamo.variables.torch_function`` runs
    ``populate_builtin_to_tensor_fn_map()`` at import time: it executes every
    builtin operator (``abs``, ``pow``, ``<<``, ``&``, comparisons, in-place
    and reflected variants, ...) on a one-element tensor subclass that defines
    ``__torch_function__``.  When that import happens lazily *during* a
    capture (Mamba2's ``einops``/``einsum`` path triggers it), the recorder
    sees those probe calls and torchview emits ~140 stray FunctionNodes
    (``Model.__ilshift__``, ``Model.positive``, two ``[1]x[1]`` matmuls, ...)
    into the model's graph.  The lineage avoided this only by accident: its
    constructor probe imported ``torch._dynamo`` before the real capture.
    Importing it here makes capture coverage independent of process history.
    Returns True when the module is importable (recorded in the capability
    record as ``dynamo_warmup``).
    """
    try:
        import torch._dynamo  # noqa: F401
        return True
    except Exception:  # pragma: no cover - torch builds without dynamo
        return False


def _check_torchview_parameter_support() -> None:
    """Verify that the installed torchview exposes parameter-tensor nodes.

    Kept for callers/tests of the lineage API; it is a thin view of the
    engine-wide capability gate (:mod:`solar.graph.torchview_capability`,
    issue #45), which is the single source of truth.  Raises ``RuntimeError``
    (same guidance as before) when the probe sees no ``parameter-tensor``
    node; the constructor's gate additionally requires call-attribute
    propagation and honours ``allow_degraded_capture``.
    """
    # the canonical probe context: what a real capture of this processor sees (P2-2)
    record = _torchview_capability.get_torchview_capability(draw_context=_patch_torchview_collect_attributes)
    if record.parameter_tensors:
        return
    raise RuntimeError(
        "Installed torchview does not generate parameter-tensor nodes.\n"
        "SOLAR depends on a patched torchview (or its in-memory parameter shim) to extract\n"
        "weight/bias tensor nodes (parameter-tensor). Please follow solar/install.sh or\n"
        "solar/install_uv.sh to install and patch torchview.\n"
        "The patch can be found at: solar/patches/torchview-parameter-tensors.patch\n"
        f"Capability record: {record.to_dict()}"
    )


def _unwrap_unsupported_operand(exc):
    """torchview wraps forward errors in a generic RuntimeError; recover the
    typed fail-closed diagnostic from the exception chain if present."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, UnsupportedFrontendOperandError):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


class _BufferSnapshot(dict):
    """Pre-capture buffer state of a model.

    * the dict itself: registered name -> CPU copy, for every buffer whose
      values can be copied (not meta, not ``None``);
    * ``alias_of``: alias name -> primary name (two names for one tensor);
    * ``registry``: the COMPLETE original registry, every registered name
      (including meta tensors and ``None`` slots) -> ``{"shape", "dtype",
      "device", "none", "persistent"}``; this, not the value map, decides
      which slots existed before the first attempt;
    * ``supported``: ``True`` for an ``nn.Module`` (a valid registry, even an
      empty one); ``False`` for a stand-in that has no registry at all.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alias_of: dict = {}
        self.registry: dict = {}
        self.supported: bool = True


class PyTorchProcessor:
    """Process PyTorch model into a saved torch graph."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the PyTorchProcessor.

        Args:
            config: Processing configuration. If None, uses defaults.
        """
        self.config = config or ProcessingConfig()
        self.torchview_processor = TorchviewProcessor(debug=self.config.debug)
        # Engine-wide torchview capability gate (issue #45): probe once per
        # process, fail closed unless the caller explicitly accepts a degraded
        # capture.  The record travels with every output stage.
        record = _torchview_capability.get_torchview_capability(draw_context=_patch_torchview_collect_attributes)
        # Capture coverage must not depend on whether torch._dynamo was already
        # imported (its import-time builtin probe would be recorded as model ops);
        # the mitigation flag is serialized with the record (P3-1).
        self.dynamo_warmup = _import_torch_dynamo_before_capture()
        import dataclasses as _dc
        self.capture_capability = _dc.replace(record, dynamo_warmup=self.dynamo_warmup)
        if self.capture_capability.degraded and not getattr(self.config, "allow_degraded_capture", False):
            raise TorchviewCapabilityError(self.capture_capability)
        if self.capture_capability.degraded:
            logger.info(f"⚠️ Degraded capture accepted (--allow-degraded-capture): {self.capture_capability.reason}; "
                  "every layer will be marked not supportable")
        self._setup_environment()
        # parameter support is part of the capability gate below (single source of truth)

    def _setup_environment(self) -> None:
        """Set up safe execution environment."""
        if self.config.safe_mode:
            setup_safe_environment()

    def process_model_file(self, file_path: str, output_dir: str) -> bool:
        """Process PyTorch model file into `output_dir`.

        Args:
            file_path: Path to a Python file containing `Model`/`ReferenceModel` and `get_inputs()`.
            output_dir: Output directory to write graph artifacts to.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.config.debug:
                logger.debug(f"Processing {file_path}...")

            output_path = Path(output_dir)
            ensure_directory(output_path)

            # Check if already processed — reuse requires the artifact's
            # recorded source bytes AND capture-implementation identity to
            # match the current ones; a stale pre-fix artifact must never
            # stand in for a fresh (possibly fail-closed) run.
            if self._is_already_processed(output_path, file_path) \
                    and not self.config.force_rerun:
                logger.info(f"Skipping {file_path} - already processed "
                      f"(provenance matches)")
                return True

            # Copy source file for reproducibility (best effort).
            try:
                import shutil

                filename = Path(file_path).name
                # Avoid double-prefixing if already starts with "source_"
                if filename.startswith("source_"):
                    source_copy = output_path / filename
                else:
                    source_copy = output_path / f"source_{filename}"

                # Clean up any old source_* files to avoid accumulation
                for old_source in output_path.glob("source_*.py"):
                    old_source.unlink()

                # Always copy the latest source file
                shutil.copy2(file_path, source_copy)
            except Exception:
                pass

            # Load and process model
            model, inputs, module = self._load_model(file_path)
            if model is None:
                return False

            # Generate torchview graph (pass output_path for saving visualization)
            # Layout records are bound to the tensors torchview traces (inputs,
            # nested ones included) while the capture runs (an earlier decision round 4,
            # input-3); parameters bind through the traced model object.
            with _tensor_layouts.bind_input_layouts():
                graph = self._generate_torchview_graph(model, inputs, str(output_path), module=module)
            if graph is None:
                return False

            # Process and save graph
            kernel_name = Path(file_path).stem
            self.torchview_processor.process_graph(
                graph, str(output_path), kernel_name, original_model=model,
                capture_capability=self.capture_capability.to_dict(),
            )

            # Bind EVERY returned output of the model's forward (issue #53):
            # tuple/list/dict leaves recorded by the forward hook during the
            # capture, each bound to its torchview output-tensor node when the
            # recorder traced it, or left UNBOUND (allocation + in-place
            # fill, e.g. ``zeros(n); out[1:] = ...``) so the converter and
            # analyzer can charge its footprint and fail closed.
            self._write_returned_outputs(output_path)

            # Patch input tensor dtypes into pytorch_graph.yaml.
            # The torchview tracer loses non-float dtypes (e.g. torch.bool)
            # because the fallback infers dtype from model parameters.
            # We fix this by reading the actual dtypes from get_inputs().
            self._patch_input_dtypes(output_path, inputs)

            # Record provenance for safe resume: source bytes + capture
            # implementation identity (see _is_already_processed).
            self._write_provenance(output_path, file_path)

            # Clean up
            self._cleanup(model, inputs, graph)

            logger.info(f"✓ Successfully processed {file_path}")
            return True

        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            return False

    # Files whose bytes define the capture implementation (P2-1): processor
    # (recorder wrappers, input mover, warm-up), node writer, capability gate,
    # the lineage's parameter recorder shim, and the installed torchview
    # recorder/driver modules the wrappers patch.
    @staticmethod
    def _capture_implementation_files() -> list:
        from solar.graph import _torchview_param_shim as shim, torchview_processor as tvp, torchview_capability as cap
        files = [__file__, tvp.__file__, cap.__file__, shim.__file__]
        try:
            import torchview.recorder_tensor as rt_mod, torchview.torchview as tv_mod
            files += [rt_mod.__file__, tv_mod.__file__]
        except Exception:
            pass
        return [str(Path(f).resolve()) for f in files]

    @classmethod
    def _capture_rev(cls) -> str:
        """Identity of the complete capture implementation: sha256 over the
        bytes of every file in :meth:`_capture_implementation_files`."""
        import hashlib
        h = hashlib.sha256()
        for f in cls._capture_implementation_files():
            try:
                h.update(Path(f).name.encode()); h.update(Path(f).read_bytes())
            except Exception:
                h.update(f"missing:{f}".encode())
        return h.hexdigest()

    @staticmethod
    def _capability_identity(record: Any) -> Dict[str, Any]:
        """The effective dependency/capability identity a saved artifact must match to be reused."""
        d = record if isinstance(record, dict) else (record.to_dict() if record is not None else {})
        return {k: d.get(k) for k in ("source", "torchview_version", "torchview_file", "parameter_tensors",
                                      "call_attributes", "degraded", "dynamo_warmup")}

    def _write_provenance(self, output_dir: Path, file_path: str) -> None:
        """Append source/capture provenance to pytorch_graph.yaml."""
        import hashlib

        import yaml
        graph_path = output_dir / "pytorch_graph.yaml"
        if not graph_path.exists():
            return
        doc = yaml.safe_load(graph_path.read_text())
        doc["provenance"] = {
            "source_sha256": hashlib.sha256(
                Path(file_path).read_bytes()).hexdigest(),
            "capture_rev": self._capture_rev(),
            "capture_implementation_files": self._capture_implementation_files(),
            "capability_identity": self._capability_identity(getattr(self, "capture_capability", None)),
        }
        graph_path.write_text(yaml.dump(doc, default_flow_style=False,
                                        sort_keys=False))

    def _is_already_processed(self, output_dir: Path,
                              file_path: str = None) -> bool:
        """True only if a graph artifact exists AND its recorded provenance
        (source bytes + capture-implementation identity) matches the current
        run.  Artifact existence alone is NOT sufficient: a stale pre-fix
        artifact must not turn an unsupported source into supported output.
        """
        graph_path = output_dir / "pytorch_graph.yaml"
        if not graph_path.exists():
            return False
        if file_path is None:  # legacy callers: existence only
            return True
        import hashlib

        import yaml
        try:
            doc = yaml.safe_load(graph_path.read_text())
            prov = doc.get("provenance") or {}
        except Exception:
            return False
        if not prov:
            return False  # pre-provenance artifact: reprocess
        source_now = hashlib.sha256(
            Path(file_path).read_bytes()).hexdigest()
        if prov.get("source_sha256") != source_now or prov.get("capture_rev") != self._capture_rev():
            return False
        # P2-1: a saved artifact without a capability record, with a record that
        # does not match the effective one now, or that was degraded, is stale.
        saved = doc.get("capture_capability")
        now = self._capability_identity(getattr(self, "capture_capability", None))
        if not isinstance(saved, dict) or doc.get("capture_status") != "ok":
            if self.config.debug:
                logger.debug("Saved graph has no capable capture record: reprocessing")
            return False
        if self._capability_identity(saved) != now or prov.get("capability_identity") != now:
            if self.config.debug:
                logger.debug("Saved capture record differs from the effective capability: reprocessing")
            return False
        return True

    def _get_model_class(self, module: Any) -> Tuple[Optional[type], str]:
        """Get the model class from a module.

        Supports both 'Model' and 'ReferenceModel' class names.

        Args:
            module: Loaded Python module.

        Returns:
            Tuple of (model_class, class_name) or (None, "") if not found.
        """
        # Try Model first (kernelbench convention)
        if hasattr(module, 'Model'):
            return module.Model, 'Model'

        # Try ReferenceModel (SolBench V1 convention)
        if hasattr(module, 'ReferenceModel'):
            return module.ReferenceModel, 'ReferenceModel'

        return None, ""

    def _infer_model_call_from_launch(self, module: Any, model: nn.Module, inputs: Any) -> Callable:
        """Infer how to call the model from launch_reference_implementation.

        Analyzes launch_reference_implementation to determine if the model
        should be called with *inputs (unpacked) or with inputs directly.

        Args:
            module: Loaded Python module.
            model: Model instance.
            inputs: Inputs from get_inputs().

        Returns:
            A callable that takes (model, inputs) and returns model output.
        """
        if not hasattr(module, 'launch_reference_implementation'):
            # Default: call model with unpacked inputs if tuple/list, else directly
            def default_call(m, i):
                if isinstance(i, (tuple, list)):
                    return m(*i)
                return m(i)
            return default_call

        # Try to analyze the source code of launch_reference_implementation
        try:
            source = inspect.getsource(module.launch_reference_implementation)

            # Parse the source to understand how model is called
            # Common patterns:
            # 1. return model(*inputs) - unpack inputs
            # 2. return model(inputs) - pass inputs directly (rare)
            # 3. return model(input1, input2, ...) - specific args

            if '*inputs' in source or '*args' in source:
                # Unpacking pattern
                def unpack_call(m, i):
                    if isinstance(i, (tuple, list)):
                        return m(*i)
                    return m(i)
                return unpack_call
            else:
                # Try to detect single argument pattern
                def single_call(m, i):
                    if isinstance(i, (tuple, list)):
                        return m(*i)
                    return m(i)
                return single_call

        except Exception:
            pass

        # Fallback: try to call launch_reference_implementation signature
        def fallback_call(m, i):
            if isinstance(i, (tuple, list)):
                return m(*i)
            return m(i)
        return fallback_call

    def _infer_init_args_from_module(self, module: Any, model_class: type) -> Tuple[tuple, dict]:
        """Infer model init args from module globals.

        Looks for global constants that match the model's __init__ signature.
        Common patterns:
        - HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS, etc.

        Args:
            module: Loaded Python module.
            model_class: The model class to instantiate.

        Returns:
            Tuple of (args, kwargs) for model instantiation.
        """
        try:
            sig = inspect.signature(model_class.__init__)
            params = list(sig.parameters.items())[1:]  # Skip 'self'

            args = []
            kwargs = {}

            # Common name mappings from parameter names to global constants
            name_mappings = {
                'hidden_size': ['HIDDEN_SIZE', 'hidden_size', 'D_MODEL', 'd_model'],
                'intermediate_size': ['INTERMEDIATE_SIZE', 'intermediate_size', 'FFN_DIM', 'ffn_dim'],
                'num_heads': ['NUM_HEADS', 'num_heads', 'N_HEADS', 'n_heads'],
                'num_attention_heads': ['NUM_ATTENTION_HEADS', 'num_attention_heads', 'NUM_HEADS'],
                'num_key_value_heads': ['NUM_KEY_VALUE_HEADS', 'num_key_value_heads', 'N_KV_HEADS'],
                'head_dim': ['HEAD_DIM', 'head_dim'],
                'num_layers': ['NUM_LAYERS', 'num_layers', 'N_LAYERS'],
                'vocab_size': ['VOCAB_SIZE', 'vocab_size'],
                'max_seq_len': ['MAX_SEQ_LEN', 'max_seq_len', 'SEQ_LEN'],
                'dropout': ['DROPOUT', 'dropout'],
                'eps': ['EPS', 'eps', 'RMS_NORM_EPS', 'LAYER_NORM_EPS'],
                'batch_size': ['BATCH_SIZE', 'batch_size'],
                'seq_len': ['SEQ_LEN', 'seq_len'],
            }

            for param_name, param in params:
                # Skip if has default
                if param.default != inspect.Parameter.empty:
                    continue

                # Try to find matching global
                found = False
                candidates = name_mappings.get(param_name, [param_name.upper(), param_name])

                for candidate in candidates:
                    if hasattr(module, candidate):
                        value = getattr(module, candidate)
                        if isinstance(value, (int, float, str, bool)):
                            args.append(value)
                            found = True
                            if self.config.debug:
                                logger.debug(f"Inferred {param_name}={value} from {candidate}")
                            break

                if not found:
                    # Can't infer this required argument
                    if self.config.debug:
                        logger.debug(f"Could not infer required arg: {param_name}")
                    return (), {}

            return tuple(args), kwargs

        except Exception as e:
            if self.config.debug:
                logger.debug(f"Error inferring init args: {e}")
            return (), {}

    def _create_model_instance(self, module: Any) -> Optional[nn.Module]:
        """Create a fresh model instance from a loaded module.

        Tries get_init_inputs(), no-arg constructor, and inferred args.

        Args:
            module: Loaded Python module containing a Model/ReferenceModel class.

        Returns:
            Model instance or None if all strategies fail.
        """
        model_class, class_name = self._get_model_class(module)
        if model_class is None:
            return None

        # Strategy 1: Use get_init_inputs() if available
        if hasattr(module, 'get_init_inputs'):
            try:
                init_inputs = module.get_init_inputs()
                model = model_class(*init_inputs) if init_inputs else model_class()
                if self.config.debug:
                    logger.debug(f"Created model using get_init_inputs()")
                return model
            except Exception as e:
                if self.config.debug:
                    logger.debug(f"get_init_inputs() failed: {e}")

        # Strategy 2: Try no-arg constructor
        try:
            model = model_class()
            if self.config.debug:
                logger.debug(f"Created model using no-arg constructor")
            return model
        except TypeError:
            pass

        # Strategy 3: Infer init args from module globals
        args, kwargs = self._infer_init_args_from_module(module, model_class)
        if args or kwargs:
            try:
                model = model_class(*args, **kwargs)
                if self.config.debug:
                    logger.debug(f"Created model using inferred args: {args}")
                return model
            except Exception as e:
                if self.config.debug:
                    logger.debug(f"Inferred args failed: {e}")

        return None

    def _load_model(self,
                   file_path: str) -> Tuple[Optional[nn.Module], Optional[Any], Optional[Any]]:
        """Load a model from a Python file.

        Supports both 'Model' and 'ReferenceModel' class names.
        Uses launch_reference_implementation to infer how to call the model.
        Falls back to inferring init args from module globals.

        Args:
            file_path: Path to the model file.

        Returns:
            Tuple of (model, inputs) or (None, None) if failed.
        """
        try:
            module = load_module_from_file(file_path)

            # Get model class (Model or ReferenceModel)
            model_class, class_name = self._get_model_class(module)
            if model_class is None:
                print(f"Warning: No Model or ReferenceModel class found in {file_path}")
                return None, None, None

            if self.config.debug:
                logger.debug(f"Found model class: {class_name}")

            # Check for get_inputs
            if not hasattr(module, 'get_inputs'):
                print(f"Warning: No get_inputs function found in {file_path}")
                return None, None, None

            model = self._create_model_instance(module)
            if model is None:
                print(f"Error: Could not instantiate model class {class_name} in {file_path}")
                return None, None, None

            # Get inputs.  Try to allocate on meta device first to avoid
            # OOM on large workloads.  Fall back to CPU if meta fails
            # (some get_inputs use ops incompatible with meta device).
            inputs = None

            # Strategy 1: SolBench v3 model with custom _ref_get_inputs
            if inputs is None and hasattr(module, '_ref_get_inputs'):
                try:
                    import re as _re
                    src = Path(file_path).read_text()
                    axes_match = _re.search(r"_axes\s*=\s*(\{[^}]+\})", src)
                    if axes_match:
                        _axes = eval(axes_match.group(1))  # noqa: S307
                        inp_dict = module._ref_get_inputs(_axes, torch.device("meta"))
                        order_match = _re.search(r"_param_order\s*=\s*(\[[^\]]+\])", src)
                        if order_match:
                            _param_order = eval(order_match.group(1))  # noqa: S307
                            inputs = [inp_dict[k] for k in _param_order if k in inp_dict]
                        if self.config.debug:
                            logger.debug("Allocated inputs on meta device (via _ref_get_inputs)")
                except Exception:
                    inputs = None

            # Strategy 2: Monkey-patch torch.randn/zeros/ones/empty to allocate
            # on meta device, then call get_inputs() normally.
            if inputs is None:
                try:
                    _orig_randn = torch.randn
                    _orig_zeros = torch.zeros
                    _orig_ones = torch.ones
                    _orig_empty = torch.empty

                    def _meta_factory(orig_fn):
                        def wrapper(*args, **kwargs):
                            kwargs.pop("device", None)
                            return orig_fn(*args, device="meta", **kwargs)
                        return wrapper

                    torch.randn = _meta_factory(_orig_randn)
                    torch.zeros = _meta_factory(_orig_zeros)
                    torch.ones = _meta_factory(_orig_ones)
                    torch.empty = _meta_factory(_orig_empty)
                    try:
                        inputs = module.get_inputs()
                        if self.config.debug:
                            logger.debug("Allocated inputs on meta device (via patched factories)")
                    finally:
                        torch.randn = _orig_randn
                        torch.zeros = _orig_zeros
                        torch.ones = _orig_ones
                        torch.empty = _orig_empty
                except Exception:
                    inputs = None

            # Strategy 3: Fall back to normal CPU allocation
            if inputs is None:
                inputs = module.get_inputs()

            # Store the inferred call pattern for later use
            self._model_caller = self._infer_model_call_from_launch(module, model, inputs)

            return model, inputs, module

        except Exception as e:
            print(f"Error loading model from {file_path}: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            return None, None, None

    def _generate_torchview_graph(
        self,
        model: nn.Module,
        inputs: Any,
        output_dir: Optional[str] = None,
        module: Any = None,
    ) -> Optional[Any]:
        """Generate a torchview computation graph.

        Args:
            model: PyTorch model.
            inputs: Model inputs (may be on meta device).
            output_dir: Directory to save graph visualization (if save_graph enabled).
            module: Original loaded module (for re-generating CPU inputs on fallback).

        Returns:
            Computation graph or None if failed.
        """
        # Try meta device first (zero memory), fall back to CPU.
        # Meta attempt uses to_empty() which destroys the model in-place,
        # so CPU fallback re-creates the model from the module.
        # Snapshot every registered buffer (values, shapes, alias identities,
        # registered names) once, before the first attempt destroys values
        # (``to_empty``).  EVERY capture attempt on the same model object
        # starts from that state: the meta attempt (aliases/shapes restored
        # right after ``to_empty``) and the module-less CPU retry (values,
        # shapes, aliases, registered names; BigBird token_type_ids /
        # position_ids; earlier contracts).  The lineage's module-backed CPU
        # fallback re-creates the model from its source module instead, which
        # re-initialises buffers by construction.
        saved_buffers = self._snapshot_buffers(model)
        try:
            # every capture attempt starts with no binding-time records
            # (TARGET-ALIAS-FALLBACK-1): see tensor_layouts.reset_attempt_records
            _tensor_layouts.reset_attempt_records()
            meta_model = model.to_empty(device="meta")
            # The meta attempt is a capture attempt too (round 4,
            # meta-alias-1): ``to_empty`` allocates one fresh meta tensor
            # per registered slot, which separates two names that referred to
            # one tensor.  Restore the snapshot for this device (shapes and
            # alias identities on meta) so the program captured here is the
            # one the model had before the attempt.
            self._restore_buffers(meta_model, saved_buffers, device="meta")
            meta_model.eval()

            if self._is_rnn_model(meta_model):
                raise RuntimeError("RNN models not supported on meta device")

            meta_inputs = self._move_inputs_to_device(inputs, "meta")

            with _patch_torchview_collect_attributes(), self._record_returned_outputs(meta_model):
                graph = torchview.draw_graph(
                    meta_model,
                    input_data=meta_inputs,
                    device="meta",
                    save_graph=False,
                    expand_nested=True,
                    depth=float('inf'),
                    hide_module_functions=False,
                    hide_inner_tensors=False,
                    roll=False,
                    strict=False,
                    collect_attributes=True,
                )

            if self.config.debug:
                logger.debug(f"✅ Generated graph using meta device")

            if self.config.save_graph and output_dir:
                self._save_torchview_graph(graph, output_dir)

            return graph

        except Exception as e:
            unsupported = _unwrap_unsupported_operand(e)
            if unsupported is not None:
                # fail-closed diagnostic is device-independent: surface the
                # typed error instead of retrying on CPU or masking it
                # behind torchview's generic wrapper
                raise unsupported
            if self.config.debug:
                logger.debug(f"⚠️ Meta device failed: {e}")

        # CPU fallback: re-create model from module to get a pristine instance
        # (to_empty destroyed non-persistent buffers like position_ids).
        if module is None:
            # No source module to re-instantiate from: retry on the SAME model
            # object with the pre-capture buffer state restored for the CPU
            # device (values, shapes, aliases, registered names; a buffer that
            # disappeared refuses the retry) - the reviewed an earlier change/an earlier change contract.
            try:
                _tensor_layouts.reset_attempt_records()          # new attempt (same-object CPU retry)
                model = model.to_empty(device="cpu")
                self._restore_buffers(model, saved_buffers, device="cpu")
                model.eval()
                device_inputs = self._move_inputs_to_device(inputs, "cpu")
                with _patch_torchview_collect_attributes(), self._record_returned_outputs(model):
                    graph = torchview.draw_graph(
                        model,
                        input_data=device_inputs,
                        device="cpu",
                        save_graph=False,
                        expand_nested=True,
                        depth=float('inf'),
                        hide_module_functions=False,
                        hide_inner_tensors=False,
                        roll=False,
                        strict=False,
                        collect_attributes=True,
                    )
                if self.config.debug:
                    logger.debug("✅ Generated graph using cpu device (same-object retry)")
                if self.config.save_graph and output_dir:
                    self._save_torchview_graph(graph, output_dir)
                return graph
            except Exception as e:
                # same rule as the target's module-backed fallback: a typed
                # fail-closed diagnostic is surfaced, never torchview's wrapper
                unsupported = _unwrap_unsupported_operand(e)
                if unsupported is not None:
                    raise unsupported
                if self.config.debug:
                    logger.debug(f"⚠️ CPU device failed: {e}")
                raise

        if self.config.debug:
            logger.debug("Falling back to CPU: re-creating model from module")

        try:
            _tensor_layouts.reset_attempt_records()              # new attempt (module-backed CPU fallback)
            cpu_model = self._create_model_instance(module)
            if cpu_model is None:
                return None
            cpu_model.eval()

            cpu_inputs = module.get_inputs()

            with _patch_torchview_collect_attributes(), self._record_returned_outputs(cpu_model):
                graph = torchview.draw_graph(
                    cpu_model,
                    input_data=cpu_inputs,
                    device="cpu",
                    save_graph=False,
                    expand_nested=True,
                    depth=float('inf'),
                    hide_module_functions=False,
                    hide_inner_tensors=False,
                    roll=False,
                    strict=False,
                    collect_attributes=True,
                )

            if self.config.debug:
                logger.debug(f"✅ Generated graph using cpu device")

            if self.config.save_graph and output_dir:
                self._save_torchview_graph(graph, output_dir)

            return graph

        except Exception as e:
            unsupported = _unwrap_unsupported_operand(e)
            if unsupported is not None:
                raise unsupported
            if self.config.debug:
                logger.debug(f"⚠️ CPU device failed: {e}")
            raise

    @staticmethod
    def _registered_buffer_slots(model: nn.Module) -> list:
        """Every registered buffer slot as ``(dotted_name, owner, attr, tensor_or_None)``.

        Walks ``_buffers`` directly (``named_buffers`` omits ``None`` slots)
        over ``named_modules(remove_duplicate=False)`` so a child shared under
        two paths contributes both names, as the alias logic expects.
        """
        slots = []
        for prefix, owner in model.named_modules(remove_duplicate=False):
            for attr, buf in owner._buffers.items():
                slots.append((f"{prefix}.{attr}" if prefix else attr, owner, attr, buf))
        return slots

    @staticmethod
    def _snapshot_buffers(model: nn.Module) -> dict:
        """Record the model's pre-capture buffer state so ``to_empty`` cannot lose it.

        Returns a :class:`_BufferSnapshot`: a ``dict`` of registered name ->
        CPU copy for every buffer that carries values, ``alias_of`` for names
        that share one tensor (an alias in one module, or a buffer shared
        across children; recorded for meta tensors too, whose values cannot
        be copied), and ``registry`` with EVERY registered name
        (``named_modules(remove_duplicate=False)`` x ``_buffers``), including
        meta tensors (no values to copy, but the registration and its
        persistence flag are original state) and ``None`` slots.  Parameters
        are not snapshotted: their values do not affect the captured shapes
        and copying them would double the footprint of large models.  A
        stand-in that is not an ``nn.Module`` yields ``supported=False``.
        """
        saved = _BufferSnapshot()
        if not isinstance(model, nn.Module):
            saved.supported = False
            return saved
        primary_for_tensor: dict = {}
        for name, owner, attr, buf in PyTorchProcessor._registered_buffer_slots(model):
            is_tensor = isinstance(buf, torch.Tensor)
            saved.registry[name] = {
                "shape": tuple(buf.shape) if is_tensor else None,
                "dtype": buf.dtype if is_tensor else None,
                "device": buf.device.type if is_tensor else None,
                "none": not is_tensor,
                "persistent": attr not in owner._non_persistent_buffers_set,
            }
            if not is_tensor:
                continue  # None slot: the registry entry preserves it
            # Alias identity is recorded for EVERY tensor slot, copyable or
            # not (round 5, meta-alias-2): two names for one initially-
            # meta tensor have no values to copy but must be rebound to one
            # object after ``to_empty`` separated them.
            key = id(buf)
            if key in primary_for_tensor:
                primary = primary_for_tensor[key]
                saved.alias_of[name] = primary
                if primary in saved:
                    saved[name] = saved[primary]
                continue
            primary_for_tensor[key] = name
            if buf.device.type == "meta":
                continue  # nothing to copy; registry + alias identity are the state
            saved[name] = buf.detach().to("cpu", copy=True)
        return saved

    @staticmethod
    def _rebind_buffer(model: nn.Module, dotted_name: str, tensor: Optional[torch.Tensor]) -> None:
        """Make ``dotted_name`` refer to ``tensor`` (keeps the persistence flag)."""
        module_path, _, attr = dotted_name.rpartition(".")
        owner = model.get_submodule(module_path) if module_path else model
        owner._buffers[attr] = tensor

    @staticmethod
    def _restore_buffers(model: nn.Module, saved: dict, device: Optional[str] = None) -> None:
        """Put the model's registered buffers back into the snapshotted state.

        The complete original ``registry`` decides which slots existed before
        the first attempt: a currently registered name absent from it was
        created by a failed attempt and is dropped (``None`` slots included);
        a registry name that is no longer registered cannot be restored and
        raises ``RuntimeError`` (the retry is refused rather than run on
        invalid state); an original ``None`` slot is reset to ``None``; an
        original meta buffer keeps its registration, shape and persistence
        flag (its values were never copyable, so on a data device it keeps
        whatever ``to_empty`` allocated, as before this helper existed).  For
        every name with saved values, on a device that carries data: if the
        slot still has the saved shape the values are copied in (converted to
        the slot's dtype); if a failed attempt resized the slot it is rebound
        to a fresh tensor of the SAVED shape and values.  On the meta device
        only shapes and alias identities are restored.  Aliases are rebound to
        the restored primary tensor so ``a.buf is b.buf`` holds again.
        Persistence flags follow the registry.  Parameters are never touched.
        A snapshot from a non-module stand-in (``supported=False``) restores
        nothing.

        ``device`` is the CURRENT attempt's device (the capture loop passes
        it).  It matters for a slot the failed attempt set to ``None``:
        ``to_empty`` skips ``None`` slots, so nothing tells the slot which
        device the retry runs on.  An originally valued slot is then rebuilt
        from the saved values on ``device`` (shape, dtype, alias identity
        included); on meta only the shape.  Without ``device`` the saved
        values are restored where they live (CPU), never as a data-free meta
        tensor.  An original meta buffer whose slot is ``None`` is rebuilt as
        an uninitialised tensor on ``device`` (meta if unknown).
        """
        if not getattr(saved, "supported", True) or not isinstance(model, nn.Module):
            return
        registry = getattr(saved, "registry", None)
        if registry is None:  # a plain dict snapshot: values only (legacy callers)
            registry = {name: {"none": False, "persistent": None, "shape": tuple(v.shape)} for name, v in saved.items()}
        alias_of = getattr(saved, "alias_of", {})
        current = {name: (owner, attr, buf) for name, owner, attr, buf in PyTorchProcessor._registered_buffer_slots(model)}
        missing = [name for name in registry if name not in current]
        if missing:
            raise RuntimeError(
                "capture retry refused: registered buffer(s) disappeared during a "
                f"failed attempt and cannot be restored: {missing}")
        for name in [n for n in current if n not in registry]:
            PyTorchProcessor._drop_buffer(model, name)
        restored: dict = {}
        with torch.no_grad():
            for name, entry in registry.items():
                owner, attr, buf = current[name]
                if entry.get("persistent") is not None:
                    if entry["persistent"]:
                        owner._non_persistent_buffers_set.discard(attr)
                    else:
                        owner._non_persistent_buffers_set.add(attr)
                if name in alias_of:
                    continue  # aliases are rebound after their primary
                if entry.get("none"):
                    if buf is not None:
                        PyTorchProcessor._rebind_buffer(model, name, None)
                    continue
                if name not in saved:
                    # original meta buffer: registration, shape and alias
                    # identity are the state (no values are invented)
                    if isinstance(buf, torch.Tensor) and tuple(buf.shape) != tuple(entry["shape"]):
                        buf = torch.empty(tuple(entry["shape"]), dtype=buf.dtype, device=buf.device)
                        PyTorchProcessor._rebind_buffer(model, name, buf)
                    elif buf is None:
                        buf = torch.empty(tuple(entry["shape"]), dtype=entry.get("dtype") or torch.float32,
                                          device=device or "meta")
                        PyTorchProcessor._rebind_buffer(model, name, buf)
                    restored[name] = buf      # aliases of an initially-meta primary rebind to this object
                    continue
                src = saved[name]
                if buf is None:
                    # The failed attempt replaced an originally VALUED slot with
                    # None; to_empty skipped it, so rebuild it for THIS attempt:
                    # saved values on a data device, shape only on meta.
                    target = device or str(src.device)
                    dtype = entry.get("dtype") or src.dtype
                    if torch.device(target).type == "meta":
                        buf = torch.empty(tuple(src.shape), dtype=dtype, device="meta")
                    else:
                        buf = src.to(device=target, dtype=dtype, copy=True)
                    PyTorchProcessor._rebind_buffer(model, name, buf)
                elif buf.device.type == "meta":
                    if tuple(buf.shape) != tuple(src.shape):
                        buf = torch.empty(tuple(src.shape), dtype=buf.dtype, device="meta")
                        PyTorchProcessor._rebind_buffer(model, name, buf)
                elif tuple(buf.shape) == tuple(src.shape):
                    buf.copy_(src.to(device=buf.device, dtype=buf.dtype))
                else:
                    buf = src.to(device=buf.device, dtype=buf.dtype, copy=True)
                    PyTorchProcessor._rebind_buffer(model, name, buf)
                restored[name] = buf
            for name, primary in alias_of.items():
                PyTorchProcessor._rebind_buffer(model, name, restored[primary])

    @staticmethod
    def _drop_buffer(model: nn.Module, dotted_name: str) -> None:
        """Remove a buffer registered during a failed attempt (not in the original registry)."""
        module_path, _, attr = dotted_name.rpartition(".")
        owner = model.get_submodule(module_path) if module_path else model
        owner._buffers.pop(attr, None)
        owner._non_persistent_buffers_set.discard(attr)

    @staticmethod
    def _inputs_on_meta(inputs: Any) -> bool:
        """Check if any input tensor is on meta device."""
        if isinstance(inputs, torch.Tensor):
            return inputs.device.type == "meta"
        if isinstance(inputs, (list, tuple)):
            return any(
                isinstance(x, torch.Tensor) and x.device.type == "meta"
                for x in inputs
            )
        if isinstance(inputs, dict):
            return any(
                isinstance(v, torch.Tensor) and v.device.type == "meta"
                for v in inputs.values()
            )
        return False

    @staticmethod
    def _move_inputs_to_device(inputs: Any, device: str) -> Any:
        """Move input tensors to the given device.

        For ``meta`` device this replaces real tensors with zero-memory
        meta tensors that preserve shape and dtype, avoiding large CPU
        allocations when only graph structure is needed.
        """
        import torch

        def _fresh(obj: torch.Tensor, dev: str) -> torch.Tensor:
            # Zero-memory stand-in that keeps the real tensor's layout, so a
            # layout record bound to the traced input is the source's (an earlier decision
            # round 4, input-3).  Overlapping layouts (stride 0, expand)
            # fall back to a contiguous stand-in; the source layout is still
            # noted for the binding.
            strides = [int(x) for x in obj.stride()]
            if all(st > 0 for st in strides):
                moved = torch.empty_strided(tuple(obj.shape), tuple(strides), dtype=obj.dtype, device=dev)
            else:
                moved = torch.empty(obj.shape, dtype=obj.dtype, device=dev)
            _tensor_layouts.note_source_layout(moved, obj)
            return moved

        def _move(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                if device == "meta":
                    return _fresh(obj, "meta")
                # Handle meta->cpu: can't .to() a meta tensor, recreate instead
                if obj.device.type == "meta" and device != "meta":
                    return _fresh(obj, device)
                return obj.to(device)
            if isinstance(obj, (list, tuple)):
                moved = [_move(x) for x in obj]
                return type(obj)(moved)
            if isinstance(obj, dict):
                return {k: _move(v) for k, v in obj.items()}
            return obj

        return _move(inputs)

    _MAX_RENDER_EDGES = 20000

    def _save_torchview_graph(self, graph: Any, output_dir: str) -> None:
        """Save torchview graph visualization to the output directory.

        Args:
            graph: Torchview graph object.
            output_dir: Directory to save the graph visualization.
        """
        try:
            output_path = Path(output_dir)
            graph_filename = output_path / "torchview_graph"

            # torchview's visual_graph is a graphviz.Digraph object
            if hasattr(graph, 'visual_graph'):
                edge_count = len(getattr(graph, 'edge_list', []))
                if edge_count > self._MAX_RENDER_EDGES:
                    graph.visual_graph.save(filename=str(graph_filename) + ".gv")
                    if self.config.debug:
                        logger.debug(f"⚠️ Graph has {edge_count} edges (>{self._MAX_RENDER_EDGES}); "
                            f"saved DOT file only (PDF rendering would be very slow)")
                    return

                graph.visual_graph.render(
                    filename=str(graph_filename),
                    format='pdf',
                    cleanup=True,
                )
                if self.config.debug:
                    logger.debug(f"Saved torchview graph: {graph_filename}.pdf")
            else:
                if self.config.debug:
                    logger.debug("⚠️ Graph object does not have visual_graph attribute")
        except Exception as e:
            if self.config.debug:
                logger.debug(f"⚠️ Failed to save torchview graph: {e}")

    def _is_rnn_model(self, model: nn.Module) -> bool:
        """Check if a model is RNN-like.

        Args:
            model: PyTorch model.

        Returns:
            True if RNN-like, False otherwise.
        """
        # Check for RNN attributes
        if hasattr(model, 'hidden'):
            return True

        # Check module names
        for name, _ in model.named_modules():
            if any(rnn_type in name.lower() for rnn_type in ['rnn', 'gru', 'lstm']):
                return True

        return False

    # ------------------------------------------------------------------
    # Returned-output binding (issue #53, SOL58-OUTPUT-BOUNDARY)
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_returned(out: Any) -> list:
        """Flatten a forward() return value into leaf records.

        Tuples/lists (named tuples included), dicts and nested combinations
        are walked; every ``torch.Tensor`` leaf becomes a record with its
        path, shape, dtype and byte footprint.  A leaf the torchview recorder
        traced (a ``RecorderTensor`` with ``tensor_nodes``) is *bound* to its
        latest TensorNode id; a plain tensor (allocated inside forward and
        filled in place, or otherwise untraced) is recorded UNBOUND.
        Non-tensor leaves are recorded by type so the count of returned
        values is complete.
        """
        leaves: list = []
        seen: dict = {}   # storage_key -> first return path (exact aliases)

        def _tensor_identity(t: torch.Tensor) -> dict:
            """Identity at hook time: python object id, storage identity
            (C storage handle; falls back to data_ptr / object id), storage
            offset, SHAPE, STRIDE, element size and the storage's byte size.
            Two leaves are the same bytes (an exact alias) only when storage,
            offset, shape, stride and element size are ALL equal
            (return-stride-4: ``b[:4]`` and ``b[::2]`` share storage,
            offset, numel and width but touch different bytes).  The
            converter computes the per-storage union of byte regions from
            these fields, so equal numel never stands in for equal extent."""
            storage_id = None
            storage_nbytes = None
            identity_kind = "object_fallback"
            # real-alias-1: a traced leaf is a RecorderTensor subclass whose
            # storage accessors go through the recorder's __torch_function__; read
            # the storage from the PLAIN view of the same tensor so a bound output
            # and a plain-tensor alias of it (``y.as_subclass(torch.Tensor)``)
            # record the SAME storage handle.  ``obj:`` identity is used only when
            # the storage truly cannot be obtained, and is marked as such.
            try:
                plain = t.as_subclass(torch.Tensor) if type(t) is not torch.Tensor else t
            except Exception:
                plain = t
            try:
                st = plain.untyped_storage()
                cdata = getattr(st, "_cdata", None)
                if cdata:
                    storage_id = cdata; identity_kind = "storage_handle"
                else:
                    ptr = st.data_ptr()
                    if ptr:
                        storage_id = ptr; identity_kind = "data_ptr"
                try:
                    storage_nbytes = int(st.nbytes())
                except Exception:
                    storage_nbytes = None
            except Exception:
                storage_id = None
            if not storage_id:
                storage_id = f"obj:{id(t)}"; identity_kind = "object_fallback"
            try:
                offset = int(t.storage_offset())
            except Exception:
                offset = 0
            numel = int(t.numel())
            esize = int(t.element_size())
            shape = [int(d) for d in t.shape]
            try:
                stride = [int(d) for d in t.stride()]
            except Exception:
                stride = None
            stride_key = ",".join(str(d) for d in stride) if stride is not None else "?"
            shape_key = ",".join(str(d) for d in shape)
            return {"object_id": id(t), "storage_id": str(storage_id), "storage_identity": identity_kind,
                    "storage_offset": offset,
                    "numel": numel, "element_size": esize,
                    "shape": shape, "stride": stride, "storage_nbytes": storage_nbytes,
                    "storage_key": f"{storage_id}:{offset}:[{shape_key}]:[{stride_key}]:{esize}"}

        def _nbytes(t: torch.Tensor) -> int:
            try:
                return int(t.numel()) * int(t.element_size())
            except Exception:
                return 0

        def walk(obj: Any, path: str) -> None:
            if isinstance(obj, torch.Tensor):
                nodes = getattr(obj, "tensor_nodes", None)
                node_id = None
                if nodes:
                    try:
                        node_id = str(getattr(nodes[-1], "node_id", id(nodes[-1])))
                    except Exception:
                        node_id = None
                # Capture-local identity (return-alias-1 / -STRIDE-4): the
                # SAME tensor (or an exact alias: same storage, offset, shape,
                # stride, width) returned twice must be billed once; distinct
                # allocations of equal shape/dtype must never be merged; views
                # of one storage with different layouts are distinct leaves whose
                # byte regions the converter unions per storage.
                ident = _tensor_identity(obj)
                leaves.append({
                    "path": path, "kind": "tensor",
                    "shape": [int(d) for d in obj.shape],
                    "dtype": str(obj.dtype), "nbytes": _nbytes(obj),
                    "bound": node_id is not None,
                    "torchview_node_id": node_id,
                    "tensor_identity": ident,
                    "alias_of_path": seen.get(ident["storage_key"]),
                })
                seen.setdefault(ident["storage_key"], path)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{path}[{k!r}]")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")
            else:
                leaves.append({"path": path, "kind": type(obj).__name__, "bound": None})

        walk(out, "return")
        return leaves

    @contextlib.contextmanager
    def _record_returned_outputs(self, model: nn.Module):
        """Record the structure of ``model.forward``'s return value during a capture.

        A forward hook on the root module sees the raw return value (torchview's
        own hooks are registered later inside ``draw_graph``), so bound leaves
        are still ``RecorderTensor`` instances and unbound ones plain tensors.
        The last attempt's record wins (meta attempt, then CPU fallbacks).
        """
        self._returned_outputs = None

        def _hook(_mod, _inp, out):
            try:
                self._returned_outputs = self._flatten_returned(out)
            except Exception as exc:  # never break the capture, but never look complete either
                self._returned_outputs = [{"path": "return", "kind": "unrecorded",
                                           "bound": None, "error": f"{type(exc).__name__}: {exc}"}]
            return None

        handle = None
        try:
            handle = model.register_forward_hook(_hook)
        except Exception:
            handle = None
        try:
            yield
        finally:
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass

    def _write_returned_outputs(self, output_path: Path) -> None:
        """Append ``returned_outputs`` (+ provenance) to pytorch_graph.yaml.

        Each tensor leaf carries ``bound_node``: the clean id of the torchview
        ``output-tensor`` node it was traced to, or ``null`` when the recorder
        never saw it.  The converter turns unbound leaves into declared model
        outputs and the analyzer charges their footprint and marks the
        accounting incomplete (fail closed).
        """
        import yaml
        graph_path = output_path / "pytorch_graph.yaml"
        recorded = getattr(self, "_returned_outputs", None)
        if not graph_path.exists():
            return
        try:
            doc = yaml.safe_load(graph_path.read_text()) or {}
        except Exception:
            return
        # return-hook-error-2: missing or failed metadata must never look like
        # a verified legacy result.  A graph produced by THIS capture implementation
        # always carries the key; "unrecorded" (hook never fired, or the extractor
        # raised) is an explicit incomplete reason downstream.
        if recorded is None:
            recorded = [{"path": "return", "kind": "unrecorded", "bound": None,
                         "error": "forward hook did not observe the model's return value"}]
        unrecorded = [r for r in recorded if isinstance(r, dict) and r.get("kind") == "unrecorded"]
        id_map = getattr(self.torchview_processor, "_original_to_clean_id", {}) or {}
        out_list = []
        n_unbound = 0
        for leaf in recorded:
            rec = dict(leaf)
            if rec.get("kind") == "tensor":
                tv_id = rec.pop("torchview_node_id", None)
                clean = id_map.get(str(tv_id)) if tv_id is not None else None
                if clean is not None and not str((doc.get("layers") or {}).get(clean, {}).get("type", "")).lower().startswith("output-tensor"):
                    # traced, but not as a model output node (e.g. an
                    # intermediate also returned): keep the binding to the
                    # traced tensor node, the analyzer charges declared
                    # outputs through model_outputs
                    pass
                rec["bound_node"] = clean
                rec["bound"] = clean is not None
                if clean is None:
                    n_unbound += 1
            out_list.append(rec)
        doc["returned_outputs"] = out_list
        doc["returned_outputs_unbound"] = int(n_unbound)
        doc["returned_outputs_unrecorded"] = bool(unrecorded)
        if unrecorded:
            doc["returned_outputs_error"] = "; ".join(str(r.get("error")) for r in unrecorded)
        doc["model_output_provenance"] = "pytorch-return-v1"
        from solar.common.utils import NoAliasDumper
        graph_path.write_text(yaml.dump(doc, Dumper=NoAliasDumper, default_flow_style=False, sort_keys=False))

    def _patch_input_dtypes(self, output_path: Path, inputs: Any) -> None:
        """Patch actual input tensor dtypes into pytorch_graph.yaml.

        The torchview tracer falls back to model-parameter dtypes for tensor
        nodes, losing non-float dtypes (e.g. ``torch.bool``).  This method
        reads the actual dtypes from the ``get_inputs()`` return value and
        overwrites the corresponding ``auxiliary-tensor`` / ``input-tensor``
        nodes in the saved graph.
        """
        import yaml
        graph_path = output_path / "pytorch_graph.yaml"
        if not graph_path.exists():
            return

        # Collect actual dtypes from inputs
        actual_dtypes: list[str] = []
        if isinstance(inputs, (list, tuple)):
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    actual_dtypes.append(str(inp.dtype))
        elif isinstance(inputs, dict):
            for inp in inputs.values():
                if isinstance(inp, torch.Tensor):
                    actual_dtypes.append(str(inp.dtype))

        if not actual_dtypes:
            return

        try:
            with open(graph_path) as f:
                graph = yaml.safe_load(f) or {}
        except Exception:
            return

        layers = graph.get("layers", {})
        # Find auxiliary-tensor / input-tensor nodes (model inputs) in order
        input_nodes = [
            (nid, node) for nid, node in layers.items()
            if str(node.get("type", "")).lower() in ("auxiliary-tensor", "input-tensor")
        ]

        changed = False
        for idx, (nid, node) in enumerate(input_nodes):
            if idx >= len(actual_dtypes):
                break
            actual = actual_dtypes[idx]
            existing = node.get("output_dtypes", [])
            if existing != [actual]:
                node["output_dtypes"] = [actual]
                node["input_dtypes"] = [actual]
                changed = True

        if changed:
            from solar.common.utils import NoAliasDumper
            with open(graph_path, "w") as f:
                yaml.dump(graph, f, Dumper=NoAliasDumper, sort_keys=False,
                          default_flow_style=False)

    def _cleanup(self, *objects: Any) -> None:
        """Clean up objects and run garbage collection.

        Args:
            *objects: Objects to delete.
        """
        for obj in objects:
            del obj
        gc.collect()
