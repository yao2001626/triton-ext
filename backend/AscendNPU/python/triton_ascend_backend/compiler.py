"""A5 SIMT compilation through an external BiShengIR toolchain."""
from dataclasses import dataclass, asdict, field
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from triton.backends.compiler import BaseBackend
from triton.runtime.cache import get_dump_manager
from triton._C.libtriton import ir, passes
from .driver import NPUUtils
from . import _bindings as ascend
from . import knobs
from .program_grid import (PROGRAM_GRID_TRANSFORMS_ATTR,
                           ProgramGridContractError, get_persistent_transform,
                           normalize_program_grid_transforms)
from .utils import (_get_npucompiler_path, _is_ascend_sanitizer_enabled,
                    _is_debug_line_info_disabled, _enable_msdebug,
                    _is_auto_map_parallel_blocks_enabled,
                    _get_auto_blockify_blacklist_reasons)


def _get_then_remove_rc(mod, attr_name: str) -> int:
    get_int_attr = getattr(ascend.ir, "get_int_attr", None)
    remove_attr = getattr(ascend.ir, "remove_attr", None)

    if get_int_attr is None:
        return -1
    try:
        attr_value = get_int_attr(mod, attr_name)
    except TypeError:
        return -1

    if remove_attr:
        remove_attr(mod, attr_name)

    if not isinstance(attr_value, int):
        return -1

    return attr_value


def _get_then_remove_program_grid_transforms(mod):
    get_transforms = getattr(ascend.ir, "get_program_grid_transforms", None)
    remove_attr = getattr(ascend.ir, "remove_attr", None)
    if get_transforms is None:
        if PROGRAM_GRID_TRANSFORMS_ATTR in str(mod):
            raise RuntimeError(
                "hacc.program_grid_transforms requires the matching Ascend C++ binding"
            )
        return None
    try:
        raw = get_transforms(mod)
    except TypeError:
        if PROGRAM_GRID_TRANSFORMS_ATTR in str(mod):
            raise RuntimeError(
                "hacc.program_grid_transforms requires an MLIR module accepted by "
                "the Ascend C++ binding")
        return None
    if raw is not None and remove_attr:
        remove_attr(mod, PROGRAM_GRID_TRANSFORMS_ATTR)
    return raw


def _export_coalesce_metadata(mod, metadata, *, require_row_contract=False):
    # Tile/strided coalescing (TritonToLinalg) records the chosen coalesce factor
    # H and the grid axis it applies to as module attrs hacc.coalesce_factor /
    # hacc.coalesce_axis. In the full-TA design the *launcher* (driver.py) owns
    # the grid division: it divides grid[axis] by H before launch, and bishengir
    # no longer interprets the attrs. Read them into metadata here and strip them
    # from the module so the hacc.* attrs never reach hivmc (which rejects
    # unknown module attrs). Absent attrs -> factor 1 (no-op) / axis -1.
    # RowCoalescing also records whether the launcher should use ceil-div when
    # shrinking the grid, because its runtime valid-count guard handles tails.
    # A Row result is only valid as the complete triple.  The regular T2L
    # Axis/Chunk ABI predates ceil-div and deliberately remains compatible with
    # a missing ceil-div attr.
    factor = _get_then_remove_rc(mod, "hacc.coalesce_factor")
    axis = _get_then_remove_rc(mod, "hacc.coalesce_axis")
    ceil_div = _get_then_remove_rc(mod, "hacc.coalesce_grid_ceil_div")
    has_any_contract_attr = any(value != -1
                                for value in (factor, axis, ceil_div))
    valid_factor = isinstance(factor, int) and factor > 1
    valid_axis = isinstance(axis, int) and axis in (0, 1, 2)
    valid_ceil_div = isinstance(ceil_div, int) and ceil_div > 0
    if has_any_contract_attr and (not valid_factor or not valid_axis):
        raise RuntimeError("invalid hacc.coalesce launch contract")
    if require_row_contract and has_any_contract_attr and not valid_ceil_div:
        raise RuntimeError(
            "RowCoalescing requires hacc.coalesce_grid_ceil_div")

    metadata["coalesce_factor"] = factor if valid_factor else 1
    metadata["coalesce_axis"] = axis if valid_axis else -1
    metadata["coalesce_grid_ceil_div"] = valid_ceil_div
    metadata["row_coalescing_applied"] = metadata["coalesce_factor"] > 1


def _export_program_grid_metadata(mod,
                                  metadata,
                                  *,
                                  require_row_contract=False):
    raw_transforms = _get_then_remove_program_grid_transforms(mod)
    if raw_transforms is not None:
        factor = _get_then_remove_rc(mod, "hacc.coalesce_factor")
        axis = _get_then_remove_rc(mod, "hacc.coalesce_axis")
        ceil_div = _get_then_remove_rc(mod, "hacc.coalesce_grid_ceil_div")
        has_legacy_attrs = any(value != -1
                               for value in (factor, axis, ceil_div))
        if has_legacy_attrs:
            raise RuntimeError(
                "hacc.program_grid_transforms conflicts with legacy hacc.coalesce_* metadata"
            )
        try:
            transforms = normalize_program_grid_transforms(raw_transforms)
        except ProgramGridContractError as error:
            raise RuntimeError(
                f"invalid hacc.program_grid_transforms: {error}") from error
        metadata["program_grid_transforms"] = transforms
        metadata["program_grid_mapping_applied"] = True
        metadata["coalesce_factor"] = 1
        metadata["coalesce_axis"] = -1
        metadata["coalesce_grid_ceil_div"] = False
        metadata["row_coalescing_applied"] = False
        return

    metadata["program_grid_transforms"] = None
    metadata["program_grid_mapping_applied"] = False
    _export_coalesce_metadata(
        mod,
        metadata,
        require_row_contract=require_row_contract,
    )


def _finalize_program_launch_policy(metadata, opt):
    required_fields = (
        "program_grid_transforms",
        "program_grid_mapping_applied",
        "row_coalescing_applied",
        "has_auto_blockify_blacklist_op",
        "mix_mode",
    )
    missing = [name for name in required_fields if name not in metadata]
    if missing:
        raise RuntimeError(
            "cannot finalize program launch policy; missing metadata: " +
            ", ".join(missing))

    raw_transforms = metadata["program_grid_transforms"]
    mapping_applied = metadata["program_grid_mapping_applied"]
    row_coalescing_applied = metadata["row_coalescing_applied"]
    has_auto_blockify_blacklist_op = metadata["has_auto_blockify_blacklist_op"]
    if not isinstance(mapping_applied, bool):
        raise RuntimeError("program_grid_mapping_applied must be a boolean")
    if not isinstance(row_coalescing_applied, bool):
        raise RuntimeError("row_coalescing_applied must be a boolean")
    if not isinstance(has_auto_blockify_blacklist_op, bool):
        raise RuntimeError("has_auto_blockify_blacklist_op must be a boolean")

    transforms = None
    if raw_transforms is not None:
        try:
            transforms = normalize_program_grid_transforms(raw_transforms)
        except ProgramGridContractError as error:
            raise RuntimeError(
                f"invalid exported program_grid_transforms: {error}"
            ) from error
    if mapping_applied != (transforms is not None):
        raise RuntimeError(
            "program_grid_mapping_applied disagrees with program_grid_transforms"
        )
    if mapping_applied and row_coalescing_applied:
        raise RuntimeError(
            "program-grid mapping conflicts with legacy RowCoalescing")

    persistent_transform = get_persistent_transform(
        transforms) if transforms is not None else None
    ptsm_cap_authorized = False
    if persistent_transform is not None:
        if metadata["mix_mode"] != "aiv":
            raise RuntimeError(
                "persistent program-grid transform requires final mix_mode=aiv"
            )
        if not (persistent_transform["persistent_coverage"]
                and persistent_transform["grid_stride_abi_verified"]):
            raise RuntimeError(
                "persistent program-grid transform lacks coverage/ABI verification"
            )
        ptsm_cap_authorized = True

    if opt.is_pure_simt:
        auto_blockify_enabled = _is_auto_map_parallel_blocks_enabled(
        ) and not row_coalescing_applied
    else:
        auto_blockify_enabled = (_is_auto_map_parallel_blocks_enabled()
                                 and not has_auto_blockify_blacklist_op
                                 and not ptsm_cap_authorized)

    if auto_blockify_enabled and ptsm_cap_authorized:
        raise RuntimeError(
            "AutoBlockify and persistent-grid cap cannot both be enabled")

    metadata["auto_blockify_enabled"] = auto_blockify_enabled
    metadata["ptsm_cap_authorized"] = ptsm_cap_authorized


def _serialize_module(mod, opt):
    if getattr(
            opt, "debug",
            False) or not _is_debug_line_info_disabled() or _enable_msdebug():
        return str(mod)
    return mod.str_nodebug()


def _parse_ttir_metadata(ttir: str, metadata: dict):
    """
    Parse TTIR to extract metadata required for NPU compilation.
    Extracts and updates the following fields in metadata:
      - kernel_name
      - shared (currently hardcoded)
    """
    # --- Regular expressions and examples ---
    # Example: tt.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"tt\.func\spublic\s+@(\w+)"

    # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
    TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'

    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    # Note: Currently, for TTIR inputs, we only support vector kernels.
    metadata["mix_mode"] = "aiv"
    match = re.search(KERNEL_NAME_REGEX, ttir)
    if match is None:
        raise ValueError("Expected a public TTIR kernel entry point")
    metadata["kernel_name"] = match.group(1)
    metadata["name"] = metadata["kernel_name"]
    # Keep this as compiler-derived safety metadata.  In particular, do not
    # trust a caller-provided False value to override an unsafe TTIR pattern.
    metadata["has_auto_blockify_blacklist_op"] = bool(
        _get_auto_blockify_blacklist_reasons(ttir))
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [
        int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, ttir)
    ]
    return metadata


def _is_a5_target_arch(arch: str) -> bool:
    return isinstance(arch, str) and arch.startswith(
        ("Ascend910_95", "Ascend950"))


def get_common_bishengir_compile_options(metadata):
    bishengir_target = metadata['target'].arch
    bishengir_target_opt = f"--target={bishengir_target}"
    return [bishengir_target_opt]


def make_ttir(mod, metadata, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod, 'ascend_ttir')
    if opt.debug:
        dump_manager = get_dump_manager(metadata["hash"])
        print(f"Dumping intermediate results to {dump_manager.cache_dir}")
        dump_manager.put(str(mod), "kernel.ttir.mlir", binary=False)
    return mod


@dataclass(frozen=True)
class AscendOptions:
    arch: str
    num_warps: int = 32
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 32
    debug: bool = False
    enable_fp_fusion: bool = True
    shared_mem_dynamic_size: int = 122880
    backend_name: str = 'npu'
    compile_mode: str = 'simt_only'
    is_pure_simt: bool = field(default=True, init=False)
    parallel_mode: str = field(default='simt', init=False)
    compile_on_910_95: bool = field(default=True, init=False)
    simt_optimization_mode: int = 0
    superblock_factor: int = 1
    bisheng_options: str | None = None
    disable_fma: bool = False
    num_aicore: int = 0
    num_vectorcore: int = 0
    allowed_dot_input_precisions: tuple = ('ieee', )
    default_dot_input_precision: str = 'ieee'
    supported_fp8_dtypes: tuple = ()
    deprecated_fp8_dot_operand_dtypes: tuple = ()
    max_num_imprecise_acc_default: int = 0
    sanitize_overflow: bool = False
    fpsan_homomorphic_casts: bool = False
    instrumentation_mode: str = 'none'
    launch_cooperative_grid: bool = False
    extern_libs: tuple = ()

    def __post_init__(self):
        # Upstream serializes options.__dict__, including derived launch fields.
        for name, option_field in self.__dataclass_fields__.items():
            if not option_field.init:
                object.__setattr__(self, name, option_field.default)
        if self.shared_mem_dynamic_size is None:
            object.__setattr__(self, 'shared_mem_dynamic_size', 122880)
        if '950' not in self.arch:
            raise ValueError('Only Ascend950/A5 targets are supported')
        if self.compile_mode != 'simt_only' or self.num_ctas != 1:
            raise ValueError('Only single-CTA simt_only kernels are supported')
        if self.num_warps <= 0 or self.num_warps & (self.num_warps - 1):
            raise ValueError('num_warps must be a positive power of two')
        if self.warp_size != 32 or (self.shared_mem_dynamic_size is not None
                                    and self.shared_mem_dynamic_size < 0):
            raise ValueError('Invalid SIMT resource configuration')
        if not self.enable_fp_fusion:
            object.__setattr__(self, 'disable_fma', True)

    def hash(self):
        return hashlib.sha256(repr(asdict(self)).encode()).hexdigest()


def _get_dump_paths(hash_key, src_path, dst_path):
    dump_manager = get_dump_manager(hash_key)
    return (dump_manager._make_path(os.path.basename(src_path)),
            dump_manager._make_path(os.path.basename(dst_path)))


def _get_save_temps_dir(hash_key):
    return tempfile.mkdtemp(prefix='bishengir-',
                            dir=get_dump_manager(hash_key).cache_dir)


def _save_npuir_debug_output(stdout_bytes, stderr_bytes, tmpdir,
                             metadata_hash):
    stdout = (stdout_bytes or b'').decode('utf-8', errors='replace')
    stderr = (stderr_bytes or b'').decode('utf-8', errors='replace')
    root = Path(tmpdir)
    (root / 'stdout.log').write_text(stdout)
    (root / 'stderr.log').write_text(stderr)
    combined = stdout + stderr or 'No output captured.'
    (root / 'kernel.npuir.mlir').write_text(combined)
    get_dump_manager(metadata_hash).put(combined,
                                        'kernel.npuir.mlir',
                                        binary=False)


def ttir_to_npubin(mod, metadata, opt):
    _export_program_grid_metadata(mod, metadata, require_row_contract=True)
    ttir_code = _serialize_module(mod, opt)
    metadata = _parse_ttir_metadata(ttir_code, metadata)
    _finalize_program_launch_policy(metadata, opt)

    debug_hash = metadata.get('hash') or hashlib.sha256(
        (ttir_code + opt.hash()).encode()).hexdigest()
    workspace = nullcontext(_get_save_temps_dir(
        debug_hash)) if opt.debug else tempfile.TemporaryDirectory()
    with workspace as tmpdir:
        # prepare input
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        Path(src_path).write_text(ttir_code)
        # prepare output
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel.o")
        # build compile options
        _compile_option_list = get_common_bishengir_compile_options(metadata)
        if opt.is_pure_simt:
            if _is_ascend_sanitizer_enabled():
                _compile_option_list += ["--enable-sanitizer=true"]
            if not _is_debug_line_info_disabled():
                _compile_option_list += ["--enable-debug-info=true"]
            if _enable_msdebug():
                _compile_option_list += ["--enable-debug-variables=true"]
            _compile_option_list += ["--enable-hivm-compile=false"]
            _compile_option_list += ["--enable-triton-ir-compile"]
            _compile_option_list += ["--pure-simt"]
            _compile_option_list += [f"--num-warps={opt.num_warps}"]
            _compile_option_list += [f"--threads-per-warp={opt.warp_size}"]
            if opt.simt_optimization_mode != 0000000:
                _compile_option_list += [
                    f"--simt-optimization-mode={opt.simt_optimization_mode}"
                ]
            _compile_option_list += [
                f"--simt-stack-limit={get_simt_stack_limit()}"
            ]
            if opt.shared_mem_dynamic_size is not None:
                _compile_option_list += [
                    f"--shared-mem-dynamic-size={opt.shared_mem_dynamic_size}"
                ]
            if opt.disable_fma:
                _compile_option_list += ["--disable-fma"]

            bisheng_options = metadata["bisheng_options"]
            if bisheng_options is not None:
                _compile_option_list += [
                    f"--append-bisheng-options={bisheng_options}"
                ]

            if _is_auto_map_parallel_blocks_enabled() and not metadata.get(
                    "row_coalescing_applied", False):
                _compile_option_list += ["--enable-auto-blockify-loop"]
                if opt.superblock_factor > 1:
                    _compile_option_list += [
                        f"--super-block-factor={opt.superblock_factor}"
                    ]
        if opt.debug:
            _compile_option_list += ["--mlir-print-ir-after-failure"]
        npu_compiler_path, env = _get_npucompiler_path()
        cmd_list = ([npu_compiler_path, src_path] + _compile_option_list +
                    ["-o", bin_file])
        if opt.debug:
            Path(tmpdir, 'command.sh').write_text('#!/bin/sh\nset -eu\n' +
                                                  shlex.join(cmd_list) + '\n')
        if opt.debug or knobs.autotuning.print:
            get_dump_manager(debug_hash).put(ttir_code,
                                             os.path.basename(src_path),
                                             binary=False)
            print_cmd_list = cmd_list.copy()
            print_cmd_list[1], print_cmd_list[-1] = _get_dump_paths(
                debug_hash, src_path, bin_file)
            print(f"[DEBUG] cmd_list: {shlex.join(print_cmd_list)}",
                  flush=True)
        try:
            ret = subprocess.run(cmd_list,
                                 env=env,
                                 capture_output=True,
                                 check=True)
        except subprocess.CalledProcessError as error:
            if opt.debug:
                _save_npuir_debug_output(error.stdout, error.stderr, tmpdir,
                                         debug_hash)
            raise
        if opt.debug:
            _save_npuir_debug_output(ret.stdout, ret.stderr, tmpdir,
                                     debug_hash)
        if not Path(bin_path).exists():
            error_msg = (ret.stderr or b'').decode('utf-8', errors='replace')
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list,
                                                ret.stdout, ret.stderr)
        return Path(bin_path).read_bytes()


def get_simt_stack_limit():
    # simt_stack_limit resolution precedence:
    #  1. torch_npu's acl_default.json "StackSize":{"simt_stack_size":N}.
    #  2. The kernel-time default simt_stack_limit=1152.
    _simt_stack_limit = 1152
    try:
        import torch_npu
        torch_npu_basic_path = os.path.dirname(torch_npu.__file__)
        _acl_cfg_path = os.path.join(torch_npu_basic_path, "acl_default.json")
        with open(_acl_cfg_path, "r") as f:
            _acl_cfg = json.load(f)
        _cfg_stack = _acl_cfg.get("StackSize", {}).get("simt_stack_size", None)
        if isinstance(
                _cfg_stack,
                int) and not isinstance(_cfg_stack, bool) and _cfg_stack > 0:
            _simt_stack_limit = _cfg_stack
    except Exception as e:
        print(f"[DEBUG] read acl_default.json failed: {e}")
    return _simt_stack_limit


class AscendBackend(BaseBackend):
    binary_ext = 'npubin'

    @staticmethod
    def supports_target(target):
        return target.backend == 'npu' and '950' in str(target.arch)

    def parse_options(self, opts):
        args = {
            k: v
            for k, v in opts.items() if k in AscendOptions.__dataclass_fields__
            and AscendOptions.__dataclass_fields__[k].init
        }
        args['arch'] = self.target.arch
        device = NPUUtils()
        args['num_aicore'] = device.get_aicore_num()
        args['num_vectorcore'] = device.get_aivector_core_num()
        return AscendOptions(**args)

    def pack_metadata(self, metadata):
        return metadata._asdict()

    def get_codegen_implementation(self, options):
        return {'min_dot_size': lambda lhs, rhs: (1, 1, 1)}

    def get_module_map(self):
        return {}

    def load_dialects(self, context):
        ir.load_dialects(context)

    def hash(self):
        path, env = _get_npucompiler_path()
        version = subprocess.check_output([path, '--version'],
                                          env=env,
                                          stderr=subprocess.STDOUT)
        utils_source = Path(__file__).with_name('utils.py').read_bytes()
        return hashlib.sha256(
            Path(__file__).read_bytes() + utils_source + path.encode() +
            version).hexdigest()

    def add_stages(self, stages, options, language):
        stages['ttir'] = lambda src, metadata: make_ttir(
            src, metadata, options)
        stages['npubin'] = lambda src, metadata: ttir_to_npubin(
            src, metadata, options)
