# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted-side child program of ``solar.frontend.dispatcher`` (isolation-23,
round 7).

This file IS the trusted comparator program.  It is executed FROM ITS INSTALLED
PACKAGE PATH (``python -I -S <this file> <phase> <meta.json>``) — never from a
copy inside the artifact tree.  The PARENT hashes this file and the stdlib/numpy
files it imports before launching every phase and again after the child exits;
a mismatch is ``trusted_program_tampered``.  The hash this program reports about
itself is recorded by the parent but NOT trusted; the verdict digest is computed
by the parent over parent-hashed artifacts.

Every phase is a fresh interpreter in its own SESSION (``start_new_session``)
and its own private directory (mode 0700).  After a phase that executed
generated code the parent kills the whole session, verifies on ``/proc`` that
no descendant survived (``candidate_descendants_alive`` otherwise) and only then
continues.  ``sys.path`` is exactly the trusted roots the parent computed ONCE
and passed down — never the candidate's or adapter's directory, cwd or
PYTHONPATH.

Round 11 (CI-FAIL-27 root cause): the census' descent rule is stated precisely in
``attributed_submodule`` -- a submodule whose ``__file__`` is the parent's verified
file (pybind11 >= 3.0 copies it onto ``def_submodule`` modules) names no file of its
own -- and every refusal records a diagnostic (``census_diagnostic``: the failed
condition and the parent's census state) for the private receipt.
Round 10 (result-boundary-29 / REFERENCE-27): everything this program or
the generated code writes for the parent (``report.json``, result archives, the
staged receipt) is a plain regular file the parent INGESTS through a hardened
open (``O_NOFOLLOW|O_NONBLOCK``, fstat: regular file, one link, same uid, size
cap, directory-entry identity, bounded read under the remaining dispatch
budget) BEFORE it unlocks its private evidence root; the report is parsed
against an allowlist schema -- a fifo, a link or an unknown key is never an
admission.  The compare verdict names pass/fail per output; its error
statistics are private until the dispatch ends.

Phases, in the order the parent runs them:

* ``verify``    — the facade verifier (``solar.translate.verify``) and, on a
                  numerical pass, the sealed workload receipt into a staging path.
                  Executes adapter and candidate code: untrusted, diagnostic; the
                  facade's runner CONSUMES the parent's trusted roots.
* ``inputs``    — executes adapter code: writes the input snapshot from
                  ``get_inputs()`` and the declared entry metadata.  The parent
                  reads the bytes ONCE and keeps them: that is the frozen state.
* ``candidate`` — runs in ITS OWN PROCESS (round 9): a fresh 0700 working root
                  (cwd) holding only this launch's inputs — parent-owned ``.npy``
                  files, mode 0400, mapped READ-ONLY — and the candidate source;
                  no path to any other attempt, evidence or reference (those live
                  under the parent-private evidence root, locked to mode 0000
                  while this process is alive).  An in-place write faults here
                  (``candidate_mutated_inputs``).  Outputs are a result file the
                  PARENT reads after this process EXITED; completion is
                  established by the parent from the exit status and the result
                  files — there is no nonce, and nothing this process reports
                  about its inputs is admissible.
* ``reference`` — AFTER the candidate session is dead and verified gone, in a
                  directory created afterwards: executes the ORIGINAL requested
                  source (hash-checked) under the import guard; audits provenance
                  afterwards.
* ``compare``   — never runs generated code: reads the parent-controlled copies
                  of the three artifacts by hash and compares under the approved
                  rtol/atol plus a floor derived ONLY from the trusted outputs'
                  dtypes; ``success`` is computed once from the comparison.

Provenance (module-audit-26): a module counts as trusted only if it was
registered THROUGH the import system by the guard (origin realpath under a
trusted root, content fingerprint unchanged since import) or existed, unchanged
and un-replaced, before generated code ran.  Attributes generated code writes on
a module object (``__spec__``, ``__file__``) establish nothing.
"""
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import types

_OPEN = open
_TYPE = type
_DICT = dict
_LIST = list
_ISABS = os.path.isabs
_OBJECT_GETATTR = object.__getattribute__
_MODULE_TYPE = types.ModuleType


def _self_sha256():
    h = hashlib.sha256()
    with _OPEN(__file__, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


if __name__ == "__main__":
    # The program runs ONLY when executed as the trusted child.  A multiprocessing spawn /
    # forkserver worker started by the facade's isolated verification re-imports its parent's
    # main module as ``__mp_main__`` (target pipeline 77147099, layer 2); that import must define
    # the helpers below and do nothing else -- in particular never reach ``done()`` -> ``sys.exit``.
    PHASE = sys.argv[1] if len(sys.argv) > 1 else "?"
    META = json.load(_OPEN(sys.argv[2])) if len(sys.argv) > 2 else {}
    PROGRAM_SHA256_AT_START = _self_sha256()
    REPORT = {"ok": False, "phase": PHASE, "program_path": os.path.realpath(__file__),
              "program_sha256_at_start": PROGRAM_SHA256_AT_START,
              "argv": _LIST(sys.argv), "cwd": os.getcwd(), "pid": os.getpid(),
              "sid": os.getsid(0) if hasattr(os, "getsid") else None,
              "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None,
              "flags": {"isolated": bool(sys.flags.isolated), "no_site": bool(sys.flags.no_site),
                        "ignore_environment": bool(sys.flags.ignore_environment)}}
else:
    # imported, not executed: nothing above ran and nothing below will (no argv, no meta, no
    # report, no exit); the name records the fact for the parent and the tests
    IMPORTED_NOT_EXECUTED = __name__


def done(code, **kw):
    REPORT.update(kw)
    REPORT["program_sha256_at_end"] = _self_sha256()
    # The facade's isolated verification may have started helper processes in THIS session
    # (multiprocessing forkserver + resource tracker).  They are reaped BEFORE this leader
    # exits, bounded, so that none of them outlives the phase: the parent's session boundary
    # refuses any survivor it samples the instant the leader is gone, and a helper that was
    # merely slow to notice our exit was refused as candidate_descendants_alive on a loaded
    # runner.  Nothing is imported here: only a module already loaded by the phase is used.
    _V = sys.modules.get("solar.einsum_lib.verify")
    if _V is not None and hasattr(_V, "shutdown_isolation"):
        try:
            _V.shutdown_isolation()
        except Exception:  # noqa: BLE001 - the exit status and the report must still be produced
            pass
    tmp = META["report_path"] + ".tmp"
    with _OPEN(tmp, "w") as fh:
        json.dump(REPORT, fh, default=str)
    os.replace(tmp, META["report_path"])
    sys.exit(code)


def sha(path):
    h = hashlib.sha256()
    with _OPEN(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def real(p):
    return os.path.realpath(p)


if __name__ == "__main__":  # program state, only under direct execution (see the first guard)
    ROOTS = [real(r) for r in META.get("trusted_roots", [])]
    _roots_digest = hashlib.sha256(json.dumps(_LIST(META.get("trusted_roots", [])), sort_keys=True,
                                              separators=(",", ":")).encode("utf-8")).hexdigest()
    if META.get("trusted_roots_sha256") and META["trusted_roots_sha256"] != _roots_digest:
        done(14, error="trusted roots do not hash to the parent's digest")


def under_root(p):
    rp = real(p)
    return any(rp == r or rp.startswith(r + os.sep) for r in ROOTS)


class Violation(ImportError):
    pass


_MAX_HASH_BYTES = 256 << 20


def fingerprint(path):
    """Content identity of a file: sha256 of the bytes (size, mtime and inode
    for files above 256 MiB, where re-hashing on every audit is prohibitive)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    if st.st_size <= _MAX_HASH_BYTES:
        return "sha256:" + sha(path)
    return "stat:%d:%d:%d" % (st.st_size, st.st_mtime_ns, st.st_ino)


class Guard(importlib.abc.MetaPathFinder):
    """The ONLY admitted way for a module to enter ``sys.modules`` while
    generated code runs: records every module it resolves (origin realpath and
    content fingerprint at import time) and refuses origins outside the
    trusted roots."""

    def __init__(self):
        self.loaded = {}

    def find_spec(self, name, path=None, target=None):
        for finder, kind in ((importlib.machinery.BuiltinImporter, "builtin"),
                             (importlib.machinery.FrozenImporter, "frozen")):
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                self.loaded[name] = {"kind": kind, "origin": None, "fingerprint": None}
                return spec
        spec = importlib.machinery.PathFinder.find_spec(name, path, target)
        if spec is None:
            return None
        origins = []
        if spec.origin and spec.origin not in ("built-in", "frozen"):
            origins.append(spec.origin)
        if spec.submodule_search_locations:
            origins += _LIST(spec.submodule_search_locations)
        bad = [o for o in origins if not under_root(o)]
        if bad:
            raise Violation("reference_dependency_untrusted: import of %r resolves outside the "
                            "trusted roots: %s" % (name, bad))
        if spec.origin and spec.origin not in ("built-in", "frozen"):
            rp = real(spec.origin)
            self.loaded[name] = {"kind": "file", "origin": rp, "fingerprint": fingerprint(rp)}
        else:
            self.loaded[name] = {"kind": "namespace", "origin": None, "fingerprint": None,
                                 "locations": [real(l) for l in (spec.submodule_search_locations or [])]}
        return spec


GUARD = Guard()


def set_trusted_path(prepend=()):
    """``sys.path`` := the interpreter's own stdlib entries (all under the
    prefix roots) + the trusted roots the parent passed, in order.  With
    ``-I -S`` nothing else was ever on it."""
    keep = [p for p in sys.path if p and under_root(p)]
    extra = [p for p in META.get("sys_path", []) if p and under_root(p)]
    pre = [p for p in prepend if p and under_root(p)]
    seen, out = set(), []
    for p in pre + keep + extra:
        if p not in seen:
            seen.add(p)
            out.append(p)
    sys.path[:] = out
    importlib.invalidate_caches()
    return _LIST(sys.path)


def arm_guard(prepend=()):
    snapshot = set_trusted_path(prepend)
    sys.meta_path.insert(0, GUARD)
    return snapshot


def _module_attrs(mod):
    """{attribute name: id(object)} for the attributes of *mod* that are module
    objects (the submodules an extension or package exposes), read from the
    module's own ``__dict__`` without triggering descriptors."""
    try:
        d = _OBJECT_GETATTR(mod, "__dict__")
    except Exception:
        return {}
    return {k: id(v) for k, v in _LIST(d.items()) if isinstance(v, _MODULE_TYPE)}


def snapshot_modules():
    """Identity + content of every module loaded so far (taken immediately
    before generated code runs): the baseline the post-execution audit holds
    the module table to.  Also the module-typed attribute names of every
    module, so that a submodule registered later can be checked to have been
    exposed by its parent BEFORE generated code ran."""
    snap = {}
    for name, mod in _LIST(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        origin = real(f) if (f and _ISABS(f)) else None
        snap[name] = {"id": id(mod), "origin": origin, "fingerprint": fingerprint(origin) if origin else None,
                      "module_attrs": _module_attrs(mod)}
    return snap


_EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)


def _parent_kind(origin):
    return "extension" if origin.endswith(_EXTENSION_SUFFIXES) else "package"


def _own_file_facts(mod):
    f = getattr(mod, "__file__", None)
    spec = getattr(mod, "__spec__", None)
    return {"name_attr": getattr(mod, "__name__", None),
            "own_file": f if isinstance(f, str) else (None if f is None else repr(f)[:200]),
            "own_file_absolute": bool(isinstance(f, str) and _ISABS(f)),
            "own_spec_name": getattr(spec, "name", None) if spec is not None else None,
            "own_spec_origin": getattr(spec, "origin", None) if spec is not None else None,
            "type": _TYPE(mod).__name__}


def attributed_submodule(name, mod, before, depth=0):
    """Manager ruling (round 10, CI finding on torch._C.*; stated precisely in round
    11 after the runner traces): trust descends from a census-verified FILE module
    (an extension such as ``torch._C`` whose .so is hashed under a trusted root, or
    a trusted package file) to a submodule ONLY when
      (a) the parent IS that verified file: it existed before generated code ran
          with its file unchanged since the snapshot, or the guard imported it
          during execution from under a trusted root with its file unchanged;
      (b) the name is the parent's name plus one dotted component;
      (c) the object in ``sys.modules`` IS the very object the parent exposes as a
          module attribute (the extension may expose it under another attribute
          name: ``torch._C._dynamo.autograd_compiler`` is ``compiled_autograd``),
          and -- when the parent existed before generated code ran -- the parent
          exposed that very object before too (a module planted into
          ``sys.modules`` AND set on a pre-existing parent by generated code fails);
      (d) it names no file of its own: no ``__spec__`` origin other than None,
          "built-in" or "frozen", and either no absolute ``__file__`` (``torch.ops``
          and ``torch.classes`` carry the relative markers "_ops.py" / "_classes.py",
          which are not files) or a ``__file__`` that IS the parent's verified file.
          pybind11 >= 3.0 (#5584) copies the parent's ``__file__`` onto every
          ``def_submodule`` module created after the parent finished initialising;
          torch >= 2.9 therefore registers ``torch._C._distributed_c10d``,
          ``_distributed_rpc``, ``_distributed_autograd`` and ``_autograd`` (created
          by Python-called init functions) with ``__file__ == torch._C.__file__``
          while the submodules created inside the extension's own init carry none.
          A ``__file__`` naming any other path refuses.
    A chain of file-less parents is followed (depth <= 4) up to a verified file
    module.  -> ((parent_name, kind, attribute_name, parent_origin) or None, diag):
    the diagnostic record evaluates every condition and names the failing one in
    ``failed`` (or the admitting ``basis``); it goes to the private receipt."""
    diag = {"module": name, "depth": depth}
    diag.update(_own_file_facts(mod))

    def refuse(code, **more):
        diag.update(more)
        diag["failed"] = code
        return None, diag

    if depth > 4:
        return refuse("chain_too_deep")
    if "." not in name:
        return refuse("top_level_name")
    own_file = getattr(mod, "__file__", None)
    own_abs = real(own_file) if (isinstance(own_file, str) and _ISABS(own_file)) else None
    own_spec = getattr(mod, "__spec__", None)
    if own_spec is not None and getattr(own_spec, "origin", None) not in (None, "built-in", "frozen"):
        return refuse("own_spec_origin")                        # (d): a spec naming an origin of its own
    parent_name, comp = name.rsplit(".", 1)
    diag["parent"] = parent_name
    parent = sys.modules.get(parent_name)
    if parent is None:
        return refuse("parent_not_in_sys_modules")
    if getattr(mod, "__name__", None) != name:
        return refuse("name_mismatch")                          # (b)
    exposed_as = [k for k, v in _module_attrs(parent).items() if v == id(mod)]
    diag["exposed_as"] = ",".join(exposed_as)
    if not exposed_as:
        return refuse("not_identical_to_a_parent_attribute")    # (c): identity with the parent's attribute
    attr = comp if comp in exposed_as else exposed_as[0]
    pb = before.get(parent_name)
    rec = GUARD.loaded.get(parent_name)
    diag["parent_in_snapshot"] = pb is not None
    diag["parent_guard_kind"] = rec.get("kind") if rec else None
    diag["parent_guard_origin"] = rec.get("origin") if rec else None
    if pb is not None:
        if pb["id"] != id(parent):
            return refuse("parent_replaced")
        at_snapshot = (pb.get("module_attrs") or {}).get(attr) == id(mod)
        diag["parent_exposed_this_object_at_snapshot"] = at_snapshot
        if not at_snapshot:
            return refuse("attribute_absent_at_snapshot")       # (c) for a pre-existing parent
        if pb.get("origin"):
            diag["parent_origin"] = pb["origin"]
            diag["parent_under_root"] = under_root(pb["origin"])
            diag["parent_fingerprint_unchanged"] = fingerprint(pb["origin"]) == pb["fingerprint"]
            if not (diag["parent_under_root"] and diag["parent_fingerprint_unchanged"]):
                return refuse("parent_file_not_verified")       # (a)
            origin, kind = pb["origin"], _parent_kind(pb["origin"])
            basis = "pre-existing parent: attribute present at snapshot, parent file unchanged"
        else:
            up, up_diag = attributed_submodule(parent_name, parent, before, depth + 1)
            if up is None:
                return refuse("file_less_parent_unattributed", parent_failed=up_diag.get("failed"))
            origin, kind = up[3], up[1]
            basis = "chain through file-less parent %s" % parent_name
    elif rec is not None:
        has_origin = bool(rec.get("origin"))
        diag["parent_origin"] = rec.get("origin")
        diag["parent_under_root"] = has_origin and under_root(rec["origin"])
        diag["parent_fingerprint_unchanged"] = has_origin and fingerprint(rec["origin"]) == rec["fingerprint"]
        if not (rec.get("kind") == "file" and diag["parent_under_root"] and diag["parent_fingerprint_unchanged"]):
            return refuse("parent_guard_record_not_a_verified_file")   # (a)
        origin, kind = rec["origin"], _parent_kind(rec["origin"])
        basis = "parent imported through the guard during execution, file unchanged"
    else:
        up, up_diag = attributed_submodule(parent_name, parent, before, depth + 1)   # file-less parent created during execution
        if up is None:
            return refuse("file_less_parent_unattributed", parent_failed=up_diag.get("failed"))
        origin, kind = up[3], up[1]
        basis = "chain through file-less parent %s" % parent_name
    if own_abs is not None:
        diag["own_file_is_parent_file"] = own_abs == origin
        if own_abs != origin:
            return refuse("own_file_not_the_parent_file", parent_origin=origin)   # (d)
        basis += "; __file__ is the parent's verified file (pybind11 >= 3.0 def_submodule)"
    diag["basis"] = basis
    return (parent_name, kind, attr, origin), diag


def _trusted_file_record(rf, before):
    """The import record (guard or pre-execution snapshot) whose origin is the
    real file ``rf`` and whose fingerprint still matches; None otherwise."""
    if not rf or not under_root(rf):
        return None
    for rec in GUARD.loaded.values():
        if rec.get("origin") == rf and rec.get("fingerprint") == fingerprint(rf):
            return rec
    for b in before.values():
        if b.get("origin") == rf and b.get("fingerprint") == fingerprint(rf):
            return b
    return None


ATTRIBUTION = {"extension_submodule_of": {}, "package_submodule_of": {}, "attribution_basis": {}}
CENSUS_DIAGNOSTIC = {}        # refused module -> the diagnostic record (private receipt only)


def audit_after(path_snapshot, before):
    problems = []
    if not sys.meta_path or sys.meta_path[0] is not GUARD:
        problems.append("import guard removed or displaced from sys.meta_path[0]")
    before_ids = set(b["id"] for b in before.values())
    for name, mod in _LIST(sys.modules.items()):
        if name in ("__main__", "__mp_main__"):
            continue          # this program (torch.multiprocessing aliases it)
        f = getattr(mod, "__file__", None)
        rf = real(f) if (f and _ISABS(f)) else None
        b = before.get(name)
        if b is not None:
            if id(mod) != b["id"]:
                problems.append("module %s replaced in sys.modules during execution" % name)
            elif b["origin"] and fingerprint(b["origin"]) != b["fingerprint"]:
                problems.append("module %s: file %s changed since the pre-execution snapshot" % (name, b["origin"]))
            continue
        if id(mod) in before_ids:
            continue          # an alias of a module that existed before execution
        spec = getattr(mod, "__spec__", None)
        canonical = getattr(spec, "name", None)
        if (canonical and canonical != name and sys.modules.get(canonical) is mod
                and canonical in GUARD.loaded):
            # an ALIAS registration of a module the guard imported now (numba
            # registers ``pkg.mod.np`` -> numpy, Cython registers ``_ext`` ->
            # ``scipy.sparse._ext``, numpy 1.26 maps ``numpy._core.x`` onto
            # ``numpy.core.x``): the canonical name holds this very object and is
            # audited under its own name.  A forged __spec__ naming an existing
            # module fails this test because that name holds a different object.
            continue
        rec = GUARD.loaded.get(name)
        # a submodule object exposed by a census-verified file module (the torch._C.*
        # names torch's Python code registers after torch._C loaded; torch.ops /
        # torch.classes / torch._VF created by the torch package): accepted ONLY
        # through attributed_submodule (conditions (a)-(d)); a top-level synthetic
        # name or a planted entry never is.  Every refusal leaves its diagnostic
        # (which condition failed, the parent's census state) for the private receipt,
        # so that a refusal on another host is self-explaining.
        attribution, diag = (attributed_submodule(name, mod, before) if (rec is None or rf is None)
                             else (None, None))
        if attribution is not None:
            parent_name, kind, attr, _origin = attribution
            ATTRIBUTION[kind + "_submodule_of"][name] = parent_name if attr == name.rsplit(".", 1)[1] \
                else "%s (exposed as attribute %s)" % (parent_name, attr)
            ATTRIBUTION["attribution_basis"][name] = diag.get("basis")
        if rec is None:
            if attribution is not None:
                continue
            CENSUS_DIAGNOSTIC[name] = diag
            problems.append("module %s registered without passing through the import system [census: %s]"
                            % (name, diag.get("failed")))
            continue
        if rec["kind"] in ("builtin", "frozen"):
            if f:
                problems.append("%s module %s carries a file attribute %r" % (rec["kind"], name, f))
            continue
        if rec["kind"] == "namespace":
            if f:
                problems.append("namespace package %s carries a file attribute %r" % (name, f))
            continue
        if rf is None:
            if attribution is not None:
                continue
            if diag is not None:
                CENSUS_DIAGNOSTIC[name] = diag
            problems.append("module %s imported from %s has no absolute file attribute afterwards (%r)" % (name, rec["origin"], f))
            continue
        if rf != rec["origin"]:
            # the code that ran under this name came from the guard-imported origin
            # (must be unchanged); the attribute it now carries must itself be a
            # real, unchanged trusted file (numpy's ``_core/_multiarray_umath.py``
            # shim copies every global of the extension module, __file__ and
            # __spec__ included).  A claimed file that is not such a file refuses.
            claimed = _trusted_file_record(rf, before)
            if fingerprint(rec["origin"]) != rec["fingerprint"]:
                problems.append("module %s: file %s changed after import" % (name, rec["origin"]))
            elif claimed is None:
                problems.append("module %s: __file__ %r is not the imported origin %s and not an unchanged "
                                "trusted file" % (name, f, rec["origin"]))
            continue
        if not under_root(rf):
            problems.append("module %s loaded from %s" % (name, rf))
        elif fingerprint(rf) != rec["fingerprint"]:
            problems.append("module %s: file %s changed after import" % (name, rf))
    for p in sys.path:
        if p not in path_snapshot:
            problems.append("sys.path entry added during execution: %s" % p)
    return problems


def check_inputs():
    if sha(META["inputs_path"]) != META["inputs_sha256"]:
        done(9, error="trusted input snapshot hash mismatch before execution")


if __name__ == "__main__":  # the phase program; inert on import (see the first guard)
    try:
        if PHASE == "verify":
            # the facade verifier + receipt emission, in a child so that no adapter
            # or candidate code ever executes in the dispatcher's own process
            set_trusted_path(prepend=[META["checkout_root"]])
            import solar.translate as _T
            import solar.translate._runner as _R
            from solar.translate.receipts import ReceiptRefusal, emit_workload_receipt
            # ONE trusted-roots computation: the facade's runner consumes the parent's
            # list instead of recomputing its own (CI-FAIL-25)
            _R.use_trusted_roots(META["trusted_roots"], META.get("trusted_roots_sha256"))
            for label, path, want in (("candidate", META["dsl_path"], META["dsl_sha256"]),
                                      ("model", META["model_path"], META.get("model_sha256"))):
                if want and sha(path) != want:
                    done(4, error="%s bytes do not match the bound sha256" % label)
            kw = {}
            if META.get("rtol") is not None and META.get("atol") is not None:
                kw = {"rtol": float(META["rtol"]), "atol": float(META["atol"])}
            try:
                res = _T.verify(META["model_path"], META["dsl_path"], source_type=META["source_type"],
                                mem_budget_gb=META.get("mem_budget_gb"), timeout=META.get("verify_timeout"),
                                quiet=True, log=lambda *_: None, **kw)
            except BaseException as exc:  # noqa: BLE001 - reported with its type for classification
                done(12, error="verify raised %s: %s" % (_TYPE(exc).__name__, exc),
                     exception={"type": _TYPE(exc).__name__, "mro": [c.__name__ for c in _TYPE(exc).__mro__],
                                "message": str(exc)[:800]})
            ev = getattr(res, "execution", None)
            summary = {"status": res.status, "message": res.message, "max_abs_error": res.max_abs_error,
                       "vacuous_outputs": _LIST(res.vacuous_outputs or []),
                       "trusted_roots_consumed_by_runner": _R.trusted_roots_in_use() is not None,
                       "execution": None if ev is None else {
                           "source_sha256": ev.source_sha256, "reference_sha256": ev.reference_sha256,
                           "local_dependencies": [_LIST(d) for d in (ev.local_dependencies or ())],
                           "unsupported_dependencies": _LIST(ev.unsupported_dependencies or ()),
                           "min_tol": float(ev.min_tol or 0.0)}}
            if META.get("receipt_path") and META.get("definition_dir") and META.get("dtype") and res.status == "pass":
                try:
                    out = emit_workload_receipt(res, source_path=META["model_path"], dsl_path=META["dsl_path"],
                                                definition_dir=META["definition_dir"], dtype=META["dtype"],
                                                out_path=META["receipt_path"], workload_idx=int(META.get("workload_idx") or 0))
                    summary["receipt_path"] = str(out)
                    summary["receipt_sha256"] = sha(str(out))
                except ReceiptRefusal as exc:
                    summary["receipt_refused"] = str(exc)
                except BaseException as exc:  # noqa: BLE001
                    summary["receipt_error"] = {"type": _TYPE(exc).__name__, "message": str(exc)[:800]}
            done(0, ok=True, verify=summary)
        elif PHASE == "inputs":
            # executes adapter code: the input snapshot and the declared reference
            # metadata, nothing else; the parent keeps the bytes
            set_trusted_path(prepend=[META["checkout_root"]])
            import numpy as np
            adir = os.path.dirname(real(META["adapter_path"]))
            sys.path.insert(0, adir)
            spec = importlib.util.spec_from_file_location("_solar_dispatch_adapter_inputs", META["adapter_path"])
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            raw = mod.get_inputs()
            if not isinstance(raw, dict) or not raw:
                done(6, error="get_inputs() must return a non-empty dict {name: ndarray}")
            inputs = {}
            for k, v in raw.items():
                arr = np.asarray(v)
                if np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float64)
                inputs[str(k)] = np.ascontiguousarray(arr)
            np.savez(META["inputs_path"], **inputs)
            entry = getattr(mod, "REFERENCE_ENTRY", None)
            args = getattr(mod, "REFERENCE_ARGS", None)
            done(0, ok=True, inputs_sha256=sha(META["inputs_path"]), names=_LIST(inputs),
                 shapes={k: _LIST(v.shape) for k, v in inputs.items()},
                 dtypes={k: str(v.dtype) for k, v in inputs.items()},
                 entry=entry if isinstance(entry, str) else None,
                 args=[str(a) for a in args] if isinstance(args, (list, tuple)) else None)
        elif PHASE == "reference":
            snapshot = arm_guard()
            check_inputs()
            import inspect
            import numpy as np
            src = _OPEN(META["source_path"], "rb").read()
            digest = hashlib.sha256(src).hexdigest()
            REPORT["source_sha256"] = digest
            if digest != META["source_sha256"]:
                done(4, error="source bytes do not match the requested sha256")
            inputs = dict(np.load(META["inputs_path"]))
            before = snapshot_modules()
            ns = {"__name__": "__trusted_reference__", "__file__": META["source_path"]}
            try:
                exec(compile(src, META["source_path"], "exec"), ns)
            except Violation as exc:
                done(8, error=str(exc), violations=[str(exc)])
            defined = {}
            for k, v in ns.items():
                if isinstance(v, types.FunctionType):
                    try:
                        if inspect.unwrap(v).__code__.co_filename == META["source_path"]:
                            defined[k] = v
                    except Exception:
                        pass
            entry = META.get("entry")
            if entry:
                fn = ns.get(entry)
                if not callable(fn):
                    done(6, error="declared reference entry %r is not a callable in the source" % entry)
            elif len(defined) == 1:
                fn = next(iter(defined.values()))
            else:
                done(6, error="reference entry ambiguous: %s; the adapter must declare REFERENCE_ENTRY"
                     % sorted(defined))
            torch = sys.modules.get("torch")

            def to_fw(a):
                return torch.from_numpy(np.ascontiguousarray(a)) if torch is not None else a

            def to_np(v):
                if torch is not None and isinstance(v, torch.Tensor):
                    return v.detach().cpu().numpy()
                return np.asarray(v)
            params = inspect.signature(fn).parameters
            try:
                if all(n in params for n in inputs):
                    out = fn(**{k: to_fw(v) for k, v in inputs.items()})
                else:
                    out = fn(*[to_fw(inputs[n]) for n in META["arg_order"]])
            except Violation as exc:
                done(8, error=str(exc), violations=[str(exc)])
            names = META["output_names"]
            if isinstance(out, dict):
                outs = {str(k): to_np(v) for k, v in out.items()}
            elif isinstance(out, (list, tuple)):
                if len(out) != len(names):
                    done(7, error="reference returned %d outputs, %d expected" % (len(out), len(names)))
                outs = {n: to_np(v) for n, v in zip(names, out)}
            else:
                outs = {names[0]: to_np(out)}
            problems = audit_after(snapshot, before)
            if problems:
                done(8, error="reference_dependency_untrusted: " + "; ".join(problems), violations=problems,
                     extension_submodule_of=dict(ATTRIBUTION["extension_submodule_of"]),
                     package_submodule_of=dict(ATTRIBUTION["package_submodule_of"]),
                     attribution_basis=dict(ATTRIBUTION["attribution_basis"]),
                     census_diagnostic=dict(CENSUS_DIAGNOSTIC))
            np.savez(META["outputs_path"], **outs)
            if sha(META["inputs_path"]) != META["inputs_sha256"]:
                done(9, error="trusted input snapshot changed during reference execution")
            done(0, ok=True, entry=getattr(fn, "__name__", str(fn)),
                 framework="torch" if torch is not None else "numpy",
                 outputs_sha256=sha(META["outputs_path"]),
                 output_shapes={k: _LIST(v.shape) for k, v in outs.items()},
                 output_dtypes={k: str(v.dtype) for k, v in outs.items()},
                 sys_path=_LIST(sys.path), guard_loaded=sorted(GUARD.loaded),
                 extension_submodule_of=dict(ATTRIBUTION["extension_submodule_of"]),
                 package_submodule_of=dict(ATTRIBUTION["package_submodule_of"]),
                 attribution_basis=dict(ATTRIBUTION["attribution_basis"]),
                 census_diagnostic=dict(CENSUS_DIAGNOSTIC))
        elif PHASE == "candidate":
            # ONE candidate process per launch (round 9): this process runs the
            # generated code.  Its working root (cwd) holds only this launch's inputs
            # and source; completion is established by the PARENT from the exit
            # status and the result files, never by anything reported from here.
            snapshot = arm_guard(prepend=[META["checkout_root"]])
            import numpy as np
            import solar.einsum_lib.verify as _V
            from solar.einsum_lib.verify import execute_translated
            from solar.einsum_lib.cuda_frontend import _np_dtype_to_solar
            code_bytes = _OPEN(META["candidate_path"], "rb").read()
            if hashlib.sha256(code_bytes).hexdigest() != META["candidate_sha256"]:
                done(4, error="candidate bytes do not match the bound sha256")

            def open_readonly(path, _np=np):
                """A parent-owned .npy (mode 0400) mapped READ-ONLY (PROT_READ, MAP_SHARED):
                the array handed to the candidate is a plain ndarray view with
                ``writeable=False``; an in-place write raises in THIS process and the
                parent's file cannot change through the mapping."""
                mm = _np.load(path, mmap_mode="r")
                arr = mm.view(_np.ndarray)
                arr.flags.writeable = False
                return arr, mm

            def is_readonly_write_fault(exc, _str=str):
                """A write to a read-only mapping surfaces as numpy's ValueError ("output
                array is read-only" / "assignment destination is read-only"), possibly
                wrapped by the execution harness (RuntimeError("run_model() failed: ...")) or
                by numba's typing errors: the exception CHAIN is inspected.  Both outcomes
                (this refusal and a plain execution failure) are scored refusals, so a
                candidate gains nothing by faking the text."""
                seen = 0
                while exc is not None and seen < 8:
                    text = _str(exc).lower()
                    if "read-only" in text or "readonly" in text or "read only" in text:
                        return True
                    exc = exc.__cause__ or exc.__context__
                    seen += 1
                return False

            def run_candidate(inputs, _asarray=np.asarray, _hash=hashlib.sha256, _contig=np.ascontiguousarray,
                              _tobytes=np.ndarray.tobytes):
                """Executes the candidate on the read-only views.  Afterwards the ARRAY
                OBJECTS it handed over are re-hashed with primitives captured before the
                candidate ran (a candidate that re-points an object at a private buffer
                and writes there is caught here) and the identity record says whether the
                candidate rebound the entries.  Both records may only REFUSE: the parent's
                own observation of its 0400 files after the boundary is the purity gate and
                nothing reported from this process is admissible for admission."""
                originals = {k: v for k, v in inputs.items()}

                def fp(v):
                    return _hash(_tobytes(_contig(_asarray(v)))).hexdigest()
                before_fp = {k: fp(v) for k, v in originals.items()}
                shapes = {k: tuple(v.shape) for k, v in inputs.items()}
                dtypes = {k: _np_dtype_to_solar(v.dtype) for k, v in inputs.items()}
                out, used = execute_translated(code_bytes.decode("utf-8"), shapes, input_dtypes=dtypes,
                                               all_inputs=inputs, use_fp64=True,
                                               mem_budget_bytes=META.get("mem_budget_bytes"))
                identity = [k + ": changed in place (candidate-process array object)"
                            for k, v in originals.items() if fp(v) != before_fp[k]]
                if _TYPE(used) is not _DICT:
                    identity.append("<inputs dict replaced by %s>" % _TYPE(used).__name__)
                    used = {}
                for k in originals:
                    if k not in used:
                        identity.append("%s: removed" % k)
                    elif used[k] is not originals[k]:
                        identity.append("%s: replaced" % k)
                for k in used:
                    if k not in originals:
                        identity.append("%s: added" % k)
                return {str(k): _asarray(v) for k, v in out.items()}, identity

            before = snapshot_modules()
            inputs, mappings = {}, {}
            for name, path in META["inputs_npy"].items():
                if sha(path) != META["inputs_npy_sha256"][name]:
                    done(9, error="parent input file hash mismatch before execution: " + name)
                inputs[name], mappings[name] = open_readonly(path)
            try:
                outs, identity = run_candidate(inputs)
            except Violation as exc:
                done(8, error=str(exc), violations=[str(exc)])
            except _V.IsolationUnavailableError as exc:
                # the facade could not start its isolating child here: an environment fact, typed
                done(18, error="isolation_unavailable: %s" % exc,
                     exception={"type": _TYPE(exc).__name__, "mro": [c.__name__ for c in _TYPE(exc).__mro__]})
            except BaseException as exc:  # noqa: BLE001
                if is_readonly_write_fault(exc):
                    done(10, error="candidate_mutated_inputs: the candidate attempted an in-place write to a "
                                   "read-only input (%s: %s)" % (_TYPE(exc).__name__, str(exc)[:300]),
                         fault="readonly_input_write",
                         exception={"type": _TYPE(exc).__name__, "mro": [c.__name__ for c in _TYPE(exc).__mro__]})
                raise
            np.savez(META["outputs_path"], **outs)
            problems = audit_after(snapshot, before)
            REPORT.update(outputs_sha256=sha(META["outputs_path"]),
                          output_shapes={k: _LIST(v.shape) for k, v in outs.items()},
                          output_dtypes={k: str(v.dtype) for k, v in outs.items()},
                          identity_audit=identity, sys_path=_LIST(sys.path), guard_loaded=sorted(GUARD.loaded),
                          extension_submodule_of=dict(ATTRIBUTION["extension_submodule_of"]),
                          package_submodule_of=dict(ATTRIBUTION["package_submodule_of"]),
                          attribution_basis=dict(ATTRIBUTION["attribution_basis"]),
                          census_diagnostic=dict(CENSUS_DIAGNOSTIC))
            if problems:
                done(8, error="candidate_dependency_untrusted: " + "; ".join(problems), violations=problems)
            for name, path in META["inputs_npy"].items():
                # diagnostic re-check of the files this process was given; the parent's
                # own re-hash after the boundary is the authoritative observation
                if sha(path) != META["inputs_npy_sha256"][name]:
                    done(9, error="parent input file changed during candidate execution: " + name)
            done(0, ok=True)
        elif PHASE == "compare":
            arm_guard()
            import numpy as np
            hashes = {}
            for key in ("inputs", "trusted_outputs", "candidate_outputs"):
                h = sha(META[key + "_path"])
                hashes[key] = h
                if h != META[key + "_sha256"]:
                    done(11, error="artifact %s hash mismatch (%s != %s)" % (key, h, META[key + "_sha256"]))
            trusted = dict(np.load(META["trusted_outputs_path"]))
            cand = dict(np.load(META["candidate_outputs_path"]))
            rtol, atol = float(META["rtol"]), float(META["atol"])
            # the tolerance floor is derived ONLY from the TRUSTED reference's own
            # output dtypes (4 ulp headroom, as the facade does for its reference)
            floor = 0.0
            for arr in trusted.values():
                if np.issubdtype(arr.dtype, np.floating):
                    floor = max(floor, 4.0 * float(np.finfo(arr.dtype).eps))
            rtol_eff, atol_eff = max(rtol, floor), max(atol, floor)

            def canonical_stat(x):
                # round 13 (the review NUMERIC-FEEDBACK-30/D): a NON-FINITE statistic -- finite operands
                # overflow (1e308 - (-1e308)) or cancel (inf - inf) -- is carried as one of three fixed
                # string tokens.  The parent's allowlist parser accepts exactly these tokens and the
                # child's and the parent's digests bind the same representation; a JSON Infinity/NaN
                # literal was dropped by the parser and made the digests (and the public failure class)
                # differ with the hidden value.
                x = float(x)
                if x != x:
                    return "nan"
                if x == float("inf") or x == float("-inf"):
                    return "inf" if x > 0 else "-inf"
                return x
            error = None
            vacuous = []
            max_err = None
            outputs_pass = {}        # pass/fail per output: the only comparison outcome attempt-visible state carries
            per_output = {}          # error statistics: private until the dispatch ends (final receipt only)
            nan_seen = False
            if set(trusted) != set(cand):
                error = "Output key mismatch: candidate has %s, trusted reference has %s" % (sorted(cand), sorted(trusted))
            else:
                max_err = 0.0
                for key in sorted(trusted):
                    t = np.asarray(cand[key], dtype=np.float64)
                    r = np.asarray(trusted[key], dtype=np.float64)
                    if t.shape != r.shape:
                        outputs_pass[key] = False
                        error = error or "Shape mismatch for %r: candidate %s vs trusted %s" % (key, t.shape, r.shape)
                        continue
                    if r.size == 0:
                        outputs_pass[key] = False
                        error = error or "Output %r is empty: nothing to compare" % key
                        continue
                    finite = r[np.isfinite(r)]
                    if finite.size == 0 or not np.any(finite):
                        vacuous.append(key)
                    with np.errstate(all="ignore"):      # overflow/invalid are statistics, not events
                        diff = float(np.max(np.abs(t - r)))
                        ok = bool(np.allclose(t, r, rtol=rtol_eff, atol=atol_eff))
                    per_output[key] = canonical_stat(diff)
                    if diff != diff:
                        nan_seen = True                  # max() would silently keep 0.0 over a nan
                    else:
                        max_err = max(max_err, diff)
                    outputs_pass[key] = ok
                    if not ok:
                        # no error statistic in the text: the number lives in max_abs_error /
                        # per_output_max_abs_error, which the dispatcher keeps private
                        error = error or "Output %r values differ beyond the tolerance policy (rtol=%s, atol=%s)" % (
                            key, rtol_eff, atol_eff)
            if max_err is not None:
                max_err = "nan" if nan_seen else canonical_stat(max_err)
            # SINGLE source of truth: success is true only when every output was
            # compared and none produced an error
            success = error is None
            verdict = {"success": success, "error": error, "max_abs_error": max_err, "vacuous_outputs": vacuous,
                       "compared_outputs": sorted(trusted), "outputs_pass": outputs_pass,
                       "per_output_max_abs_error": per_output,
                       "tolerance": {"rtol": rtol, "atol": atol, "trusted_dtype_floor": floor,
                                     "rtol_effective": rtol_eff, "atol_effective": atol_eff},
                       "artifact_sha256": hashes}
            program_sha_end = _self_sha256()
            if program_sha_end != PROGRAM_SHA256_AT_START:
                done(13, error="trusted program bytes changed during the compare phase")
            # the launch environment is bound through the parent's own list; the
            # child only verifies it is running in exactly that environment (Python's
            # locale coercion may add LC_CTYPE, nothing else is tolerated)
            expected = META.get("env_expected") or []
            expected_set = set(tuple(kv) for kv in expected)
            extras = {k: v for k, v in os.environ.items() if (k, v) not in expected_set and k != "LC_CTYPE"}
            missing = [kv for kv in expected if tuple(kv) not in set(os.environ.items())]
            if extras or missing:
                done(13, error="environment differs from the launch environment (extra %s, missing %s)"
                     % (sorted(extras), [kv[0] for kv in missing]))
            # the child's digest is a cross-check only: the parent computes the
            # authoritative digest over ITS program hash and ITS artifact hashes
            bound = {"verdict": verdict, "program_sha256": program_sha_end, "argv": _LIST(sys.argv),
                     "env": [_LIST(kv) for kv in expected]}
            digest = hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(",", ":"),
                                               default=str).encode("utf-8")).hexdigest()
            done(0, ok=True, verdict=verdict, verdict_sha256=digest, program_sha256=program_sha_end)
        else:
            done(2, error="unknown phase %r" % PHASE)
    except SystemExit:
        raise
    except Violation as exc:
        done(8, error=str(exc), violations=[str(exc)])
    except BaseException as exc:  # noqa: BLE001
        done(3, error="%s: %s" % (_TYPE(exc).__name__, exc),
             exception={"type": _TYPE(exc).__name__, "mro": [c.__name__ for c in _TYPE(exc).__mro__]})
