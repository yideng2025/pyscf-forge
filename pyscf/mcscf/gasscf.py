#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""PySCF-style generalized active-space self-consistent field.

This module is introduced in small reviewable stages.  Current commits define
object construction, explicit GASCI solver adaptation, GAS orbital-rotation
masks, GASSCF-owned GAS helper plan lifetimes, public solver dispatch through
those plans, GASCI-like object-level wrappers, a fixed-orbital GASCI bridge,
and a minimal native Newton/CIAH driver bridge.  Current validation also
guards unsupported staged feature combinations, kernel lifetimes and
ordinary state-average wrappers, GAS-safe canonicalization, energy scanner support,
GASCI-native spin-penalty hooks and GAS-labeled native driver output.
"""

from collections import OrderedDict
import hashlib
import json
import sys

import numpy

from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import addons as fci_addons
from pyscf.mcscf import addons
from pyscf.mcscf import addons_gas
from pyscf.mcscf import fci_gas
from pyscf.mcscf import gasci
from pyscf.mcscf import newton_casscf

__all__ = ["GASSCF"]


def _unsupported(feature):
    raise NotImplementedError(feature + " is not implemented for GASSCF")



class _GASSCFLogFilter:
    """Write-through stream filter for native Newton/CASSCF messages.

    GASSCF deliberately reuses PySCF's native ``newton_casscf`` driver.
    The numerical driver still contains CASSCF/CASCI text labels and one
    CASSCF-specific experimental-feature warning.  This filter changes only
    user-visible text while leaving the driver and all numerical data untouched.
    """

    _DROP_SUBSTRINGS = (
        "SO-CASSCF (Second order CASSCF) is an experimental feature. "
        "Its performance is bad for large systems.",
    )
    _REPLACEMENTS = (
        ("Start SO-CASSCF (newton CASSCF)", "Start SO-GASSCF"),
        ("newton CASSCF", "GASSCF"),
        ("Second order CASSCF", "Second order GASSCF"),
        ("SO-CASSCF", "SO-GASSCF"),
        ("CASSCF", "GASSCF"),
        ("CASCI", "GASCI"),
        ("CAS (", "GAS ("),
        ("CAS space", "GAS active space"),
        ("CAS-space", "GAS active-space"),
        ("E(CI)", "E(GASCI)"),
    )

    def __init__(self, stream):
        self._stream = sys.stdout if stream is None else stream
        self._pending = ""

    def _rewrite_line(self, line):
        if any(text in line for text in self._DROP_SUBSTRINGS):
            return ""
        if "SO-CASSCF" in line and "experimental feature" in line:
            return ""
        for old, new in self._REPLACEMENTS:
            line = line.replace(old, new)
        return line

    def write(self, text):
        original_length = len(text)
        text = self._pending + text
        if not text:
            return original_length
        if text.endswith("\n"):
            self._pending = ""
            lines = text.splitlines(True)
        else:
            lines = text.splitlines(True)
            if lines and not lines[-1].endswith("\n"):
                self._pending = lines.pop()
            else:
                self._pending = ""
        rewritten = "".join(self._rewrite_line(line) for line in lines)
        if rewritten:
            self._stream.write(rewritten)
        return original_length

    def flush(self):
        if self._pending:
            rewritten = self._rewrite_line(self._pending)
            self._pending = ""
            if rewritten:
                self._stream.write(rewritten)
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _GASFCISolver(fci_gas.FCISolver):
    """GASCI solver shell with GASSCF-owned helper plans.

    Ordinary GASCI remains implemented by :mod:`fci_gas`.  This subclass owns
    only reusable helper plans needed by the staged GASSCF orbital optimizer.
    Adapted or copied solvers always start with empty caches, so C workspaces
    are never borrowed across solver objects.
    """

    _keys = set(fci_gas.FCISolver._keys) | {"cache_plans", "ss_penalty", "ss_value"}
    _MAX_CONTRACT_PLANS = 3

    # PySCF native Newton probes these optional FCI hooks with ``getattr``.
    # GASCI deliberately does not provide CAS linkstr arrays, and arbitrary
    # active-space CI transformations do not preserve a restricted GAS space.
    # Hide the inherited GASCI diagnostics here so native CASSCF falls back to
    # link_index=None and does not call CAS-only helper APIs.
    gen_linkstr = None
    transform_ci_for_orbital_rotation = None

    def __init__(self, mol=None, gas_orbs=None, gas_restr=None,
                 gas_restr_type=addons_gas.GAS_RESTR_SPIN_SUPERGROUP,
                 lib=None, *, cache_plans=True):
        super().__init__(
            mol, gas_orbs=gas_orbs, gas_restr=gas_restr,
            gas_restr_type=gas_restr_type, lib=lib)
        self.cache_plans = bool(cache_plans)
        self._init_plan_cache()

    def _init_plan_cache(self):
        """Detach Newton-owned helper plans without closing borrowed objects.

        This method is used immediately after shallow adaptation or copy.  It
        must not close anything because the copied attributes may still belong
        to the source solver.  Live resources are released by :meth:`close`.
        """

        self._topology_key = None
        self._contract_space = None
        self._contract_plans = OrderedDict()
        self._rdm_plan = None
        self._spin_plan = None

    def _space_key(self, norb, nelec):
        """Return a normalized topology key for GAS helper-plan reuse."""

        gas_orbs, nelec, blocks = self._space_spec(norb, nelec)
        return (
            tuple(int(value) for value in gas_orbs),
            tuple(int(value) for value in nelec),
            tuple(tuple(int(item) for item in row) for row in blocks),
            id(self.lib),
        )

    def _ensure_topology(self, norb, nelec):
        """Drop cached plans when the normalized GAS topology changes."""

        key = self._space_key(norb, nelec)
        if getattr(self, "_topology_key", None) != key:
            self.close()
            self._topology_key = key
        return key

    @staticmethod
    def _contract_eri_key(eri):
        array = numpy.ascontiguousarray(eri, dtype=numpy.float64)
        digest = hashlib.sha256(array.view(numpy.uint8)).digest()
        return array, (tuple(int(value) for value in array.shape), digest)

    def _get_contract_plan(self, eri, norb, nelec):
        """Return a Newton-owned Hamiltonian contraction plan for one ERI."""

        self._ensure_topology(norb, nelec)
        if self._contract_space is None:
            self._contract_space = self.make_space(
                norb, nelec, compress_links=True)
        eri, key = self._contract_eri_key(eri)
        plan = self._contract_plans.pop(key, None)
        if plan is None:
            if len(self._contract_plans) >= self._MAX_CONTRACT_PLANS:
                _, evicted = self._contract_plans.popitem(last=False)
                evicted.close()
            plan = fci_gas.GasContractPlan(self._contract_space, eri.copy())
            plan.eri.flags.writeable = False
            plan.gos.flags.writeable = False
        self._contract_plans[key] = plan
        return plan

    def _get_rdm_plan(self, norb, nelec):
        """Return a Newton-owned raw-link GAS RDM plan."""

        self._ensure_topology(norb, nelec)
        if self._rdm_plan is None:
            self._rdm_plan = self.make_rdm_plan(norb, nelec)
        return self._rdm_plan

    def _get_spin_plan(self, norb, nelec):
        """Return a Newton-owned independent ``S^2`` contraction plan."""

        self._ensure_topology(norb, nelec)
        if self._spin_plan is None:
            self._spin_plan = self.make_spin_plan(norb, nelec)
        return self._spin_plan

    def spin_square(self, ci, norb, nelec, *args, **kwargs):
        """Return ``(<S^2>, 2S+1)`` without re-entering SA RDM wrappers.

        PySCF's dynamic state-average solver calls the base solver's
        ``spin_square`` through ``super(StateAverageFCISolver, self)`` while
        ``self`` is still the decorated state-average object.  The GASCI base
        implementation computes spin from ``self.make_rdm12s``; on a decorated
        object that name resolves back to the state-average wrapper and can
        incorrectly split a single GAS CI vector into scalar elements.  Build
        the spin-resolved RDMs directly from a Newton-owned GAS RDM plan here.
        """

        ci = self._as_state_specific_ci(ci)
        nelec = fci_addons._unpack_nelec(nelec, self.spin)
        if self.cache_plans:
            rdm1s, rdm2s = self._get_rdm_plan(norb, nelec).make_rdm12s(ci, ci)
        else:
            with self.make_rdm_plan(norb, nelec) as plan:
                rdm1s, rdm2s = plan.make_rdm12s(ci, ci)
        return fci_gas.spin_square_from_rdm12s(rdm1s, rdm2s, nelec)


    def close(self):
        """Release Newton-owned helper plans; repeated calls are safe."""

        contract_plans = getattr(self, "_contract_plans", None)
        if contract_plans is not None:
            for plan in list(contract_plans.values()):
                plan.close()
            contract_plans.clear()
        base = getattr(self, "base", None)
        if base is not None and base is not self and hasattr(base, "close"):
            base.close()
        contract_space = getattr(self, "_contract_space", None)
        if contract_space is not None:
            contract_space.close()
            self._contract_space = None
        rdm_plan = getattr(self, "_rdm_plan", None)
        if rdm_plan is not None:
            rdm_plan.close()
            self._rdm_plan = None
        self._spin_plan = None
        self._topology_key = None

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    *args, **kwargs):
        """Contract an absorbed Hamiltonian, reusing a Newton-owned plan."""

        plan = kwargs.pop("plan", None)
        if plan is not None:
            return super().contract_2e(
                eri, fcivec, norb, nelec, link_index,
                *args, plan=plan, **kwargs)
        compress_links = bool(kwargs.pop("compress_links", True))
        if args or kwargs or not self.cache_plans or not compress_links:
            return super().contract_2e(
                eri, fcivec, norb, nelec, link_index,
                *args, compress_links=compress_links, **kwargs)
        return self._get_contract_plan(eri, norb, nelec).contract(fcivec)

    @staticmethod
    def _as_state_specific_ci(ci):
        """Unwrap native Newton's singleton CI-list convention.

        PySCF's native Newton CASSCF helper packs a state-specific CI vector as
        ``[ci]`` when building initial density matrices.  GASCI public methods
        operate on one flattened GAS CI vector.  This bridge accepts only the
        singleton state-specific form here; true multiroot/state-average logic
        is staged later at the GASSCF object level.
        """

        if isinstance(ci, (list, tuple)):
            if len(ci) != 1:
                _unsupported("multiroot CI density dispatch")
            return ci[0]
        return ci

    def make_rdm1s(self, ci, norb, nelec, link_index=None):
        ci = self._as_state_specific_ci(ci)
        return self.trans_rdm1s(ci, ci, norb, nelec, link_index)

    def make_rdm1(self, ci, norb, nelec, link_index=None):
        ci = self._as_state_specific_ci(ci)
        return self.trans_rdm1(ci, ci, norb, nelec, link_index)

    def make_rdm12s(self, ci, norb, nelec, link_index=None, reorder=True):
        ci = self._as_state_specific_ci(ci)
        if not reorder:
            raise NotImplementedError("reorder=False is not supported")
        if not self.cache_plans:
            return super().make_rdm12s(
                ci, norb, nelec, link_index=link_index, reorder=reorder)
        return self._get_rdm_plan(norb, nelec).make_rdm12s(ci, ci)

    def make_rdm12(self, ci, norb, nelec, link_index=None, reorder=True):
        ci = self._as_state_specific_ci(ci)
        if not reorder:
            raise NotImplementedError("reorder=False is not supported")
        if not self.cache_plans:
            return super().make_rdm12(
                ci, norb, nelec, link_index=link_index, reorder=reorder)
        return self._get_rdm_plan(norb, nelec).make_rdm12(ci, ci)

    def make_rdm2(self, ci, norb, nelec, link_index=None, reorder=True):
        return self.make_rdm12(ci, norb, nelec, link_index, reorder)[1]

    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None):
        cibra = self._as_state_specific_ci(cibra)
        ciket = self._as_state_specific_ci(ciket)
        if not self.cache_plans:
            return super().trans_rdm1s(
                cibra, ciket, norb, nelec, link_index=link_index)
        return self._get_rdm_plan(norb, nelec).make_rdm1s(cibra, ciket)

    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None):
        cibra = self._as_state_specific_ci(cibra)
        ciket = self._as_state_specific_ci(ciket)
        if not self.cache_plans:
            return super().trans_rdm1(
                cibra, ciket, norb, nelec, link_index=link_index)
        return self._get_rdm_plan(norb, nelec).make_rdm1(cibra, ciket)

    def trans_rdm12s(self, cibra, ciket, norb, nelec, link_index=None,
                     reorder=True):
        cibra = self._as_state_specific_ci(cibra)
        ciket = self._as_state_specific_ci(ciket)
        if not reorder:
            raise NotImplementedError("reorder=False is not supported")
        if not self.cache_plans:
            return super().trans_rdm12s(
                cibra, ciket, norb, nelec,
                link_index=link_index, reorder=reorder)
        plan = self._get_rdm_plan(norb, nelec)
        dm1s, (dm2aa, dm2ab, dm2bb) = plan.make_rdm12s(cibra, ciket)
        _, (_, dm2ba_ji, _) = plan.make_rdm12s(ciket, cibra)
        dm2ba = dm2ba_ji.transpose(3, 2, 1, 0)
        return dm1s, (dm2aa, dm2ab, dm2ba, dm2bb)

    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None,
                    reorder=True):
        cibra = self._as_state_specific_ci(cibra)
        ciket = self._as_state_specific_ci(ciket)
        if not reorder:
            raise NotImplementedError("reorder=False is not supported")
        if not self.cache_plans:
            return super().trans_rdm12(
                cibra, ciket, norb, nelec,
                link_index=link_index, reorder=reorder)
        return self._get_rdm_plan(norb, nelec).make_rdm12(cibra, ciket)

    def contract_ss(self, fcivec, norb, nelec):
        """Contract ``S^2`` with a GAS CI vector, reusing a spin plan."""

        fcivec = self._as_state_specific_ci(fcivec)
        if not self.cache_plans:
            return super().contract_ss(fcivec, norb, nelec)
        return self._get_spin_plan(norb, nelec).contract(fcivec)

    def copy(self):
        result = super().copy()
        result._init_plan_cache()
        return result


def _new_gas_solver(mf, gas_orbs, gas_restr, gas_restr_type, cache_plans):
    if gas_orbs is None:
        raise ValueError(
            "gas_orbs is required without an explicit GASCI solver")
    gas_orbs = tuple(int(value) for value in gas_orbs)
    if any(value <= 0 for value in gas_orbs):
        raise ValueError("gas_orbs entries must be positive integers")
    if gas_restr_type is None:
        gas_restr_type = addons_gas.GAS_RESTR_SPIN_SUPERGROUP
    if cache_plans is None:
        cache_plans = True
    return _GASFCISolver(
        getattr(mf, "mol", None), gas_orbs=gas_orbs,
        gas_restr=gas_restr, gas_restr_type=gas_restr_type,
        cache_plans=cache_plans)


def _decorated_solver_classes():
    """Return native wrapper classes that should decorate GASSCF, not solver."""

    return tuple(cls for cls in (
        getattr(addons, "StateAverageFCISolver", None),
        getattr(addons, "StateAverageMixFCISolver", None),
        getattr(addons, "StateSpecificFCISolver", None),
        getattr(fci_addons, "SpinPenaltyFCISolver", None),
    ) if cls is not None)


def _adapt_solver(fcisolver, cache_plans):
    """Return a Newton-GASSCF-owned shell for an explicit GASCI solver.

    Scientific GASCI settings are copied from the input solver.  Newton-owned
    helper plans are intentionally not borrowed.  Predecorated solvers are
    rejected because state averaging, state specificity and spin penalty must
    decorate the outer GASSCF object where orbital derivatives are visible.
    """

    if isinstance(fcisolver, _decorated_solver_classes()):
        _unsupported("predecorated solver input; decorate the GASSCF object")
    if isinstance(fcisolver, _GASFCISolver):
        solver = fcisolver.copy()
    elif type(fcisolver) is fci_gas.FCISolver:
        solver = lib.view(fcisolver, _GASFCISolver)
        solver._init_plan_cache()
    else:
        _unsupported("external/non-GASCI solver adaptation")

    if solver.gas_orbs is None:
        raise ValueError("explicit GASCI solver must define gas_orbs")
    solver.gas_orbs = tuple(int(value) for value in solver.gas_orbs)
    if any(value <= 0 for value in solver.gas_orbs):
        raise ValueError("explicit GASCI solver gas_orbs entries must be positive")
    if solver.gas_restr_type is None:
        solver.gas_restr_type = addons_gas.GAS_RESTR_SPIN_SUPERGROUP
    if cache_plans is None:
        solver.cache_plans = bool(getattr(fcisolver, "cache_plans", True))
    else:
        solver.cache_plans = bool(cache_plans)
    return solver


def _digest(value):
    """Return a stable digest for small JSON-like scanner invariants."""

    payload = json.dumps(
        value, sort_keys=True, default=lambda item: numpy.asarray(item).tolist(),
        allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _system_signature(mol):
    """Return molecular invariants that an energy scanner must preserve."""

    return (
        tuple(mol.atom_symbol(i) for i in range(mol.natm)),
        int(mol.charge), int(mol.spin), bool(mol.cart),
        _digest(mol._basis), _digest(mol._ecp), int(mol.nao_nr()),
    )


def _problem_signature(mc):
    """Return GAS/Newton invariants that must remain fixed during scans."""

    gas_orbs, nelec, blocks = mc.fcisolver._space_spec(mc.ncas, mc.nelecas)
    limits = addons_gas.check_kernel_limits(gas_orbs, nelec, blocks)
    return {
        "ncore": int(mc.ncore),
        "ncas": int(mc.ncas),
        "nelec": tuple(int(value) for value in nelec),
        "gas_orbs": tuple(int(value) for value in gas_orbs),
        "spin_supergroups": _digest(numpy.asarray(blocks).tolist()),
        "ndet": int(limits["ndet_estimate"]),
        "nroots": int(getattr(mc.fcisolver, "nroots", 1)),
        "weights": tuple(float(value) for value in getattr(mc, "weights", (1.0,))),
        "spin_penalty": (
            None if not hasattr(mc.fcisolver, "ss_penalty") else
            (float(mc.fcisolver.ss_penalty),
             None if getattr(mc.fcisolver, "ss_value", None) is None else
             float(mc.fcisolver.ss_value))),
        "frozen": _digest(mc.frozen),
    }


def _copy_ci(ci, signature):
    """Copy and validate a scanner CI guess against the GAS model."""

    if ci is None:
        return None
    nroots = int(signature["nroots"])
    ndet = int(signature["ndet"])
    if nroots == 1:
        values = list(ci) if isinstance(ci, (list, tuple)) else [ci]
    elif isinstance(ci, (list, tuple)):
        values = list(ci)
    else:
        array = numpy.asarray(ci)
        if array.shape == (nroots, ndet):
            values = list(array)
        elif array.shape == (ndet, nroots):
            values = list(array.T)
        else:
            raise ValueError("scanner CI must contain one vector per GAS root")
    if len(values) != nroots:
        raise ValueError("scanner CI root count does not match the GAS model")

    roots = []
    for value in values:
        array = numpy.asarray(value)
        if numpy.iscomplexobj(array):
            raise NotImplementedError("complex scanner CI coefficients")
        array = numpy.asarray(array, dtype=float).reshape(-1)
        norm = numpy.linalg.norm(array)
        if array.size != ndet or not numpy.all(numpy.isfinite(array)) or norm < 1e-14:
            raise ValueError(
                "scanner CI has invalid GAS length, norm or coefficients")
        roots.append(numpy.array(array, copy=True))
    return roots[0] if nroots == 1 else roots


def _orbital_groups(mc):
    """Return projection priority groups that preserve GAS subspace ordering."""

    groups = []
    start = int(mc.ncore)
    for size in mc.fcisolver._space_spec(mc.ncas, mc.nelecas)[0]:
        stop = start + int(size)
        groups.append(list(range(start, stop)))
        start = stop
    if mc.ncore:
        groups.append(list(range(int(mc.ncore))))
    return groups


def _as_scanner(mc):
    """Return an energy-only scanner with fixed GAS/Newton objective metadata."""

    if isinstance(mc, lib.SinglePointScanner):
        return mc
    mc.validate_capabilities()
    source = mc.copy()
    source._scf = mc._scf.copy()
    source.mo_coeff = None if mc.mo_coeff is None else numpy.array(
        mc.mo_coeff, copy=True)
    source.ci = _copy_ci(mc.ci, _problem_signature(mc))
    return lib.set_class(
        _GASSCFScanner(source), (_GASSCFScanner, source.__class__),
        source.__class__.__name__ + "Scanner")


class _GASSCFScanner(lib.SinglePointScanner):
    """Energy scanner for a fixed GASSCF objective and atom/basis model."""

    _keys = {"scan_info", "_scan_problem", "_scan_system"}

    def __init__(self, mc):
        self.__dict__.update(mc.__dict__)
        self._scf = mc._scf.as_scanner()
        self._scan_problem = _problem_signature(mc)
        self._scan_system = _system_signature(mc.mol)
        self.scan_info = None

    def __call__(self, mol_or_geom, mo_coeff=None, ci0=None):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        if _system_signature(mol) != self._scan_system:
            raise ValueError(
                "energy scanner requires the same atoms/order, charge, spin, "
                "basis/ECP and AO size")

        self.validate_capabilities()
        signature = _problem_signature(self)
        if signature != self._scan_problem:
            raise ValueError(
                "GAS definition, roots, weights or frozen orbitals changed; "
                "create a new scanner")

        old_mol = self.mol
        previous_mo = None if self.mo_coeff is None else numpy.array(
            self.mo_coeff, copy=True)
        guess_ci = _copy_ci(self.ci if ci0 is None else ci0, signature)

        self.reset(mol)
        self._scf(mol)
        self.mol = mol
        self.validate_capabilities()

        if mo_coeff is not None:
            guess_mo = numpy.asarray(mo_coeff)
            mo_source = "explicit MO in current AO basis"
        elif previous_mo is None:
            guess_mo = numpy.asarray(self._scf.mo_coeff)
            mo_source = "SCF MO guess"
        else:
            guess_mo = addons.project_init_guess(
                self, previous_mo, prev_mol=old_mol,
                priority=_orbital_groups(self), use_hf_core=False)
            mo_source = "projected previous GASSCF MO by GAS blocks"

        guess_mo = numpy.asarray(guess_mo)
        if (guess_mo.ndim != 2 or numpy.iscomplexobj(guess_mo) or
                not numpy.all(numpy.isfinite(guess_mo))):
            raise ValueError("scanner MO guess must be a finite real matrix")
        if guess_mo.shape[0] != mol.nao_nr() or guess_mo.shape[1] < self.ncore + self.ncas:
            raise ValueError("scanner MO guess has an incompatible shape")
        overlap = self._scf.get_ovlp()
        metric = guess_mo.T.dot(overlap).dot(guess_mo)
        error = float(numpy.max(numpy.abs(metric - numpy.eye(guess_mo.shape[1]))))
        if error > 1e-7:
            raise ValueError(
                "scanner initial MO is not orthonormal in the new AO metric: "
                "%.6g" % error)

        self.scan_info = {
            "MO_source": mo_source,
            "CI_source": "explicit" if ci0 is not None else (
                "previous GAS CI guess" if guess_ci is not None else
                "native initial guess"),
            "initial_MO_metric_error": error,
            "problem": signature,
            "projection_groups": _orbital_groups(self),
            "native_converged": None,
            "scope": "Energy only; root order follows macro GASCI.",
        }
        energy = self.kernel(numpy.array(guess_mo, copy=True), guess_ci)[0]
        self.scan_info.update(
            native_converged=bool(self.converged),
            energy=float(energy),
            e_states=numpy.atleast_1d(getattr(self, "e_states", energy)).tolist(),
        )
        return energy


class _StateAverageGASSCF(addons.StateAverageMCSCF):
    """State-average marker with GAS-specific undo/cache cleanup."""

    def undo_state_average(self):
        self.close()
        result = super().undo_state_average()
        result.fcisolver.nroots = 1
        result.fcisolver._init_plan_cache()
        result.fcisolver.mol = result.mol
        return result


class GASSCF(newton_casscf.CASSCF):
    """Joint GASSCF orbital optimizer for a determinant GASCI active space.

    Args:
        mf : SCF object
            Mean-field object that supplies molecular data and orbitals.
        gas_orbs : sequence of ints
            Ordered numbers of active orbitals in the GAS subspaces.  The total
            active-space size is ``sum(gas_orbs)``.
        gas_restr : object, optional
            GAS restriction in the syntax selected by ``gas_restr_type``.
        nelecas : int or pair of ints
            Number of active electrons, optionally resolved as alpha/beta.
        gas_restr_type : str, optional
            ``spin-supergroup``, ``supergroup``, ``cumulative-occ`` or ``ras``.
            If omitted, the GASCI default ``spin-supergroup`` is used.
        ncore : int, optional
            Number of inactive doubly occupied orbitals.
        frozen : int or sequence of ints, optional
            Frozen orbital specification forwarded to the native Newton class.
        cache_plans : bool, optional
            Future policy for Newton-owned GAS contraction/RDM/spin plans.

    Notes:
        This first implementation stage establishes the public object and
        solver ownership convention only.  Later commits add the GAS orbital
        rotation mask and native CIAH derivative adapter.
    """

    _keys = set(newton_casscf.CASSCF._keys) | {
        "gas_orbs", "gas_restr", "gas_restr_type", "cache_plans"}

    def __init__(self, mf, gas_orbs=None, gas_restr=None, *, nelecas,
                 gas_restr_type=None, ncore=None, frozen=None,
                 fcisolver=None, cache_plans=None):
        if fcisolver is None:
            solver = _new_gas_solver(
                mf, gas_orbs, gas_restr, gas_restr_type, cache_plans)
        else:
            if (gas_orbs is not None or gas_restr is not None or
                    gas_restr_type is not None):
                raise ValueError(
                    "gas_orbs, gas_restr and gas_restr_type are supplied "
                    "by the explicit fcisolver")
            solver = _adapt_solver(fcisolver, cache_plans)

        super().__init__(
            mf, sum(solver.gas_orbs), nelecas, ncore=ncore, frozen=frozen)
        self.fcisolver = solver
        self.fcisolver.mol = self.mol


    def _push_gasscf_log_labels(self):
        """Install temporary stream filters for native Newton log labels."""

        stdout = getattr(self, "stdout", None)
        restore_stdout = not isinstance(stdout, _GASSCFLogFilter)
        if restore_stdout:
            self.stdout = _GASSCFLogFilter(stdout)

        stderr = sys.stderr
        restore_stderr = not isinstance(stderr, _GASSCFLogFilter)
        if restore_stderr:
            sys.stderr = _GASSCFLogFilter(stderr)

        return stdout, restore_stdout, stderr, restore_stderr

    def dump_flags(self, verbose=None):
        """Print GASSCF flags using GAS terminology."""

        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        if self.mo_coeff is None:
            log.info("GAS (%de+%de, %do), ncore = %d",
                     self.nelecas[0], self.nelecas[1], ncas, ncore)
        else:
            nvir = self.mo_coeff.shape[1] - ncore - ncas
            log.info("GAS (%de+%de, %do), ncore = %d, nvir = %d",
                     self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        log.info("gas_orbs = %s", self.gas_orbs)
        log.info("gas_restr_type = %s", self.gas_restr_type)
        log.info("gas_restr = %s", self.gas_restr)
        log.info("cache GAS helper plans = %s",
                 getattr(self.fcisolver, "cache_plans", None))
        if self.frozen is not None:
            log.info("frozen orbitals %s", str(self.frozen))
        if hasattr(self.fcisolver, "ss_penalty"):
            target = getattr(self.fcisolver, "ss_value", None)
            log.info("spin penalty shift = %g", self.fcisolver.ss_penalty)
            log.info("target S^2 = %s", "minimum" if target is None else target)
        log.info("max_cycle_macro = %d", self.max_cycle_macro)
        log.info("max_cycle_micro = %d", self.max_cycle_micro)
        log.info("conv_tol = %g", self.conv_tol)
        log.info("conv_tol_grad = %s", self.conv_tol_grad)
        log.info("orbital rotation max_stepsize = %g", self.max_stepsize)
        log.info("augmented hessian ah_max_cycle = %d", self.ah_max_cycle)
        log.info("augmented hessian ah_conv_tol = %g", self.ah_conv_tol)
        log.info("augmented hessian ah_linear dependence = %g", self.ah_lindep)
        log.info("augmented hessian ah_level shift = %g", self.ah_level_shift)
        log.info("augmented hessian ah_start_tol = %g", self.ah_start_tol)
        log.info("augmented hessian ah_start_cycle = %d", self.ah_start_cycle)
        log.info("augmented hessian ah_grad_trust_region = %g",
                 self.ah_grad_trust_region)
        log.info("kf_trust_region = %g", self.kf_trust_region)
        log.info("kf_interval = %d", self.kf_interval)
        log.info("natorb = %s", self.natorb)
        log.info("canonicalization = %s", self.canonicalization)
        log.info("chkfile = %s", self.chkfile)
        log.info("max_memory %d MB (current use %d MB)",
                 self.max_memory, lib.current_memory()[0])
        log.info("internal_rotation = %s", self.internal_rotation)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except AttributeError:
            pass
        if self.mo_coeff is None:
            log.warn("Orbital for GASSCF is not specified.  You probably need "
                     "call SCF.kernel() to initialize orbitals.")
        return self


    def validate_capabilities(self):
        """Validate the currently staged GASSCF feature set.

        The production algorithm is still assembled in small commits.  This
        guard makes unsupported combinations fail before entering the native
        CASSCF driver and sets the native ``internal_rotation`` flag whenever
        active-active inter-GAS rotations are part of the orbital variables.
        Ordinary state averaging is supported only when both the outer MCSCF
        object and the inner GASCI solver carry PySCF's matching SA wrappers.
        """

        if not isinstance(self.fcisolver, _GASFCISolver):
            _unsupported("non-adapted GASCI solver")
        if isinstance(self.fcisolver, addons.StateAverageMixFCISolver):
            _unsupported("state-average-mix GASCI solver")
        if isinstance(self.fcisolver, addons.StateSpecificFCISolver):
            _unsupported("state-specific GASCI solver wrapper")

        if isinstance(self.fcisolver, fci_addons.SpinPenaltyFCISolver):
            _unsupported(
                "PySCF SpinPenaltyFCISolver wrapper; use GASSCF.fix_spin_")

        if getattr(self, "natorb", False):
            _unsupported("GAS natural-orbital rotation")

        is_sa_mc = isinstance(self, addons.StateAverageMCSCF)
        is_sa_solver = isinstance(self.fcisolver, addons.StateAverageFCISolver)
        if is_sa_mc != is_sa_solver:
            raise ValueError(
                "state_average requires matching MCSCF and GASCI solver wrappers")
        if is_sa_mc:
            weights = self._validate_weights(self.weights)
            if int(getattr(self.fcisolver, "nroots", 1)) != len(weights):
                raise ValueError("nroots/weights mismatch")
        elif int(getattr(self.fcisolver, "nroots", 1)) != 1:
            raise ValueError(
                "nroots>1 requires ordinary state_average support")

        gas_orbs, gas_restr = self._normalized_restriction()
        nelecas = self._effective_nelecas()
        addons_gas.check_kernel_limits(gas_orbs, nelecas, gas_restr)
        if hasattr(self.fcisolver, "ss_penalty"):
            if not addons_gas.is_spin_complete(gas_orbs, nelecas, gas_restr):
                raise ValueError(
                    "fix_spin_ requires a spin-complete GAS restriction")
            fci_gas._spin_penalty_parameters(
                self.fcisolver, self.ncas, nelecas)
        self.internal_rotation = len(gas_orbs) > 1
        self.fcisolver.mol = self.mol
        return self

    def close(self):
        """Release GASSCF-owned GAS helper plans; repeated calls are safe."""

        self.fcisolver.close()
        return self

    def reset(self, mol=None):
        """Reset molecular data and drop GAS helper-plan caches."""

        self.close()
        result = super().reset(mol)
        self.fcisolver.mol = self.mol
        return result

    def copy(self):
        """Return a copy with independent Newton-owned GAS solver caches."""

        result = super().copy()
        result.fcisolver = self.fcisolver.copy()
        result.fcisolver.mol = result.mol
        return result

    def newton(self):
        """Return this GASSCF object after validation."""

        return self.validate_capabilities()

    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        """Return the independent orbital-rotation mask for GAS orbital optimization.

        The native CASSCF mask contains core-active, core-external and
        active-external rotations.  GASSCF adds active-active rotations between
        different GAS subspaces because such rotations change the constrained
        GAS wave function.  Rotations within one GAS subspace remain redundant
        orbital gauge degrees of freedom and are excluded.
        """

        nmo = int(nmo)
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        gas_orbs = tuple(int(value) for value in self.gas_orbs)
        if sum(gas_orbs) != ncas:
            raise ValueError("sum(gas_orbs) must equal ncas")

        mask = numpy.zeros((nmo, nmo), dtype=bool)
        mask[ncore:nocc, :ncore] = True
        mask[nocc:, :nocc] = True

        first_active = ncore
        offset = ncore
        for norb in gas_orbs:
            start = offset
            stop = start + norb
            mask[start:stop, first_active:start] = True
            offset = stop

        if self.extrasym is not None:
            extrasym = numpy.asarray(self.extrasym)
            extrasym_allowed = extrasym.reshape(-1, 1) == extrasym
            mask = mask * extrasym_allowed
        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = mask[:, :frozen] = False
            else:
                frozen = numpy.asarray(frozen)
                mask[frozen] = mask[:, frozen] = False
        return mask

    def _effective_nelecas(self, nelecas=None):
        """Return active alpha/beta counts after applying ``fcisolver.spin``."""

        if nelecas is None:
            nelecas = self.nelecas
        return fci_addons._unpack_nelec(nelecas, self.fcisolver.spin)

    def _normalized_restriction(self, return_info=False):
        """Return the normalized GAS definition used by GASCI kernels."""

        gas_orbs = tuple(int(value) for value in self.gas_orbs)
        return addons_gas.normalize_gas_spec(
            gas_orbs, self._effective_nelecas(),
            self.gas_restr, self.gas_restr_type,
            return_info=return_info)

    def gas_space_info(self):
        """Return normalized GAS metadata and compact C-space information."""

        gas_orbs, blocks, info = self._normalized_restriction(return_info=True)
        restriction_type = self.gas_restr_type
        if self.gas_restr is None:
            restriction = None
        elif restriction_type == addons_gas.GAS_RESTR_SPIN_SUPERGROUP:
            restriction = numpy.array(
                info["canonical_spin_supergroups"], copy=True)
        elif restriction_type == addons_gas.GAS_RESTR_SUPERGROUP:
            restriction = numpy.array(info["canonical_supergroups"], copy=True)
        elif restriction_type == addons_gas.GAS_RESTR_CUMULATIVE_OCC:
            restriction = numpy.array(info["cumulative_bounds"], copy=True)
        elif restriction_type == addons_gas.GAS_RESTR_RAS:
            restriction = {
                "max_holes": int(info["max_holes"]),
                "max_particles": int(info["max_particles"]),
            }
        else:  # normalize_gas_spec rejects this before reaching this branch.
            raise RuntimeError("unrecognized normalized GAS restriction type")

        with fci_gas.GasSpace(
                gas_orbs, self._effective_nelecas(), blocks,
                lib=self.fcisolver.lib) as space:
            core = space.core_info()
        return {
            "metadata": {
                "gas_orbs": tuple(int(value) for value in self.gas_orbs),
                "gas_restr_type": restriction_type,
                "gas_restr": restriction,
                "kernel_gas_orbs": tuple(int(value) for value in gas_orbs),
                "spin_supergroups": numpy.array(blocks, copy=True),
            },
            "core": core,
        }

    def _ci_for_active_property(self, ci=None):
        ci = self.ci if ci is None else ci
        if ci is None:
            raise ValueError("CI vector is not available")
        if isinstance(ci, (list, tuple)):
            _unsupported("state-averaged GAS density wrappers")
        return ci

    def make_gasdm1s(self, ci=None, ncas=None, nelecas=None):
        """Return alpha and beta active-space GAS one-particle DMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        return self.fcisolver.make_rdm1s(
            self._ci_for_active_property(ci), ncas, nelecas)

    def make_gasdm1(self, ci=None, ncas=None, nelecas=None):
        """Return the spin-summed active-space GAS one-particle DM."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        return self.fcisolver.make_rdm1(
            self._ci_for_active_property(ci), ncas, nelecas)

    def make_gasdm12s(self, ci=None, ncas=None, nelecas=None):
        """Return spin-resolved active-space GAS 1- and 2-particle DMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        return self.fcisolver.make_rdm12s(
            self._ci_for_active_property(ci), ncas, nelecas)

    def make_gasdm12(self, ci=None, ncas=None, nelecas=None):
        """Return spin-summed active-space GAS 1- and 2-particle DMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        return self.fcisolver.make_rdm12(
            self._ci_for_active_property(ci), ncas, nelecas)

    def make_gasdm2(self, ci=None, ncas=None, nelecas=None):
        """Return the spin-summed active-space GAS two-particle DM."""

        return self.make_gasdm12(ci, ncas, nelecas)[1]

    def trans_gasdm1s(self, cibra=None, ciket=None, ncas=None, nelecas=None):
        """Return alpha and beta active-space GAS transition 1-DMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        bra = self._ci_for_active_property(cibra)
        ket = self._ci_for_active_property(ciket)
        return self.fcisolver.trans_rdm1s(bra, ket, ncas, nelecas)

    def trans_gasdm1(self, cibra=None, ciket=None, ncas=None, nelecas=None):
        """Return the spin-summed active-space GAS transition 1-DM."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        bra = self._ci_for_active_property(cibra)
        ket = self._ci_for_active_property(ciket)
        return self.fcisolver.trans_rdm1(bra, ket, ncas, nelecas)

    def trans_gasdm12s(self, cibra=None, ciket=None, ncas=None, nelecas=None):
        """Return spin-resolved active-space GAS transition 1- and 2-DMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        bra = self._ci_for_active_property(cibra)
        ket = self._ci_for_active_property(ciket)
        return self.fcisolver.trans_rdm12s(bra, ket, ncas, nelecas)

    def trans_gasdm12(self, cibra=None, ciket=None, ncas=None, nelecas=None):
        """Return spin-summed active-space GAS transition 1- and 2-DMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        bra = self._ci_for_active_property(cibra)
        ket = self._ci_for_active_property(ciket)
        return self.fcisolver.trans_rdm12(bra, ket, ncas, nelecas)

    def trans_gasdm2(self, cibra=None, ciket=None, ncas=None, nelecas=None):
        """Return the spin-summed active-space GAS transition 2-DM."""

        return self.trans_gasdm12(cibra, ciket, ncas, nelecas)[1]

    def spin_square(self, ci=None, ncas=None, nelecas=None):
        """Return ``(<S^2>, 2S+1)`` for a state-specific GAS CI vector."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        return self.fcisolver.spin_square(
            self._ci_for_active_property(ci), ncas, nelecas)

    def get_h1gas(self, mo_coeff=None, ncas=None, ncore=None):
        """Return the effective one-electron Hamiltonian in the GAS space."""

        return self.get_h1eff(mo_coeff, ncas, ncore)

    def get_h2gas(self, mo_coeff=None):
        """Return active-space two-electron integrals for GASSCF."""

        return self.get_h2eff(mo_coeff)

    def _prepare_fixed_orbital_gasci(self, mo_coeff=None, ci0=None):
        """Return orbitals and CI guess for a fixed-orbital GASCI call."""

        if mo_coeff is None:
            if self.mo_coeff is None and self._scf.mol.nelectron > 0:
                self._scf.run()
                self.mo_coeff = self._scf.mo_coeff
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci
        self.fcisolver.mol = self.mol
        return mo_coeff, ci0

    def _run_fixed_orbital_gasci(self, mo_coeff=None, ci0=None, verbose=None):
        """Run fixed-orbital GASCI and update PySCF-style result slots."""

        mo_coeff, ci0 = self._prepare_fixed_orbital_gasci(mo_coeff, ci0)
        self.e_tot, self.e_cas, self.ci = gasci.kernel(
            self, mo_coeff, ci0=ci0, verbose=verbose)
        if getattr(self.fcisolver, "converged", None) is not None:
            self.converged = bool(numpy.all(self.fcisolver.converged))
        else:
            self.converged = True
        self._sync_spin_penalty_results()
        return self.e_tot, self.e_gas, self.ci

    def gasci(self, mo_coeff=None, ci0=None, verbose=None):
        """Run the fixed-orbital GASCI problem associated with this object.

        This method is a convenience bridge used while the Newton/CIAH
        derivative adapter is staged separately.  It does not optimize
        orbitals.
        """

        e_tot, e_gas, ci = self._run_fixed_orbital_gasci(
            mo_coeff, ci0, verbose)
        return e_tot, e_gas, ci, self.mo_coeff, self.mo_energy

    def casci(self, mo_coeff=None, ci0=None, eris=None, verbose=None, envs=None):
        """Run a fixed-orbital GASCI solve with native-CASSCF call signature."""

        e_tot, e_gas, ci = self._run_fixed_orbital_gasci(
            mo_coeff, ci0, verbose)
        if numpy.ndim(e_gas) != 0:
            raise RuntimeError(
                "Multiple roots are detected in fcisolver.  GASSCF "
                "does not yet know which state to optimize.\n"
                "Use a state-specific solver or wait for staged state-average "
                "support.")
        return e_tot, e_gas, ci

    def canonicalize(self, mo_coeff=None, ci=None, eris=None, sort=False,
                     gas_natorb=False, gasdm1=None, verbose=None,
                     cas_natorb=None, **kwargs):
        """Canonicalize core/external orbitals without rotating GAS subspaces.

        Native CASSCF calls this method with its ``natorb`` flag as the fifth
        positional argument.  In GASSCF, any automatic active-space natural
        orbital rotation would generally mix GAS subspaces and invalidate the
        restricted determinant space, so active-space natural orbitals are
        explicitly guarded.  The active GAS block is otherwise kept fixed;
        only inactive and external orbitals are canonicalized by the inherited
        PySCF machinery.
        """

        if gas_natorb or cas_natorb:
            _unsupported("GAS natural-orbital rotation")
        return gasci.GASCI.canonicalize(
            self, mo_coeff, ci, eris, sort=sort, gas_natorb=False,
            gasdm1=gasdm1, verbose=verbose, **kwargs)

    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      gas_natorb=False, gasdm1=None, verbose=None,
                      cas_natorb=None, **kwargs):
        mo_coeff, ci, mo_energy = self.canonicalize(
            mo_coeff, ci, eris, sort=sort, gas_natorb=gas_natorb,
            gasdm1=gasdm1, verbose=verbose, cas_natorb=cas_natorb, **kwargs)
        self.mo_coeff = mo_coeff
        self.ci = ci
        self.mo_energy = mo_energy
        return mo_coeff, ci, mo_energy

    def cas_natorb(self, *args, **kwargs):
        _unsupported("CAS/GAS natural-orbital rotation")

    cas_natorb_ = cas_natorb

    @staticmethod
    def _validate_weights(weights):
        """Return finite nonnegative state-average weights summing to one."""

        weights = numpy.asarray(weights, dtype=float)
        if weights.ndim != 1 or weights.size < 2:
            raise ValueError("state_average requires at least two weights")
        if (not numpy.all(numpy.isfinite(weights)) or
                numpy.any(weights < 0) or
                abs(float(numpy.sum(weights)) - 1.0) > 1e-10):
            raise ValueError(
                "weights must be finite, nonnegative and sum to one")
        return tuple(float(value) for value in weights)

    def state_average(self, weights=(.5, .5), wfnsym=None):
        """Return an ordinary state-average GASSCF object.

        Zero-weight roots are retained exactly as requested, so they may still
        contribute to the native CI-response path even though they do not enter
        the scalar energy objective.  ``wfnsym`` and SA-mix are staged later.
        """

        if wfnsym is not None:
            _unsupported("wfnsym")
        weights = self._validate_weights(weights)
        source = self.undo_state_average() if isinstance(
            self, addons.StateAverageMCSCF) else self.copy()
        source.validate_capabilities()
        result = addons.state_average(source, weights, wfnsym=None)
        result.__class__ = lib.replace_class(
            result.__class__, addons.StateAverageMCSCF, _StateAverageGASSCF)
        return result.validate_capabilities()

    def state_average_(self, weights=(.5, .5), wfnsym=None):
        result = self.state_average(weights, wfnsym)
        self.close()
        self.__class__ = result.__class__
        self.__dict__ = result.__dict__
        return self

    def state_average_mix(self, *args, **kwargs):
        _unsupported("state-average-mix GASSCF")

    state_average_mix_ = state_average_mix

    def _sync_spin_penalty_results(self):
        """Mirror GASCI spin-penalty bookkeeping onto the GASSCF object."""

        self.spin_penalty_method = getattr(
            self.fcisolver, "spin_penalty_method", None)
        if not hasattr(self.fcisolver, "ss_penalty"):
            self.e_spin_penalty = None
            self.e_tot_physical = self.e_tot
            self.e_gas_physical = self.e_gas
            return self
        solver_physical = getattr(self.fcisolver, "e_physical", None)
        solver_penalty = getattr(self.fcisolver, "e_spin_penalty", None)
        if (solver_physical is None or solver_penalty is None or
                self.e_tot is None or self.e_gas is None):
            self.e_spin_penalty = None
            self.e_tot_physical = None
            self.e_gas_physical = None
            return self
        physical = numpy.asarray(solver_physical, dtype=numpy.float64)
        penalty = numpy.asarray(solver_penalty, dtype=numpy.float64)
        core = (numpy.asarray(self.e_tot, dtype=numpy.float64) -
                numpy.asarray(self.e_gas, dtype=numpy.float64))
        gas_physical = physical - core
        if physical.ndim == 0:
            self.e_spin_penalty = float(penalty)
            self.e_tot_physical = float(physical)
            self.e_gas_physical = float(gas_physical)
        else:
            self.e_spin_penalty = penalty
            self.e_tot_physical = physical
            self.e_gas_physical = gas_physical
        return self

    def fix_spin_(self, shift=.2, ss=None):
        """Enable GASCI-native spin penalty for GASSCF.

        ``ss`` is the target ``S(S+1)`` value, matching PySCF's
        ``fix_spin_`` convention.  The implementation uses the
        determinant GASCI solver's native spin-penalty Hamiltonian
        instead of wrapping the solver in PySCF's CAS-oriented
        ``SpinPenaltyFCISolver`` dynamic class.  This keeps GAS
        link tables and CI vectors in the GAS representation.
        """

        self.validate_capabilities()
        gas_orbs, gas_restr = self._normalized_restriction()
        nelecas = self._effective_nelecas()
        if not addons_gas.is_spin_complete(gas_orbs, nelecas, gas_restr):
            raise ValueError(
                "fix_spin_ requires a spin-complete GAS restriction")
        shift = float(shift)
        target = None if ss is None else float(ss)
        trial = self.fcisolver.copy()
        trial.ss_penalty = shift
        trial.ss_value = target
        fci_gas._spin_penalty_parameters(trial, self.ncas, nelecas)
        self.close()
        self.fcisolver.ss_penalty = shift
        self.fcisolver.ss_value = target
        self.fcisolver.e_physical = None
        self.fcisolver.e_spin_penalty = None
        self.fcisolver.spin_penalty_method = None
        return self.validate_capabilities()

    def fix_spin(self, shift=.2, ss=None):
        """Return a copied GASSCF object with spin penalty enabled."""

        return self.copy().fix_spin_(shift=shift, ss=ss)

    def undo_fix_spin_(self):
        """Disable GASCI-native spin penalty in place."""

        self.close()
        for key in ("ss_penalty", "ss_value"):
            self.fcisolver.__dict__.pop(key, None)
        self.fcisolver.e_physical = None
        self.fcisolver.e_spin_penalty = None
        self.fcisolver.spin_penalty_method = None
        self._sync_spin_penalty_results()
        return self.validate_capabilities()

    def undo_fix_spin(self):
        """Return a copied GASSCF object without spin penalty."""

        return self.copy().undo_fix_spin_()

    def spin_energy_report(self):
        """Report root-resolved physical and penalty energies."""

        if (not hasattr(self.fcisolver, "ss_penalty") or
                getattr(self.fcisolver, "e_physical", None) is None or
                getattr(self.fcisolver, "e_spin_penalty", None) is None):
            raise ValueError("no completed spin-penalized GASCI solve")
        physical = numpy.atleast_1d(numpy.asarray(
            self.fcisolver.e_physical, dtype=numpy.float64))
        penalty = numpy.atleast_1d(numpy.asarray(
            self.fcisolver.e_spin_penalty, dtype=numpy.float64))
        weights = numpy.asarray(getattr(self, "weights", (1.0,)),
                                dtype=numpy.float64)
        if weights.size != physical.size:
            if physical.size == 1:
                weights = numpy.ones(1, dtype=numpy.float64)
            else:
                raise ValueError(
                    "state weights do not match spin-penalty roots")
        objective = physical + penalty
        return {
            "root_physical": physical.tolist(),
            "root_penalty": penalty.tolist(),
            "root_objective": objective.tolist(),
            "physical": float(numpy.dot(weights, physical)),
            "penalty": float(numpy.dot(weights, penalty)),
            "objective": float(numpy.dot(weights, objective)),
            "shift": float(self.fcisolver.ss_penalty),
            "target_s2": getattr(self.fcisolver, "ss_value", None),
            "method": getattr(self.fcisolver, "spin_penalty_method", None),
        }


    def as_scanner(self):
        """Return an energy-only scanner for a fixed GASSCF objective."""

        return _as_scanner(self)

    def state_specific_(self, *args, **kwargs):
        _unsupported("state-specific GASSCF")

    state_specific = state_specific_

    gen_g_hop = newton_casscf.gen_g_hop

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        """Run full GASSCF orbital optimization with native CIAH.

        This staged bridge reuses PySCF's native Newton/CIAH macro/micro
        control flow.  GAS-specific behavior enters through the GAS orbital
        mask, the fixed-orbital GASCI ``casci`` bridge, and the Newton-owned
        GASCI solver dispatch methods staged above.
        """

        self.validate_capabilities()
        mo = self.mo_coeff if mo_coeff is None else mo_coeff
        if (mo is not None and self.ncas == mo.shape[1] and
                not self.internal_rotation and not self.canonicalization):
            e_tot, e_gas, ci = self._run_fixed_orbital_gasci(mo, ci0)
            self.mo_energy = None
            return e_tot, e_gas, ci, self.mo_coeff, self.mo_energy
        (stdout, restore_stdout, stderr, restore_stderr) = (
            self._push_gasscf_log_labels())
        try:
            result = super().kernel(mo_coeff, ci0, callback)
            self._sync_spin_penalty_results()
            return result
        finally:
            self.close()
            if restore_stdout:
                self.stdout.flush()
                self.stdout = stdout
            if restore_stderr:
                sys.stderr.flush()
                sys.stderr = stderr

    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        _unsupported("two-step GASSCF kernel")

    @property
    def gas_orbs(self):
        """Ordered numbers of active orbitals in each GAS subspace."""

        return self.fcisolver.gas_orbs

    @gas_orbs.setter
    def gas_orbs(self, value):
        self.fcisolver.gas_orbs = (
            None if value is None else tuple(int(item) for item in value))

    @property
    def gas_restr(self):
        """GAS restriction in the syntax selected by ``gas_restr_type``."""

        return self.fcisolver.gas_restr

    @gas_restr.setter
    def gas_restr(self, value):
        self.fcisolver.gas_restr = value

    @property
    def gas_restr_type(self):
        """Restriction syntax forwarded to GASCI normalization."""

        return self.fcisolver.gas_restr_type

    @gas_restr_type.setter
    def gas_restr_type(self, value):
        self.fcisolver.gas_restr_type = value

    @property
    def cache_plans(self):
        """Whether GASSCF may reuse owned GAS helper plans."""

        return self.fcisolver.cache_plans

    @cache_plans.setter
    def cache_plans(self, value):
        self.fcisolver.cache_plans = bool(value)

    @property
    def ngas(self):
        """Number of user-visible GAS subspaces."""

        return len(self.gas_orbs)

    @property
    def e_gas(self):
        """Alias for the inherited active-space energy slot."""

        return self.e_cas
