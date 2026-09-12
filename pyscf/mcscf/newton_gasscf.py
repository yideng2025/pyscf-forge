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
masks, Newton-owned GAS helper plan lifetimes, public solver dispatch through
those plans, GASCI-like object-level wrappers, a fixed-orbital GASCI bridge,
and a minimal native Newton/CIAH driver bridge.  Current validation also
guards unsupported staged feature combinations and kernel lifetimes.
"""

from collections import OrderedDict
import hashlib

import numpy

from pyscf import lib
from pyscf.fci import addons as fci_addons
from pyscf.mcscf import addons
from pyscf.mcscf import addons_gas
from pyscf.mcscf import fci_gas
from pyscf.mcscf import gasci
from pyscf.mcscf import newton_casscf

__all__ = ["GASSCF"]


def _unsupported(feature):
    raise NotImplementedError(feature + " is not implemented for Newton GASSCF")


class _GASFCISolver(fci_gas.FCISolver):
    """GASCI solver shell with Newton GASSCF-owned helper plans.

    Ordinary GASCI remains implemented by :mod:`fci_gas`.  This subclass owns
    only reusable helper plans needed by the staged Newton orbital optimizer.
    Adapted or copied solvers always start with empty caches, so C workspaces
    are never borrowed across solver objects.
    """

    _keys = set(fci_gas.FCISolver._keys) | {"cache_plans"}
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

    def close(self):
        """Release Newton-owned helper plans; repeated calls are safe."""

        contract_plans = getattr(self, "_contract_plans", None)
        if contract_plans is not None:
            for plan in list(contract_plans.values()):
                plan.close()
            contract_plans.clear()
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


class GASSCF(newton_casscf.CASSCF):
    """Joint Newton orbital optimizer for a determinant GASCI active space.

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

    def validate_capabilities(self):
        """Validate the currently staged Newton GASSCF feature set.

        The production algorithm is still assembled in small commits.  This
        guard makes unsupported combinations fail before entering the native
        CASSCF driver and sets the native ``internal_rotation`` flag whenever
        active-active inter-GAS rotations are part of the orbital variables.
        """

        if not isinstance(self.fcisolver, _GASFCISolver):
            _unsupported("non-adapted GASCI solver")
        if isinstance(self, addons.StateAverageMCSCF):
            _unsupported("state-average Newton GASSCF")
        if isinstance(self.fcisolver, addons.StateAverageMixFCISolver):
            _unsupported("state-average-mix GASCI solver")
        if isinstance(self.fcisolver, addons.StateSpecificFCISolver):
            _unsupported("state-specific GASCI solver wrapper")
        if int(getattr(self.fcisolver, "nroots", 1)) != 1:
            raise ValueError(
                "nroots>1 requires staged state_average support")

        gas_orbs, gas_restr = self._normalized_restriction()
        addons_gas.check_kernel_limits(
            gas_orbs, self._effective_nelecas(), gas_restr)
        self.internal_rotation = len(gas_orbs) > 1
        self.fcisolver.mol = self.mol
        return self

    def close(self):
        """Release Newton-owned GAS helper plans; repeated calls are safe."""

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
        """Return this already-Newton GASSCF object after validation."""

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
                "Multiple roots are detected in fcisolver.  Newton GASSCF "
                "does not yet know which state to optimize.\n"
                "Use a state-specific solver or wait for staged state-average "
                "support.")
        return e_tot, e_gas, ci

    gen_g_hop = newton_casscf.gen_g_hop

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        """Run full Newton GASSCF orbital optimization with native CIAH.

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
        try:
            return super().kernel(mo_coeff, ci0, callback)
        finally:
            self.close()

    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        _unsupported("two-step Newton GASSCF kernel")

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
        """Whether Newton GASSCF may reuse owned GAS helper plans."""

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
