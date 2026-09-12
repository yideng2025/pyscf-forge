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

This module is introduced in small reviewable stages.  The current stage only
provides the public object skeleton and GASCI solver ownership convention.  It
does not yet replace the native Newton/CIAH orbital-gradient machinery.
"""

import numpy

from pyscf import lib
from pyscf.fci import addons as fci_addons
from pyscf.mcscf import addons
from pyscf.mcscf import addons_gas
from pyscf.mcscf import fci_gas
from pyscf.mcscf import newton_casscf

__all__ = ["GASSCF"]


def _unsupported(feature):
    raise NotImplementedError(feature + " is not implemented for Newton GASSCF")


class _GASFCISolver(fci_gas.FCISolver):
    """GASCI solver shell reserved for Newton GASSCF-owned caches.

    At this stage the class records the future cache policy and supports safe
    adaptation from ordinary :class:`fci_gas.FCISolver` objects.  Later review
    blocks will add contraction, RDM and spin-plan lifecycle handling here,
    while ordinary GASCI remains implemented by :mod:`fci_gas`.
    """

    _keys = set(fci_gas.FCISolver._keys) | {"cache_plans"}

    def __init__(self, mol=None, gas_orbs=None, gas_restr=None,
                 gas_restr_type=addons_gas.GAS_RESTR_SPIN_SUPERGROUP,
                 lib=None, *, cache_plans=True):
        super().__init__(
            mol, gas_orbs=gas_orbs, gas_restr=gas_restr,
            gas_restr_type=gas_restr_type, lib=lib)
        self.cache_plans = bool(cache_plans)
        self._init_plan_cache()

    def _init_plan_cache(self):
        """Detach future Newton-owned helper plans from any source object.

        The staged skeleton has no live helper plans yet.  This hook is kept
        here so that explicit-solver adaptation and later plan-lifecycle code
        share one ownership boundary: adapted or copied Newton solvers always
        start with empty Newton-owned caches.
        """

        return None

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
