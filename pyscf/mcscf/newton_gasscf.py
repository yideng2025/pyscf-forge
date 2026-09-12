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

from pyscf import lib
from pyscf.mcscf import addons_gas
from pyscf.mcscf import fci_gas
from pyscf.mcscf import newton_casscf

__all__ = ["GASSCF"]


def _unsupported(feature):
    raise NotImplementedError(feature + " is not implemented for Newton GASSCF")


class _GASFCISolver(fci_gas.FCISolver):
    """GASCI solver shell reserved for Newton GASSCF-owned caches.

    At this stage the class only records the future cache policy.  Later
    review blocks will add contraction, RDM and spin-plan lifecycle handling
    here, while ordinary GASCI remains implemented by :mod:`fci_gas`.
    """

    _keys = set(fci_gas.FCISolver._keys) | {"cache_plans"}

    def __init__(self, mol=None, gas_orbs=None, gas_restr=None,
                 gas_restr_type=addons_gas.GAS_RESTR_SPIN_SUPERGROUP,
                 lib=None, *, cache_plans=True):
        super().__init__(
            mol, gas_orbs=gas_orbs, gas_restr=gas_restr,
            gas_restr_type=gas_restr_type, lib=lib)
        self.cache_plans = bool(cache_plans)


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


def _adapt_solver(fcisolver, cache_plans):
    """Return a Newton-GASSCF solver shell for an explicit GASCI solver.

    Full solver adaptation will be added in the next review block.  Keeping the
    guard here avoids silently accepting a solver whose cache ownership has not
    been audited yet.
    """

    _unsupported("explicit fcisolver adaptation")


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
            solver = _adapt_solver(fcisolver, cache_plans)

        super().__init__(
            mf, sum(solver.gas_orbs), nelecas, ncore=ncore, frozen=frozen)
        self.fcisolver = solver
        self.fcisolver.mol = self.mol

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
