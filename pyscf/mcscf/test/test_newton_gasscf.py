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

"""Small construction tests for the staged Newton GASSCF module."""

import unittest

from pyscf import gto
from pyscf import scf
from pyscf.mcscf import addons_gas
from pyscf.mcscf import fci_gas
from pyscf.mcscf import newton_gasscf


class KnownValues(unittest.TestCase):

    def test_constructs_newton_gasscf_with_gasci_solver(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.75",
            basis="sto-3g",
            verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ", nelecas=(1, 1), ncore=0)

        self.assertEqual(mc.ncas, 2)
        self.assertEqual(mc.nelecas, (1, 1))
        self.assertEqual(mc.ncore, 0)
        self.assertEqual(mc.ngas, 2)
        self.assertIsInstance(mc.fcisolver, fci_gas.FCISolver)
        self.assertEqual(mc.gas_orbs, (1, 1))
        self.assertEqual(mc.fcisolver.gas_orbs, (1, 1))
        self.assertEqual(mc.gas_restr, [[1, 1], [2, 2]])
        self.assertEqual(mc.fcisolver.gas_restr, [[1, 1], [2, 2]])
        self.assertEqual(mc.gas_restr_type, "cumulative-occ")
        self.assertEqual(mc.fcisolver.gas_restr_type, "cumulative-occ")
        self.assertTrue(mc.cache_plans)

    def test_default_restriction_type_matches_gasci(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=0,
            cache_plans=False)

        self.assertEqual(
            mc.gas_restr_type, addons_gas.GAS_RESTR_SPIN_SUPERGROUP)
        self.assertFalse(mc.cache_plans)

    def test_requires_gas_orbs_without_explicit_solver(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        with self.assertRaisesRegex(ValueError, "gas_orbs is required"):
            newton_gasscf.GASSCF(mf, nelecas=(1, 1), ncore=0)


if __name__ == "__main__":
    print("Full Tests for staged Newton GASSCF skeleton")
    unittest.main()
