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

import numpy

from pyscf import gto
from pyscf import scf
from pyscf.fci import direct_spin1
from pyscf.mcscf import addons_gas
from pyscf.mcscf import fci_gas
from pyscf.mcscf import gasci
from pyscf.mcscf import newton_casscf
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

    def test_gas_orbital_rotation_mask_enables_only_intergas_internal_rotations(self):
        mol = gto.M(
            atom="; ".join("H 0 0 %g" % value for value in range(6)),
            basis="sto-3g",
            verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(1, 2, 1), gas_restr=None,
            nelecas=(2, 2), ncore=1)

        mask = mc.uniq_var_indices(6, 1, 4, None)

        self.assertEqual(mask.shape, (6, 6))
        self.assertEqual(numpy.count_nonzero(mask), 14)

        # Native CASSCF-like external/core rotations are retained.
        self.assertTrue(mask[1, 0])
        self.assertTrue(mask[4, 0])
        self.assertTrue(mask[5, 0])
        self.assertTrue(mask[5, 1])
        self.assertTrue(mask[5, 4])

        # Active-active rotations are enabled only between GAS subspaces in the
        # lower-triangular orbital-rotation convention used by pack_uniq_var.
        self.assertTrue(mask[2, 1])
        self.assertTrue(mask[3, 1])
        self.assertTrue(mask[4, 1])
        self.assertTrue(mask[4, 2])
        self.assertTrue(mask[4, 3])

        # Same-subspace active rotations and opposite triangular entries remain
        # inactive.
        self.assertFalse(mask[3, 2])
        self.assertFalse(mask[1, 2])
        self.assertFalse(mask[0, 1])
        self.assertFalse(mask[1, 5])

    def test_gas_as_cas_mask_has_no_active_internal_rotation(self):
        mol = gto.M(
            atom="; ".join("H 0 0 %g" % value for value in range(6)),
            basis="sto-3g",
            verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(4,), gas_restr=None, nelecas=(2, 2), ncore=1)

        mask = mc.uniq_var_indices(6, 1, 4, None)

        self.assertEqual(numpy.count_nonzero(mask), 9)
        active = mask[1:5, 1:5]
        self.assertFalse(numpy.any(active[numpy.tril_indices(4, -1)]))

    def test_gas_orbital_rotation_mask_honors_extrasym_and_frozen(self):
        mol = gto.M(
            atom="; ".join("H 0 0 %g" % value for value in range(6)),
            basis="sto-3g",
            verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(1, 2, 1), gas_restr=None,
            nelecas=(2, 2), ncore=1)
        mc.extrasym = numpy.asarray([0, 0, 1, 1, 0, 0])

        mask = mc.uniq_var_indices(6, 1, 4, [2])

        self.assertFalse(numpy.any(mask[2]))
        self.assertFalse(numpy.any(mask[:, 2]))
        self.assertTrue(mask[4, 1])
        self.assertTrue(mask[5, 4])
        self.assertFalse(mask[3, 1])
        self.assertFalse(mask[5, 3])

    def test_explicit_gasci_solver_is_adapted_by_copy(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        solver = fci_gas.FCISolver(
            mol, gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ")
        solver.nroots = 2
        solver.spin = 0

        mc = newton_gasscf.GASSCF(
            mf, fcisolver=solver, nelecas=(1, 1), ncore=0,
            cache_plans=False)

        self.assertIsNot(mc.fcisolver, solver)
        self.assertIsInstance(mc.fcisolver, fci_gas.FCISolver)
        self.assertEqual(mc.ncas, 2)
        self.assertEqual(mc.ngas, 2)
        self.assertEqual(mc.fcisolver.gas_orbs, (1, 1))
        self.assertEqual(mc.fcisolver.gas_restr, [[1, 1], [2, 2]])
        self.assertEqual(mc.fcisolver.gas_restr_type, "cumulative-occ")
        self.assertEqual(mc.fcisolver.nroots, 2)
        self.assertEqual(mc.fcisolver.spin, 0)
        self.assertFalse(mc.cache_plans)
        self.assertFalse(hasattr(solver, "cache_plans"))

    def test_explicit_newton_gas_solver_is_copied(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        source = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=0,
            cache_plans=False).fcisolver

        mc = newton_gasscf.GASSCF(
            mf, fcisolver=source, nelecas=(1, 1), ncore=0)

        self.assertIsNot(mc.fcisolver, source)
        self.assertEqual(mc.gas_orbs, (2,))
        self.assertFalse(mc.cache_plans)
        source.cache_plans = True
        self.assertFalse(mc.cache_plans)

    def test_explicit_solver_model_arguments_are_rejected(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        solver = fci_gas.FCISolver(mol, gas_orbs=(2,))

        with self.assertRaisesRegex(ValueError, "explicit fcisolver"):
            newton_gasscf.GASSCF(
                mf, gas_orbs=(2,), fcisolver=solver,
                nelecas=(1, 1), ncore=0)

    def test_explicit_solver_requires_gas_orbs(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        solver = fci_gas.FCISolver(mol)

        with self.assertRaisesRegex(ValueError, "explicit GASCI solver"):
            newton_gasscf.GASSCF(
                mf, fcisolver=solver, nelecas=(1, 1), ncore=0)

    def test_rejects_external_fci_solver(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        solver = direct_spin1.FCISolver(mol)

        with self.assertRaisesRegex(NotImplementedError, "external/non-GASCI"):
            newton_gasscf.GASSCF(
                mf, fcisolver=solver, nelecas=(1, 1), ncore=0)

    def test_owned_rdm_and_spin_plans_are_reused_and_closed(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))

        rdm_plan = solver._get_rdm_plan(2, (1, 1))
        spin_plan = solver._get_spin_plan(2, (1, 1))

        self.assertIs(rdm_plan, solver._get_rdm_plan(2, (1, 1)))
        self.assertIs(spin_plan, solver._get_spin_plan(2, (1, 1)))
        self.assertEqual(rdm_plan.ndet, spin_plan.ndet)
        self.assertIsNotNone(solver._topology_key)

        solver.close()
        self.assertIsNone(rdm_plan._plan)
        self.assertIsNone(rdm_plan.gas)
        self.assertIsNone(solver._rdm_plan)
        self.assertIsNone(solver._spin_plan)
        self.assertIsNone(solver._topology_key)

    def test_topology_change_invalidates_owned_plans(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))
        old_plan = solver._get_rdm_plan(2, (1, 1))

        solver.gas_orbs = (1, 1)
        solver.gas_restr = [[1, 1], [2, 2]]
        solver.gas_restr_type = "cumulative-occ"
        new_plan = solver._get_rdm_plan(2, (1, 1))

        self.assertIsNot(old_plan, new_plan)
        self.assertIsNone(old_plan._plan)
        self.assertIsNone(old_plan.gas)
        solver.close()

    def test_contract_plan_cache_reuses_and_evicts_plans(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))
        eri0 = numpy.zeros((3, 3))
        plan0 = solver._get_contract_plan(eri0, 2, (1, 1))
        self.assertIs(plan0, solver._get_contract_plan(eri0.copy(), 2, (1, 1)))

        ci = numpy.random.default_rng(5).normal(size=plan0.ndet)
        numpy.testing.assert_allclose(
            plan0.contract(ci), solver.contract_2e(eri0, ci, 2, (1, 1)),
            atol=1e-12, rtol=0)

        for scale in (1.0, 2.0, 3.0):
            solver._get_contract_plan(
                eri0 + numpy.eye(3) * scale, 2, (1, 1))
        self.assertLessEqual(
            len(solver._contract_plans), solver._MAX_CONTRACT_PLANS)
        self.assertIsNone(plan0._plan)
        solver.close()
        self.assertEqual(len(solver._contract_plans), 0)
        self.assertIsNone(solver._contract_space)

    def test_copy_detaches_owned_plan_caches(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))
        rdm_plan = solver._get_rdm_plan(2, (1, 1))
        solver._get_spin_plan(2, (1, 1))
        solver._get_contract_plan(numpy.zeros((3, 3)), 2, (1, 1))

        copied = solver.copy()

        self.assertIsNone(copied._topology_key)
        self.assertIsNone(copied._rdm_plan)
        self.assertIsNone(copied._spin_plan)
        self.assertIsNone(copied._contract_space)
        self.assertEqual(len(copied._contract_plans), 0)
        self.assertIsNot(copied._contract_plans, solver._contract_plans)
        self.assertIsNotNone(rdm_plan._plan)

        copied.close()
        solver.close()

    def test_public_contract_2e_reuses_owned_plan(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))
        reference = fci_gas.FCISolver(gas_orbs=(2,))
        eri = numpy.zeros((3, 3))
        ci = numpy.random.default_rng(61).normal(size=4)

        contracted = solver.contract_2e(eri, ci, 2, (1, 1))
        plan = next(iter(solver._contract_plans.values()))
        contracted_again = solver.contract_2e(eri.copy(), ci, 2, (1, 1))

        self.assertIs(plan, next(iter(solver._contract_plans.values())))
        numpy.testing.assert_allclose(
            contracted, reference.contract_2e(eri, ci, 2, (1, 1)),
            atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(contracted_again, contracted,
                                      atol=1e-12, rtol=0)
        solver.close()

    def test_public_rdm_methods_reuse_owned_plan(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))
        reference = fci_gas.FCISolver(gas_orbs=(2,))
        rng = numpy.random.default_rng(62)
        bra = rng.normal(size=4)
        ket = rng.normal(size=4)

        dm1 = solver.trans_rdm1(bra, ket, 2, (1, 1))
        rdm_plan = solver._rdm_plan
        dm1s = solver.trans_rdm1s(bra, ket, 2, (1, 1))
        dm1b, dm2 = solver.trans_rdm12(bra, ket, 2, (1, 1))
        dm1s_b, dm2s = solver.trans_rdm12s(bra, ket, 2, (1, 1))

        self.assertIs(rdm_plan, solver._rdm_plan)
        numpy.testing.assert_allclose(
            dm1, reference.trans_rdm1(bra, ket, 2, (1, 1)),
            atol=1e-12, rtol=0)
        ref_dm1s = reference.trans_rdm1s(bra, ket, 2, (1, 1))
        for actual, expected in zip(dm1s, ref_dm1s):
            numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        ref_dm1, ref_dm2 = reference.trans_rdm12(bra, ket, 2, (1, 1))
        numpy.testing.assert_allclose(dm1b, ref_dm1, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(dm2, ref_dm2, atol=1e-12, rtol=0)
        ref_dm1s_b, ref_dm2s = reference.trans_rdm12s(
            bra, ket, 2, (1, 1))
        for actual, expected in zip(dm1s_b, ref_dm1s_b):
            numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        for actual, expected in zip(dm2s, ref_dm2s):
            numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        solver.close()

    def test_public_spin_square_methods_reuse_owned_plans(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))
        reference = fci_gas.FCISolver(gas_orbs=(2,))
        ci = numpy.random.default_rng(63).normal(size=4)

        ss_ci = solver.contract_ss(ci, 2, (1, 1))
        self.assertIsNotNone(solver._spin_plan)
        ss = solver.spin_square(ci, 2, (1, 1))
        self.assertIsNotNone(solver._rdm_plan)

        numpy.testing.assert_allclose(
            ss_ci, reference.contract_ss(ci, 2, (1, 1)),
            atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(
            numpy.asarray(ss), numpy.asarray(reference.spin_square(ci, 2, (1, 1))),
            atol=1e-12, rtol=0)
        solver.close()

    def test_cache_plans_false_delegates_public_methods(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,), cache_plans=False)
        eri = numpy.zeros((3, 3))
        ci = numpy.random.default_rng(64).normal(size=4)

        solver.contract_2e(eri, ci, 2, (1, 1))
        solver.make_rdm12(ci, 2, (1, 1))
        solver.contract_ss(ci, 2, (1, 1))

        self.assertIsNone(solver._contract_space)
        self.assertEqual(len(solver._contract_plans), 0)
        self.assertIsNone(solver._rdm_plan)
        self.assertIsNone(solver._spin_plan)

    def test_explicit_contract_plan_bypasses_owned_cache(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))
        eri = numpy.zeros((3, 3))
        ci = numpy.random.default_rng(65).normal(size=4)
        with solver.make_space(2, (1, 1), compress_links=True) as gas:
            with fci_gas.GasContractPlan(gas, eri) as plan:
                contracted = solver.contract_2e(
                    eri, ci, 2, (1, 1), plan=plan)
                numpy.testing.assert_allclose(
                    contracted, plan.contract(ci), atol=1e-12, rtol=0)
        self.assertIsNone(solver._contract_space)
        self.assertEqual(len(solver._contract_plans), 0)

    def test_gasscf_space_info_uses_gasci_normalization(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ", nelecas=(1, 1), ncore=0)

        info = mc.gas_space_info()
        metadata = info["metadata"]

        self.assertEqual(metadata["gas_orbs"], (1, 1))
        self.assertEqual(metadata["kernel_gas_orbs"], (1, 1))
        self.assertEqual(metadata["gas_restr_type"], "cumulative-occ")
        numpy.testing.assert_array_equal(
            metadata["gas_restr"], numpy.asarray([[1, 1], [2, 2]]))
        numpy.testing.assert_array_equal(
            metadata["spin_supergroups"],
            numpy.asarray([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=numpy.int32))
        self.assertEqual(info["core"]["ndet"], 2)

    def test_effective_nelecas_honors_solver_spin(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=2, ncore=0)

        self.assertEqual(mc._effective_nelecas(), (1, 1))
        mc.fcisolver.spin = 2
        self.assertEqual(mc._effective_nelecas(), (2, 0))

    def test_gasscf_gasdm_wrappers_match_solver_methods(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=0)
        ci = numpy.random.default_rng(71).normal(size=4)
        mc.ci = ci

        dm1s = mc.make_gasdm1s()
        dm1 = mc.make_gasdm1()
        dm1_ref, dm2_ref = mc.fcisolver.make_rdm12(ci, 2, (1, 1))
        dm1s_ref, dm2s_ref = mc.fcisolver.make_rdm12s(ci, 2, (1, 1))

        for actual, expected in zip(dm1s, dm1s_ref):
            numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(dm1, dm1_ref, atol=1e-12, rtol=0)
        dm1_from_pair, dm2 = mc.make_gasdm12()
        dm1s_from_pair, dm2s = mc.make_gasdm12s()
        numpy.testing.assert_allclose(
            dm1_from_pair, dm1_ref, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(dm2, dm2_ref, atol=1e-12, rtol=0)
        for actual, expected in zip(dm1s_from_pair, dm1s_ref):
            numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        for actual, expected in zip(dm2s, dm2s_ref):
            numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(mc.make_gasdm2(), dm2_ref,
                                      atol=1e-12, rtol=0)

    def test_gasscf_transition_dm_and_spin_wrappers_match_solver(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=0)
        rng = numpy.random.default_rng(72)
        bra = rng.normal(size=4)
        ket = rng.normal(size=4)
        mc.ci = ket

        numpy.testing.assert_allclose(
            mc.trans_gasdm1(bra, ket),
            mc.fcisolver.trans_rdm1(bra, ket, 2, (1, 1)),
            atol=1e-12, rtol=0)
        ref_dm1, ref_dm2 = mc.fcisolver.trans_rdm12(bra, ket, 2, (1, 1))
        dm1, dm2 = mc.trans_gasdm12(bra, ket)
        numpy.testing.assert_allclose(dm1, ref_dm1, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(dm2, ref_dm2, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(mc.trans_gasdm2(bra, ket), ref_dm2,
                                      atol=1e-12, rtol=0)

        ref_dm1s = mc.fcisolver.trans_rdm1s(bra, ket, 2, (1, 1))
        for actual, expected in zip(mc.trans_gasdm1s(bra, ket), ref_dm1s):
            numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        ref_dm1s_pair, ref_dm2s_pair = mc.fcisolver.trans_rdm12s(
            bra, ket, 2, (1, 1))
        dm1s_pair, dm2s_pair = mc.trans_gasdm12s(bra, ket)
        for actual, expected in zip(dm1s_pair, ref_dm1s_pair):
            numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        for actual, expected in zip(dm2s_pair, ref_dm2s_pair):
            numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(
            numpy.asarray(mc.spin_square(ket)),
            numpy.asarray(mc.fcisolver.spin_square(ket, 2, (1, 1))),
            atol=1e-12, rtol=0)

    def test_gasscf_property_wrappers_require_single_ci_vector(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=0)

        with self.assertRaisesRegex(ValueError, "CI vector is not available"):
            mc.make_gasdm1()
        with self.assertRaisesRegex(NotImplementedError, "state-averaged"):
            mc.make_gasdm1([numpy.ones(4), numpy.ones(4)])

    def test_newton_solver_hides_native_cas_only_hooks(self):
        base = fci_gas.FCISolver(gas_orbs=(2,))
        self.assertTrue(callable(getattr(base, "gen_linkstr", None)))
        self.assertTrue(callable(getattr(
            base, "transform_ci_for_orbital_rotation", None)))

        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))
        self.assertIsNone(getattr(solver, "gen_linkstr", None))
        self.assertIsNone(getattr(
            solver, "transform_ci_for_orbital_rotation", None))

        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        explicit = fci_gas.FCISolver(mol, gas_orbs=(2,))
        mc = newton_gasscf.GASSCF(
            mf, fcisolver=explicit, nelecas=(1, 1), ncore=0)
        self.assertIsNone(getattr(mc.fcisolver, "gen_linkstr", None))
        self.assertIsNone(getattr(
            mc.fcisolver, "transform_ci_for_orbital_rotation", None))

    def test_native_singleton_ci_list_is_accepted_by_rdm_dispatch(self):
        solver = newton_gasscf._GASFCISolver(gas_orbs=(2,))
        ci = numpy.random.default_rng(81).normal(size=4)

        dm1, dm2 = solver.make_rdm12([ci], 2, (1, 1))
        ref_dm1, ref_dm2 = solver.make_rdm12(ci, 2, (1, 1))

        numpy.testing.assert_allclose(dm1, ref_dm1, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(dm2, ref_dm2, atol=1e-12, rtol=0)
        with self.assertRaisesRegex(NotImplementedError,
                                    "multiroot CI density dispatch"):
            solver.make_rdm12([ci, ci], 2, (1, 1))
        solver.close()

    def test_fixed_orbital_gasci_bridge_matches_gasci_object(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=0)
        ref = gasci.GASCI(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None)
        ref.canonicalization = False

        e_tot, e_gas, ci, mo_coeff, mo_energy = mc.gasci(mf.mo_coeff)
        ref_e_tot, ref_e_gas, ref_ci, _, _ = ref.kernel(mf.mo_coeff)

        self.assertAlmostEqual(e_tot, ref_e_tot, places=11)
        self.assertAlmostEqual(e_gas, ref_e_gas, places=11)
        self.assertIs(mc.mo_coeff, mo_coeff)
        self.assertIs(mc.mo_energy, mo_energy)
        self.assertIs(mc.ci, ci)
        self.assertEqual(numpy.asarray(ci).shape, numpy.asarray(ref_ci).shape)
        self.assertTrue(mc.converged)

    def test_casci_bridge_returns_native_three_tuple(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=0)

        result = mc.casci(mf.mo_coeff)

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], mc.e_tot, places=12)
        self.assertAlmostEqual(result[1], mc.e_gas, places=12)
        self.assertIs(result[2], mc.ci)

    def test_casci_bridge_rejects_multiroot_energy(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=0)
        mc.fcisolver.nroots = 2

        with self.assertRaisesRegex(RuntimeError, "Multiple roots"):
            mc.casci(mf.mo_coeff)

    def test_full_kernel_gas_as_cas_smoke_matches_newton_casscf(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=1)
        ref = newton_casscf.CASSCF(mf, 2, (1, 1), ncore=1)
        for obj in (mc, ref):
            obj.max_cycle_macro = 12
            obj.max_cycle_micro = 4
            obj.conv_tol = 1e-9
            obj.conv_tol_grad = 1e-5
            obj.canonicalization = False

        e_tot, e_gas, ci, mo_coeff, mo_energy = mc.kernel(mf.mo_coeff)
        ref_e_tot, ref_e_cas, _, _, _ = ref.kernel(mf.mo_coeff)

        self.assertEqual(mo_coeff.shape, mf.mo_coeff.shape)
        self.assertIsNone(mo_energy)
        self.assertIs(mc.mo_coeff, mo_coeff)
        self.assertIs(mc.mo_energy, mo_energy)
        self.assertIs(mc.ci, ci)
        self.assertEqual(mc.converged, ref.converged)
        self.assertAlmostEqual(e_tot, ref_e_tot, places=7)
        self.assertAlmostEqual(e_gas, ref_e_cas, places=7)

    def test_mc2step_remains_guarded(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = newton_gasscf.GASSCF(
            mf, gas_orbs=(2,), gas_restr=None, nelecas=(1, 1), ncore=0)

        with self.assertRaisesRegex(NotImplementedError,
                                    "two-step Newton GASSCF kernel"):
            mc.mc2step()

    def test_requires_gas_orbs_without_explicit_solver(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        with self.assertRaisesRegex(ValueError, "gas_orbs is required"):
            newton_gasscf.GASSCF(mf, nelecas=(1, 1), ncore=0)


if __name__ == "__main__":
    print("Full Tests for staged Newton GASSCF skeleton")
    unittest.main()
