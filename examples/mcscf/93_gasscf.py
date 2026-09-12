#!/usr/bin/env python
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""GASSCF orbital optimization with a restricted GAS active space."""

from pyscf import gto
from pyscf import scf
from pyscf.mcscf import gasscf


mol = gto.M(
    atom="; ".join("H 0 0 %.1f" % z for z in (0.0, 0.8, 1.8, 2.6)),
    basis="sto-3g",
    spin=0,
    verbose=0,
)
mf = scf.RHF(mol).run()

# The two active orbitals are split into two one-orbital GAS spaces.  The
# cumulative bounds enforce one electron after GAS1 and two electrons after
# GAS2, equivalent to one electron in each GAS space.
mc = gasscf.GASSCF(
    mf, gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
    gas_restr_type="cumulative-occ", nelecas=(1, 1), ncore=1)
mc.verbose = 4
e_tot, e_gas, ci, mo, mo_energy = mc.kernel()

print("GASSCF total energy        = %.12f" % e_tot)
print("GASSCF active-space energy = %.12f" % e_gas)
print("GASSCF converged           = %s" % mc.converged)
print("GAS determinant count             = %d" % ci.size)
print("GAS space information             = %s" % mc.gas_space_info())
