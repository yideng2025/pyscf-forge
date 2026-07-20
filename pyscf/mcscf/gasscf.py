"""Two-step CIAH orbital optimization for generalized active spaces.

The orbital gradient, Hessian action, augmented-Hessian solver, and integral
transformation are inherited from PySCF.  GASSCF owns the macro-iteration
driver, fixed-orbital GASCI steps, nonredundant rotations, and user output.
"""

import math

import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import addons
from pyscf.mcscf import addons_gas
from pyscf.mcscf import casci as casci_module
from pyscf.mcscf import fci_gas
from pyscf.mcscf import gasci
from pyscf.mcscf import mc1step


class _GasDFOutput:
    """GAS terminology layer for PySCF's density-fitting mixin."""

    __name_mixin__ = ""

    def dump_flags(self, verbose=None):
        GASSCF.dump_flags(self, verbose)
        logger.info(
            self, "GASCI/GASSCF: density fitting for JK matrix and "
            "2e integral transformation")
        return self

    def undo_df(self):
        obj = super().undo_df()
        return lib.view(
            obj, lib.drop_class(obj.__class__, _GasDFOutput))


# Adapted from pyscf.mcscf.mc2step.kernel under PySCF's Apache-2.0 license.
def _two_step_kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
                     ci0=None, callback=None, verbose=None, dump_chk=False):
    """GAS-owned macro driver around PySCF's two-step CIAH mathematics.

    Internal variable names and scheduling order deliberately follow
    :mod:`pyscf.mcscf.mc2step` so inherited schedulers and callbacks see the
    environment they expect.  GAS-specific changes are limited to the GASCI
    solve, RDM ownership, diagnostics, and user-visible terminology.
    """

    from pyscf.mcscf.addons import StateAverageMCSCFSolver

    if dump_chk:
        raise NotImplementedError(
            "checkpointing during GASSCF orbital optimization is not "
            "implemented")
    if verbose is None:
        verbose = casscf.verbose
    if callback is None:
        callback = casscf.callback
    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug("Start two-step GASSCF")
    mo = mo_coeff
    nmo = mo.shape[1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas
    eris = casscf.ao2mo(mo)
    e_tot, e_gas, fcivec = casscf.gasci(
        mo, ci0, eris, log, locals())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(tol)
        logger.info(casscf, "Set conv_tol_grad to %g", conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    de, elast = e_tot, e_tot
    totmicro = totinner = 0
    gasdm1 = 0
    r0 = None
    t2m = t1m = log.timer("Initializing two-step GASSCF", *cput0)
    imacro = 0

    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        njk = 0
        t3m = t2m
        gasdm1_old = gasdm1
        gasdm1, gasdm2 = casscf.fcisolver.make_rdm12(
            fcivec, ncas, casscf.nelecas)
        norm_ddm = numpy.linalg.norm(gasdm1 - gasdm1_old)
        # Compatibility aliases for PySCF schedulers and callbacks.
        casdm1, casdm2 = gasdm1, gasdm2
        t3m = log.timer("update GAS DM", *t3m)
        max_cycle_micro = 1
        max_stepsize = casscf.max_stepsize_scheduler(locals())

        for imicro in range(max_cycle_micro):
            rota = casscf.rotate_orb_cc(
                mo, lambda: fcivec, lambda: gasdm1, lambda: gasdm2,
                eris, r0, conv_tol_grad * .3, max_stepsize, log)
            u, g_orb, njk1, r0 = next(rota)
            rota.close()
            njk += njk1
            norm_t = numpy.linalg.norm(u - numpy.eye(nmo))
            norm_gorb = numpy.linalg.norm(g_orb)
            if imicro == 0:
                norm_gorb0 = norm_gorb
            de = numpy.dot(casscf.pack_uniq_var(u), g_orb)
            t3m = log.timer("orbital rotation", *t3m)
            eris = None
            u = u.copy()
            g_orb = g_orb.copy()
            mo = casscf.rotate_mo(mo, u, log)
            eris = casscf.ao2mo(mo)
            t3m = log.timer("update eri", *t3m)
            log.debug(
                "micro %2d  ~dE=%5.3g  |u-1|=%5.3g  |g[o]|=%5.3g  "
                "|dm1|=%5.3g", imicro, de, norm_t, norm_gorb, norm_ddm)
            if callable(callback):
                callback(locals())
            t2m = log.timer("micro iter %2d" % imicro, *t2m)
            if (norm_t < 1e-4 or abs(de) < tol * .4 or
                    norm_gorb < conv_tol_grad * .2):
                break

        totinner += njk
        totmicro += imicro + 1
        max_offdiag_u = numpy.abs(numpy.triu(u, 1)).max()
        if max_offdiag_u < casscf.small_rot_tol:
            small_rot = True
            log.debug(
                "Small orbital rotation, restart GASCI if supported by solver")
        else:
            small_rot = False
        if not isinstance(casscf, StateAverageMCSCFSolver):
            if not isinstance(fcivec, numpy.ndarray):
                fcivec = small_rot
        else:
            newvecs = []
            for subvec in fcivec:
                newvecs.append(small_rot if not isinstance(
                    subvec, numpy.ndarray) else subvec)
            fcivec = newvecs

        e_tot, e_gas, fcivec = casscf.gasci(
            mo, fcivec, eris, log, locals())
        log.timer("GASCI solver", *t3m)
        t2m = t1m = log.timer("macro iter %2d" % imacro, *t1m)
        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and norm_gorb < conv_tol_grad and
                norm_ddm < conv_tol_ddm and
                (max_offdiag_u < casscf.small_rot_tol or
                 casscf.small_rot_tol == 0)):
            conv = True
        if callable(callback):
            callback(locals())

    casscf._macro_iterations = imacro
    casscf._micro_iterations = totmicro
    casscf._jk_count = totinner
    if conv:
        log.info("two-step GASSCF converged in %3d macro "
                 "(%3d JK %3d micro) steps", imacro, totinner, totmicro)
    else:
        log.info("two-step GASSCF not converged, %3d macro "
                 "(%3d JK %3d micro) steps", imacro, totinner, totmicro)

    if casscf.canonicalization:
        log.info("GASSCF canonicalization")
        mo, fcivec, mo_energy = gasci.GASCI.canonicalize(
            casscf, mo, fcivec, eris, sort=casscf.sorting_mo_energy,
            gas_natorb=False, gasdm1=gasdm1, verbose=log)
    else:
        mo_energy = None
    log.timer("two-step GASSCF", *cput0)
    return conv, e_tot, e_gas, fcivec, mo, mo_energy


class GASSCF(mc1step.CASSCF, gasci.GASCI):
    """State-specific or state-average two-step CIAH GASSCF.

    Args:
        mf_or_mol : SCF object or :class:`pyscf.gto.Mole`
        ncas : int
            Total number of active orbitals.
        nelecas : int or pair of ints
            Number of active electrons.
        gas_orbs : sequence of ints, optional
            Number of orbitals in each ordered GAS subspace.
        gas_restr : object, optional
            Restriction in the format selected by ``gas_restr_type``.
        gas_restr_type : str
            One of ``spin-supergroup``, ``supergroup``, ``cumulative-occ``,
            or ``ras``.
        ncore : int, optional
            Number of doubly occupied core orbitals.
        frozen : int or sequence of ints, optional
            Orbitals excluded from orbital optimization.

    Notes:
        Every macro iteration fully solves the state-specific or
        state-average GASCI problem before the next CIAH orbital step.  The
        one-step CI-response and coupled orbital/CI Newton algorithms are not
        enabled.
    """

    _keys = mc1step.CASSCF._keys | gasci.GASCI._keys | {
        "gas_orbs", "gas_restr", "gas_restr_type",
    }

    def __init__(self, mf_or_mol, ncas, nelecas, gas_orbs=None,
                 gas_restr=None,
                 gas_restr_type=addons_gas.GAS_RESTR_SPIN_SUPERGROUP,
                 ncore=None, frozen=None):
        mc1step.CASSCF.__init__(
            self, mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
        self.gas_orbs = tuple(gas_orbs) if gas_orbs is not None else None
        self.gas_restr = gas_restr
        self.gas_restr_type = gas_restr_type
        self.fcisolver = fci_gas.FCISolver(
            getattr(self._scf, "mol", None), gas_orbs=self.gas_orbs,
            gas_restr=self.gas_restr, gas_restr_type=self.gas_restr_type)
        self.internal_rotation = True
        self._gas_ci_signature = None
        self._gas_full_ci_space = None
        self.e_spin_penalty = None
        self.e_tot_physical = None
        self.e_gas_physical = None
        self.spin_penalty_method = None
        self._macro_iterations = 0
        self._micro_iterations = 0
        self._jk_count = 0

    # GASSCF intentionally inherits PySCF's CASSCF orbital machinery before
    # GASCI in the MRO.  Bind every public method whose CAS and GAS semantics
    # differ explicitly here so that neither the current PySCF MRO nor a
    # future CASSCF override can silently select a CAS implementation.
    get_h1gas = gasci.GASCI.get_h1gas
    get_h2gas = gasci.GASCI.get_h2gas
    gas_space_info = gasci.GASCI.gas_space_info
    get_h1cas = gasci.GASCI.get_h1cas
    h1e_for_cas = gasci.GASCI.h1e_for_cas
    get_h2cas = gasci.GASCI.get_h2cas
    sort_mo = gasci.GASCI.sort_mo
    get_fock = gasci.GASCI.get_fock
    canonicalize = gasci.GASCI.canonicalize
    canonicalize_ = gasci.GASCI.canonicalize_
    cas_natorb = gasci.GASCI.cas_natorb
    cas_natorb_ = gasci.GASCI.cas_natorb_
    fix_spin = gasci.GASCI.fix_spin
    fix_spin_ = gasci.GASCI.fix_spin_
    state_specific = gasci.GASCI.state_specific
    state_specific_ = gasci.GASCI.state_specific_
    nuc_grad_method = gasci.GASCI.nuc_grad_method
    make_gasdm1 = gasci.GASCI.make_gasdm1
    make_gasdm1s = gasci.GASCI.make_gasdm1s
    make_gasdm12 = gasci.GASCI.make_gasdm12
    make_gasdm2 = gasci.GASCI.make_gasdm2
    make_gasdm12s = gasci.GASCI.make_gasdm12s
    trans_gasdm1 = gasci.GASCI.trans_gasdm1
    trans_gasdm1s = gasci.GASCI.trans_gasdm1s
    trans_gasdm12 = gasci.GASCI.trans_gasdm12
    trans_gasdm2 = gasci.GASCI.trans_gasdm2
    trans_gasdm12s = gasci.GASCI.trans_gasdm12s
    make_rdm1 = gasci.GASCI.make_rdm1
    make_rdm1s = gasci.GASCI.make_rdm1s
    spin_square = gasci.GASCI.spin_square
    get_gas_natorb = gasci.GASCI.get_gas_natorb
    get_gas_average_natorb = gasci.GASCI.get_gas_average_natorb
    get_gas_pseudo_natorb = gasci.GASCI.get_gas_pseudo_natorb
    get_gas_pseudo_natorb_occupations = \
        gasci.GASCI.get_gas_pseudo_natorb_occupations
    analyze = gasci.GASCI.analyze

    def _prepare_orbital_space(self):
        gas_orbs, gas_restr = self._normalized_restriction()
        limits = addons_gas.check_kernel_limits(
            gas_orbs, self.nelecas, gas_restr)
        na, nb = (int(value) for value in self.nelecas)
        full_ndet = math.comb(self.ncas, na) * math.comb(self.ncas, nb)
        self._gas_full_ci_space = (
            int(limits["ndet_estimate"]) == int(full_ndet))
        return gas_orbs, gas_restr, limits

    def check_sanity(self):
        """Validate GAS restrictions and the supported orbital workflow."""

        gasci.GASCI.check_sanity(self)
        self._prepare_orbital_space()
        if not self.internal_rotation:
            raise ValueError(
                "GASSCF requires internal_rotation=True because inter-GAS "
                "active-active rotations are nonredundant")
        state_average = isinstance(
            self.fcisolver, addons.StateAverageFCISolver)
        if state_average and not isinstance(
                self.fcisolver, addons_gas.StateAverageFCISolver):
            raise TypeError(
                "GASSCF state averaging requires the GAS-specific "
                "state_average() wrapper")
        if (not state_average and
                int(getattr(self.fcisolver, "nroots", 1)) != 1):
            raise NotImplementedError(
                "multiroot GASSCF requires state_average(weights)")
        return self

    def dump_flags(self, verbose=None):
        """Print GASCI and two-step CIAH optimization settings."""

        gasci.GASCI.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** GASSCF orbital optimization flags ********")
        log.info("orbital optimizer = two-step CIAH")
        weights = self._state_weights()
        if weights is None:
            log.info("orbital objective = state-specific")
        else:
            log.info("orbital objective = state-average")
            log.info("state weights = %s", weights)
        log.info("max_cycle_macro = %d", self.max_cycle_macro)
        log.info("conv_tol = %g", self.conv_tol)
        log.info("conv_tol_grad = %s", self.conv_tol_grad)
        log.info("orbital rotation max_stepsize = %g", self.max_stepsize)
        log.info("orbital rotation threshold for CI restart = %g",
                 self.small_rot_tol)
        log.info("augmented hessian ah_max_cycle = %d", self.ah_max_cycle)
        log.info("augmented hessian ah_conv_tol = %g", self.ah_conv_tol)
        log.info("augmented hessian ah_linear dependence = %g",
                 self.ah_lindep)
        log.info("augmented hessian ah_level_shift = %g",
                 self.ah_level_shift)
        log.info("augmented hessian ah_start_tol = %g", self.ah_start_tol)
        log.info("augmented hessian ah_start_cycle = %d",
                 self.ah_start_cycle)
        log.info("ao2mo_level = %d", self.ao2mo_level)
        log.info("intra-GAS rotations = redundant")
        log.info("inter-GAS rotations = %s",
                 "redundant (GAS space equals full CAS)" if
                 self._gas_full_ci_space else "optimized")
        log.info("checkpoint during orbital optimization = disabled")
        return self

    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        """Return the GAS-aware nonredundant orbital-rotation mask."""

        nocc = ncore + ncas
        mask = numpy.zeros((nmo, nmo), dtype=bool)
        mask[ncore:nocc, :ncore] = True
        mask[nocc:, :nocc] = True

        if self._gas_full_ci_space is None:
            self._prepare_orbital_space()
        if not self._gas_full_ci_space:
            block_sizes = ((int(ncas),) if self.gas_orbs is None else
                           tuple(int(value) for value in self.gas_orbs))
            starts = []
            offset = int(ncore)
            for size in block_sizes:
                starts.append(offset)
                offset += size
            if offset != nocc:
                raise ValueError("gas_orbs must sum to ncas")
            for right in range(1, len(block_sizes)):
                row = slice(starts[right], starts[right] + block_sizes[right])
                for left in range(right):
                    col = slice(starts[left], starts[left] + block_sizes[left])
                    mask[row, col] = True

        if self.extrasym is not None:
            extrasym = numpy.asarray(self.extrasym)
            if extrasym.ndim != 1 or extrasym.size != nmo:
                raise ValueError("extrasym must contain one label per MO")
            mask &= extrasym.reshape(-1, 1) == extrasym
        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = False
                mask[:, :frozen] = False
            else:
                frozen = numpy.asarray(frozen)
                mask[frozen] = False
                mask[:, frozen] = False
        return mask

    def _set_physical_energies(self):
        solver_physical = getattr(self.fcisolver, "e_physical", None)
        solver_penalty = getattr(self.fcisolver, "e_spin_penalty", None)
        self.spin_penalty_method = getattr(
            self.fcisolver, "spin_penalty_method", None)
        if solver_physical is None or solver_penalty is None:
            self.e_spin_penalty = None
            self.e_tot_physical = self.e_tot
            self.e_gas_physical = self.e_gas
            return
        physical = float(numpy.asarray(solver_physical))
        penalty = float(numpy.asarray(solver_penalty))
        core = float(self.e_tot) - float(self.e_gas)
        self.e_spin_penalty = penalty
        self.e_tot_physical = physical
        self.e_gas_physical = physical - core

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        """Optimize a state-specific or state-average GAS wavefunction."""

        self._sync_fcisolver()
        if mo_coeff is None:
            if self.mo_coeff is None and self._scf.mol.nelectron > 0:
                self._scf.run()
                self.mo_coeff = self._scf.mo_coeff
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if callback is None:
            callback = self.callback

        log = logger.new_logger(self, self.verbose)
        self._check_mo_orthonormality(mo_coeff, log)
        self.check_sanity()
        self.dump_flags(log)

        signature = self._gas_problem_signature()
        if ci0 is None:
            if self._ci_matches_signature(self.ci, signature):
                ci0 = self.ci
            else:
                self._clear_ci_guess()

        nmo = mo_coeff.shape[1]
        mask = self.uniq_var_indices(
            nmo, self.ncore, self.ncas, self.frozen)
        if numpy.any(mask):
            results = _two_step_kernel(
                self, mo_coeff, tol=self.conv_tol,
                conv_tol_grad=self.conv_tol_grad, ci0=ci0,
                callback=callback, verbose=self.verbose)
        else:
            eris = self.ao2mo(mo_coeff)
            e_tot, e_gas, ci = self.gasci(
                mo_coeff, ci0, eris, log, {"imacro": 0})
            results = (True, e_tot, e_gas, ci, mo_coeff, None)

        (self.converged, self.e_tot, self.e_cas, self.ci,
         self.mo_coeff, self.mo_energy) = results
        self._gas_ci_signature = signature
        self._set_physical_energies()
        weights = self._state_weights()
        if weights is None:
            logger.note(self, "GASSCF energy = %#.15g", self.e_tot)
            try:
                ss, multiplicity = self.fcisolver.spin_square(
                    self.ci, self.ncas, self.nelecas)
            except NotImplementedError:
                pass
            else:
                logger.note(self, "GASSCF E = %#.15g  E(GASCI) = %#.15g  "
                            "S^2 = %.7f  multiplicity = %.7f",
                            self.e_tot, self.e_gas, ss, multiplicity)
        else:
            totals = numpy.asarray(
                self.fcisolver.e_states, dtype=numpy.float64).reshape(-1)
            roots = self.ci if isinstance(
                self.ci, (list, tuple)) else [self.ci]
            if totals.size != weights.size or len(roots) != weights.size:
                raise RuntimeError(
                    "state-average GASSCF root count does not match weights")
            energy_core = float(self.e_tot) - float(self.e_gas)
            spin_values = self.fcisolver.states_spin_square(
                roots, self.ncas, self.nelecas)
            logger.note(self, "GASSCF state-averaged energy = %#.15g",
                        self.e_tot)
            logger.note(self, "GASSCF energy for each state")
            for state, (weight, total, ss, multiplicity) in enumerate(zip(
                    weights, totals, spin_values[0], spin_values[1])):
                logger.note(
                    self, "  State %d weight %g  E = %#.15g  "
                    "E(GASCI) = %#.15g  S^2 = %.7f  multiplicity = %.7f",
                    state, weight, total, total - energy_core,
                    ss, multiplicity)
        return self.e_tot, self.e_gas, self.ci, self.mo_coeff, self.mo_energy

    def density_fit(self, auxbasis=None, with_df=None):
        """Return a density-fitted GASSCF object.

        The JK build and integral transformation are provided by PySCF's
        MCSCF density-fitting mixin.  This wrapper changes only the
        user-visible CAS terminology and preserves ``undo_df()``.
        """

        from pyscf.mcscf import df

        obj = df.density_fit(self, auxbasis=auxbasis, with_df=with_df)
        if isinstance(obj, _GasDFOutput):
            return obj
        return lib.set_class(obj, (_GasDFOutput, obj.__class__))

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        """Run the supported two-step CIAH GASSCF optimizer."""

        return self.kernel(mo_coeff, ci0, callback)

    def mc1step(self, *args, **kwargs):
        raise NotImplementedError(
            "one-step GASSCF CI response is not implemented; use mc2step() "
            "or kernel()")

    def gasci(self, mo_coeff=None, ci0=None, eris=None, verbose=None,
              envs=None):
        """Run one fixed-orbital GASCI step inside a GASSCF object."""

        self._sync_fcisolver()
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        log = logger.new_logger(self, verbose)
        fgasci = mc1step._fake_h_for_fast_casci(self, mo_coeff, eris)
        e_tot, e_gas, ci = casci_module.kernel(
            fgasci, mo_coeff, ci0, log, envs=envs)
        if not isinstance(e_gas, (float, numpy.number)):
            raise RuntimeError(
                "Multiple roots were returned by the GASCI solver. "
                "Use state_average(weights) to define the orbital objective.")
        if numpy.ndim(e_gas) != 0:
            e_gas = e_gas[0]

        if envs is not None and log.verbose >= logger.INFO:
            log.debug("GAS space CI energy = %#.15g", e_gas)
            try:
                spin = self.fcisolver.spin_square(
                    ci, self.ncas, self.nelecas)
            except (AttributeError, NotImplementedError):
                spin = None
            if "imicro" in envs:
                if spin is None:
                    log.info(
                        "macro iter %3d (%3d JK  %3d micro), "
                        "GASSCF E = %#.15g  dE = % .8e",
                        envs["imacro"], envs["njk"], envs["imicro"],
                        e_tot, e_tot - envs["elast"])
                else:
                    log.info(
                        "macro iter %3d (%3d JK  %3d micro), "
                        "GASSCF E = %#.15g  dE = % .8e  S^2 = %.7f",
                        envs["imacro"], envs["njk"], envs["imicro"],
                        e_tot, e_tot - envs["elast"], spin[0])
                log.info(
                    "               |grad[o]|=%5.3g  |ddm|=%5.3g  "
                    "|maxRot[o]|=%5.3g", envs["norm_gorb0"],
                    envs["norm_ddm"], envs["max_offdiag_u"])
            elif spin is None:
                log.info("GASCI E = %#.15g", e_tot)
            else:
                log.info("GASCI E = %#.15g  S^2 = %.7f", e_tot, spin[0])
        return e_tot, e_gas, ci

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        """Compatibility hook required by inherited PySCF CIAH routines."""

        return self.gasci(mo_coeff, ci0, eris, verbose, envs)

    def state_average(self, weights=(0.5, 0.5), wfnsym=None):
        """Return a state-average GASSCF object with PySCF semantics."""

        return addons_gas.state_average(self, weights, wfnsym)

    def state_average_(self, weights=(0.5, 0.5), wfnsym=None):
        """Apply state averaging to this GASSCF object in place."""

        addons_gas.state_average_(self, weights, wfnsym)
        return self

    def state_average_mix(self, *args, **kwargs):
        raise NotImplementedError(
            "state_average_mix is not implemented for GASSCF")

    state_average_mix_ = state_average_mix

    def as_scanner(self):
        raise NotImplementedError(
            "the GASSCF geometry scanner is not implemented")

    def newton(self, *args, **kwargs):
        raise NotImplementedError(
            "the inherited coupled orbital/CI Newton solver is not valid for "
            "the GAS determinant space; use the two-step GASSCF CIAH kernel")

    def to_gpu(self, *args, **kwargs):
        raise NotImplementedError(
            "the libfci_gas C/OpenMP backend does not support GPU execution")

    def reset(self, mol=None):
        """Reset molecular state and invalidate cached GAS iteration data."""

        gasci.GASCI.reset(self, mol)
        self._max_stepsize = None
        self._gas_full_ci_space = None
        self._macro_iterations = 0
        self._micro_iterations = 0
        self._jk_count = 0
        return self
