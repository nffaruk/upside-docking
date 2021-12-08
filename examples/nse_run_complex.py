#!/usr/bin/env python

import os, sys, time, shutil, gc
from timeit import default_timer as timer
import subprocess as sp
import cPickle as cp
import multiprocessing
import numpy as np
import tables as tb
import mdtraj as md
from progress.bar import Bar 

base_dir = os.path.dirname(os.path.realpath(__file__)) # Examples directory
up_dir = os.path.dirname(base_dir) # Upside repo location <<
upside_utils_dir = os.path.join(up_dir, "py/")
sys.path.append(upside_utils_dir)
import run_upside as ru
from upside_config import add_docking_terms as add_nrg_term
import mdtraj_upside as mu
import upside_engine as ue

scwrl_dir = os.path.expanduser("~/software/scwrl4/") # SCWRL4 location <<

def get_bb3(traj):
    """Extract trajectory of just the three main backbone atoms."""
    bb3_sele = [a.index for a in traj.top.atoms if a.name in ['N', 'CA', 'C']]
    traj = traj.atom_slice(bb3_sele)
    for i, res in enumerate(traj.top.residues):
        res.index = i

    return traj

def slice_renum(ref, traj, a_sele_ref, a_sele_t):
    """Slice trajectories according to selections and renumber residues."""
    ref = ref.atom_slice(a_sele_ref)
    traj = traj.atom_slice(a_sele_t)

    for i, res in enumerate(ref.top.residues):
        res.index = i
    for i, res in enumerate(traj.top.residues):
        res.index = i

    return ref, traj

def rl_align_split(r_pdb_x, l_pdb_x, traj, rl_chains):
    """Align trajectory to reference and slice out sequence matched substituents."""
    ref = r_pdb_x.stack(l_pdb_x)
    n_ch_ref_pre = ref.n_chains
    n_ch_traj_pre = traj.n_chains

    r_sele_bb_n, r_sele_bb_t = mu.get_r_sele_bb(r_pdb_x, l_pdb_x, traj, rl_chains)
    traj = traj.superpose(ref, atom_indices=r_sele_bb_t, ref_atom_indices=r_sele_bb_n)

    a_sele_n, a_sele_t = mu.rl_align_slice(r_pdb_x, l_pdb_x, traj, rl_chains)
    ref, traj = slice_renum(ref, traj, a_sele_n, a_sele_t)
    n_ch_ref_post = ref.n_chains
    n_ch_traj_post = traj.n_chains

    # Account for missing chains between traj and ref
    if n_ch_ref_post == n_ch_ref_pre:
        ch_r = range(r_pdb_x.n_chains)
        ch_l = range(r_pdb_x.n_chains, r_pdb_x.n_chains+l_pdb_x.n_chains)
        res_r = np.array([res.index for ch in ch_r for res in ref.top.chain(ch).residues])
        res_l = np.array([res.index for ch in ch_l for res in ref.top.chain(ch).residues])
    elif n_ch_traj_post == n_ch_traj_pre:
        ch_r = range(rl_chains[0])
        ch_l = range(rl_chains[0], rl_chains[0]+rl_chains[1])
        res_r = np.array([res.index for ch in ch_r for res in traj.top.chain(ch).residues])
        res_l = np.array([res.index for ch in ch_l for res in traj.top.chain(ch).residues])
    else:
        raise ValueError("Chains dropped after align slice of bound and unbound")

    r_sele_n = [a.index for res in res_r for a in ref.top.residue(res).atoms]
    l_sele_n = [a.index for res in res_l for a in ref.top.residue(res).atoms]
    r_pdb_match = ref.atom_slice(r_sele_n)
    l_pdb_match = ref.atom_slice(l_sele_n)

    r_sele_d = [a.index for res in res_r for a in traj.top.residue(res).atoms]
    l_sele_d = [a.index for res in res_l for a in traj.top.residue(res).atoms]
    r_decoy = traj.atom_slice(r_sele_d)
    l_decoy = traj.atom_slice(l_sele_d)

    return ref, traj, r_pdb_match, l_pdb_match, r_decoy, l_decoy, res_r, res_l

def mp_scwrl((pdb_id, traj_frame, frame, scwrl_temp_dir)):
    """Parallel SCWRLing of Upside trajectory frames."""
    pdb_out_fn = scwrl_temp_dir + "{}_{}.pdb".format(pdb_id, frame)
    traj_frame.save(pdb_out_fn)
    scwrl_cmd = "{0}Scwrl4 -h -i {1} -o {1}".format(scwrl_dir, pdb_out_fn)
    scwrl_out = sp.check_output(scwrl_cmd, shell=True)

    return True

def irmsd_calc(r_pdb_x, l_pdb_x, traj, rl_chains):
    """Calculate IRMSD over the trajectory given reference structures of the substituents."""
    # Align to reference
    ref = r_pdb_x.stack(l_pdb_x)
    n_ch_ref_pre = ref.n_chains
    n_ch_traj_pre = traj.n_chains

    r_sele_bb_n, r_sele_bb_t = mu.get_r_sele_bb(r_pdb_x, l_pdb_x, traj, rl_chains)
    traj = traj.superpose(ref, atom_indices=r_sele_bb_t, ref_atom_indices=r_sele_bb_n)

    a_sele_n, a_sele_t = mu.rl_align_slice(r_pdb_x, l_pdb_x, traj, rl_chains)
    ref, traj = slice_renum(ref, traj, a_sele_n, a_sele_t)
    n_ch_ref_post = ref.n_chains
    n_ch_traj_post = traj.n_chains

    # Account for missing chains between traj and ref
    if n_ch_ref_post == n_ch_ref_pre:
        ch_r = range(r_pdb_x.n_chains)
        ch_l = range(r_pdb_x.n_chains, r_pdb_x.n_chains+l_pdb_x.n_chains)
        res_r = np.array([res.index for ch in ch_r for res in ref.top.chain(ch).residues])
        res_l = np.array([res.index for ch in ch_l for res in ref.top.chain(ch).residues])
    elif n_ch_traj_post == n_ch_traj_pre:
        ch_r = range(rl_chains[0])
        ch_l = range(rl_chains[0], rl_chains[0]+rl_chains[1])
        res_r = np.array([res.index for ch in ch_r for res in traj.top.chain(ch).residues])
        res_l = np.array([res.index for ch in ch_l for res in traj.top.chain(ch).residues])
    else:
        raise ValueError("Chains dropped after align slice of bound and unbound")

    # Compute
    irmsd_compy = mu.bb_interfacial_rmsd_angstroms(ref, res_r, res_l)
    irmsd = irmsd_compy.compute_irmsd(traj)

    return irmsd

def mp_recalc_pe((config_fn, frame)):
    """Parallel recaulcation of Upside trajectory energies."""
    engine = ue.Upside(config_fn)
    with tb.open_file(config_fn) as t:
        pos = t.root.output.pos[frame, 0, :]
        pe = engine.energy(pos)
        rot = engine.get_output("inter_rotamer")[0, 0]
        env = engine.get_output("inter_nonlinear_coupling_environment")[0, 0]

    return pe, rot, env

## General Settings and Paths
is_unbound = False # Whether using the unbound or bound conformations of the substituents
default_pe = False # Whether only using the folding forcefield
do_analysis = True
do_ana_only = True
n_rep = 3 # Number of simulation replicates
pool_size = 4 # Number of subprocesses for multiprocessing. Used for SCWRLing and post-sim energy re-evaluation.
seed = 1 # Random seed to use for sims

pdb_dir = os.path.join(base_dir, "PDB/")
rl_pdb_dir = pdb_dir
up_input_dir = os.path.join(base_dir, "output/")
run_dir = up_input_dir
output_dir = up_input_dir

pdb_id = "2OOB"
if is_unbound:
    rl_suff = "u"
    pdb_id_mod = pdb_id + "_u"
else:
    rl_suff = "b"
    pdb_id_mod = pdb_id
print "NSE run for {}...".format(pdb_id_mod)

# Params
param_dir_base = "{}/parameters/".format(up_dir) 
param_dir_common = os.path.join(param_dir_base, "common/")
param_dir_ff1 = os.path.join(param_dir_base, "ff_1.5/")
param_dir_dock = os.path.join(param_dir_base, "docking/")

with open(param_dir_ff1 + "sheet") as f:
    sheet_mix = float(f.read())
with open(param_dir_ff1 + "hbond") as f:
    hb = float(f.read())

rot_param_init = param_dir_dock + "inter_rot.h5"
env_param_dict_init = param_dir_dock + "wenv_param_dict.pkl"

# Load inter sc rotamer and env params
if not default_pe:
    with tb.open_file(rot_param_init) as tpf:
        rot_param = tpf.root.interaction_param[:]

    with open(env_param_dict_init, 'rb') as f:
        env_param_dict = cp.load(f)

print "Input generation..."
pdb_fn = os.path.join(pdb_dir, "{}.pdb".format(pdb_id))
input_fn = os.path.join(up_input_dir, pdb_id_mod)

# Number of receptor and ligand chains
r_ch = md.load(rl_pdb_dir + pdb_id + "_r_{}.pdb".format(rl_suff)).n_chains
l_ch = md.load(rl_pdb_dir + pdb_id + "_l_{}.pdb".format(rl_suff)).n_chains

# Input gen command
cmd = ("{}/PDB_to_initial_structure.py "
       "{} "
       "{} "
       "--record-chain-breaks "
       "--rl-chains={},{} " 
       "--allow-unexpected-chain-breaks"
       ).format(upside_utils_dir, pdb_fn, input_fn, r_ch, l_ch)
sp.check_output(cmd.split())

print "Configuring..."
# Config settings
kwargs = dict(
            sheet_mix_energy = sheet_mix,
            hbond = hb,
            dynamic_1body = True,
            rama_pot = os.path.join(param_dir_common, "rama.dat"),
            reference_rama = os.path.join(param_dir_common, "rama_reference.pkl"),
            placement = os.path.join(param_dir_ff1, "sidechain.h5"),
            rotamer_interaction_param = os.path.join(param_dir_ff1, "sidechain.h5"),
            environment = os.path.join(param_dir_ff1, "environment.h5"),
            )

# output_dir = run_dir #run_dir + "/{}/".format(pdb_id_mod)
config_fns = [os.path.join(output_dir, pdb_id_mod + ".{}.up".format(rep)) for rep in xrange(n_rep)]

if not do_ana_only:
    fasta_fn = input_fn + ".fasta"
    kwargs["init"] = input_fn + ".initial.pkl"
    kwargs["chain_break_from_file"] = input_fn + ".chain_breaks"

    config_cmd = ru.upside_config(fasta_fn, config_fns[0], **kwargs)
    print config_cmd
    sp.check_output(['python', '{}/ugly_hack_break_chain.py'.format(upside_utils_dir),
                                   '--config', config_fns[0], '--chain-break-from-file', '--ignore-weird-rama',
                                   '--no-jump', '--tether-fragments']) # --tether-fragments adds a spring between midchain breaks

    if not default_pe:
        add_nrg_term(config_fns[0], rot_param, env_param_dict)

    for rep in xrange(1, n_rep):
        shutil.copyfile(config_fns[0], config_fns[rep])  

    print "Running..."
    start_time = timer()
    job = ru.run_upside('in_process', config_fns, 1000., 2., time_limit=2*60*60, n_threads=n_rep,
                        temperature=[0.8]*n_rep, disable_recentering=True, verbose=True,
                        seed=seed)
    if job.wait() != 0:
        raise ValueError("Run failed.")
    else:
        del job
        gc.collect()
    print "Completed in {:.1f} s".format(timer()-start_time)

if do_analysis:
    print "Analyzing..."
    with tb.open_file(config_fns[0]) as t:
        rl_chains = t.root.input.chain_break.rl_chains_actual[:]

    irmsd_all = []
    lrmsd_all = []
    fnat_all = []
    for rep in xrange(n_rep):
        print "rep {}".format(rep)
        # with tb.open_file(config_fns[rep]) as t:
        #     n_frames = len(t.root.output.pos[:])
        traj = md.load(config_fns[rep])[:] #n_frames/2
        n_frames = traj.n_frames  

        if pdb_id == "1K4C" and is_unbound:
            best_mono_sele = traj.top.select("chainid < 3")
            traj = traj.atom_slice(best_mono_sele)
            for i, res in enumerate(traj.top.residues):
                res.index = i
            rl_chains = [2, 1]

        ## Perform SCWRL on each frame separately in parallel
        traj_fn = output_dir + pdb_id + "_scwrl.{}".format(rep)
        if not os.path.exists(traj_fn+".xtc"):
            start_time = timer()
            bar = Bar("SCWRLing...", max=n_frames)

            p = multiprocessing.Pool(pool_size)
            scwrl_temp_dir = output_dir + pdb_id_mod + "_scwrl/"
            if not os.path.exists(scwrl_temp_dir):
                os.makedirs(scwrl_temp_dir)
            mp_args = [(pdb_id_mod, traj[frame], frame, scwrl_temp_dir) for frame in xrange(n_frames)]
            mp_results = p.imap(mp_scwrl, mp_args)

            for frame in xrange(n_frames):
                mp_results.next()
                bar.next()
            bar.finish()
            p.close()
            p.join()

            del mp_args
            del mp_results
            del p
            gc.collect()
            print "SCWRLing completed in {:.1f} s".format(timer()-start_time)

            # Join SCWRL structs into one trajectory
            start_time = timer()
            bar = Bar("Concatenating...", max=n_frames)
            traj = md.load(scwrl_temp_dir + pdb_id_mod + "_0.pdb")
            bar.next()
            n_mismatch = 0
            for frame in xrange(1, n_frames):
                try:
                    traj = traj.join(md.load(scwrl_temp_dir + pdb_id_mod + "_{}.pdb".format(frame)))
                except ValueError:
                    traj = traj.join(traj[-1])
                    n_mismatch += 1
                bar.next()
            bar.finish()
            n_frames = traj.n_frames

            traj[0].save(traj_fn+".pdb")
            traj.save(traj_fn+".xtc")

            print "{} structs w/mismatched topologies".format(n_mismatch)
            print "Completed in {:.1f} s".format(timer()-start_time)

            # Delete individual frame SCWRL structs
            shutil.rmtree(scwrl_temp_dir)

        else:
            traj = md.load(traj_fn+".xtc", top=traj_fn+".pdb")
            n_frames = traj.n_frames

        # Recalculate energies of each frame separately, to get component energies
        start_time = timer()
        bar = Bar("Energy calc...", max=n_frames)

        p = multiprocessing.Pool(pool_size)
        mp_args = [(config_fns[rep], frame) for frame in xrange(n_frames)]

        pe_arr = np.zeros(n_frames)
        rot_arr = np.zeros(n_frames)
        env_arr = np.zeros(n_frames)
        mp_results = p.imap(mp_recalc_pe, mp_args)
        for frame in xrange(n_frames):
            pe, rot, env = mp_results.next()
            pe_arr[frame] = pe
            rot_arr[frame] = rot
            env_arr[frame] = env
            bar.next()
        bar.finish()
        p.close()
        p.join()

        del mp_args
        del mp_results
        del p
        gc.collect()
        print "calc completed in {:.1f} s".format(timer()-start_time)

        print "Starting energies (total, inter_rot, inter_env) [kT]:"
        print pe_arr[0], rot_arr[0], env_arr[0]
        print "Average energies second half of trajectory:"
        print "{:.2f} +/- {:.2f}, {:.2f} +/- {:.2f}, {:.2f} +/- {:.2f}".format((pe_arr[n_frames/2:]).mean(), (pe_arr[n_frames/2:]).std(),
                                                                               (rot_arr[n_frames/2:]).mean(), (rot_arr[n_frames/2:]).std(),
                                                                               (env_arr[n_frames/2:]).mean(), (env_arr[n_frames/2:]).std())

        # # Delete Upside trajectory
        # os.remove(config_fns[rep])

        # IRMSD
        print "IRMSD..."
        r_pdb_b = md.load(rl_pdb_dir + pdb_id + "_r_b.pdb")
        l_pdb_b = md.load(rl_pdb_dir + pdb_id + "_l_b.pdb")

        irmsd = irmsd_calc(r_pdb_b, l_pdb_b, traj, rl_chains)
        irmsd_all.append(irmsd)

        # f_nat
        print "f_nat..."
        contacts_n = mu.f_nat_computer(r_pdb_b, l_pdb_b).contacts_n
        n_contacts_n = len(contacts_n)

        ref, traj, r_pdb_match, l_pdb_match, r_decoy, l_decoy, res_r, res_l = rl_align_split(r_pdb_b, l_pdb_b, traj, rl_chains)
        f_nat_compy = mu.f_nat_computer_traj(ref, res_r, res_l)
        f_nat_compy.n_contacts_n = n_contacts_n # keep preslice number
        f_nat = f_nat_compy.compute_f_nat(traj)
        fnat_all.append(f_nat)

        # LRMSD
        print "LRMSD..."
        lrmsd = mu.bb_ligand_rmsd(l_pdb_match, l_decoy, all_frames=True)
        lrmsd_all.append(lrmsd)

        print "IRMSD: {} +/- {} Ang, LRMSD: {} +/- {}, f_nat: {} +/- {}".format(irmsd[n_frames/2:].mean(), irmsd[n_frames/2:].std(),
                                                                               lrmsd[n_frames/2:].mean(), lrmsd[n_frames/2:].std(),
                                                                               f_nat[n_frames/2:].mean(), f_nat[n_frames/2:].std())

    dat_dict = {"irmsd": irmsd_all, "lrmsd": lrmsd_all, "f_nat": fnat_all, "pe": pe_arr, "rot": rot_arr, "env": env_arr}

    dat_fn = os.path.join(output_dir, pdb_id_mod + ".nse_capri.pkl")
    with open(dat_fn, 'wb') as f:
        cp.dump(dat_dict, f)

print "Done."