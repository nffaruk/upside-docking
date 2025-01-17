#!/usr/bin/env python
import tables as tb
import numpy as np
import argparse
import sys

def find_multichain_terms(ids, chain_starts):
    assert len(ids.shape) == 2
    chain_starts = np.array(chain_starts, dtype='i')
    assert len(chain_starts.shape) == 1
    chain_num = (ids[:][:,:,None] >= chain_starts[None,None,:]).sum(axis=-1)
    multichain = np.array([len(set(x)) > 1 for x in chain_num])
    return multichain


def cut_out_rows(hdf_array, remove_row):
    assert remove_row.shape ==(hdf_array.shape[0],)
    assert remove_row.dtype == np.bool
    keep_row = np.logical_not(remove_row)
    n_keep   = keep_row.sum()

    hdf_array[:n_keep] = hdf_array[:][keep_row]
    hdf_array.truncate(n_keep)


def multicut(chain_starts, pot, grp_name, id_name, other_names, consider_row = None):
    if grp_name not in pot: return
    grp = pot._v_children[grp_name]

    multichain = find_multichain_terms(grp._v_children[id_name], chain_starts)
    if consider_row is not None: multichain = multichain & consider_row
    print 'Cutting %s (%i rows)' % (grp, multichain.sum())
    for nm in [id_name] + list(other_names):
        print '    %s' % nm
        cut_out_rows(grp._v_children[nm], multichain)
    print

def unsupported(pot, grp_name):
    if grp_name in pot: 
        print '!!!!!!!!   Error: %s is not supported   !!!!!!!!' % grp_name
        print

default_filter = tb.Filters(complib='zlib', complevel=5, fletcher32=True)
def create_array(t, grp, nm, obj=None):
    return t.create_earray(grp, nm, obj=obj, filters=default_filter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config to modify')
    parser_grp0 = parser.add_mutually_exclusive_group()
    parser_grp0.add_argument('--chain-first-residue', default=[0], type=int, action='append', help=
            'First residue index of a new chain.  May be specified multiple times.  --chain-first-residue=0 is assumed '+
            'and need not be specified')
    parser_grp0.add_argument('--chain-break-from-file', action='store_true', help='Use indices of chain first residues stored in config file.')
    parser.add_argument('--rl-chains', nargs=2, type=int, help=
            'Two (space separated) integers indicating the number of receptor and ligand chains, for attempting collective jumps')
    parser.add_argument('--jump-length-scale', type=float, default=5., help='Translational gaussian width in angstroms for Monte Carlo JumpSampler. Default: 5 angstroms')
    parser.add_argument('--jump-rotation-scale', type=float, default=30., help='Rotational gaussian width in degrees for Monte Carlo JumpSampler. Default: 30 degrees')
    parser.add_argument('--remove-pivot', action='store_true', help='Whether to remove the MC PivotSampler param group to isolate JumpSampler for testing')
    parser.add_argument('--no-jump', action='store_true', help='Whether to skip creating MC JumpSampler param group and prevent jumps')
    parser.add_argument('--ignore-weird-rama', action='store_true', help='Do not halt on detection of weird rama for slice')
    parser.add_argument('--tether-fragments', action='store_true', help='Whether to add a restraint between C and N of improper breaks')
    args = parser.parse_args()

    if args.rl_chains and not (args.chain_first_residue or chain_break_from_file):
        parser.error('must specify either --chain-first-residue or --chain-break-from-file to use --rl-chains')

    if args.tether_fragments and not args.chain_break_from_file:
        parser.error('--tether-fragments requires --chain-break-from-file')

    t = tb.open_file(args.config, 'a')
    if len(args.chain_first_residue) > 1:
        break_grp = t.create_group("/input","chain_break","Indicates that multi-chain simulation and removal of bonded potential terms accross chains requested")
        t.create_array(break_grp, "chain_first_residue", args.chain_first_residue[1:], "Contains array of chain first residues, apart from residue 0")

    if args.rl_chains:
        rl_chains = np.array(args.rl_chains, dtype='int32')
        t.create_array(t.root.input.chain_break, "rl_chains", rl_chains, "Numbers of receptor and ligand chains")

    if args.chain_break_from_file:
        try:
            args.chain_first_residue = np.append([0], t.root.input.chain_break.chain_first_residue)
        except tb.exceptions.NoSuchNodeError:
            print >>sys.stderr, 'WARNING: --chain-break-from-file requires chain first residues stored in config file. Halting breaking chains.'
            t.close()
            raise SystemExit(0)

    if not args.rl_chains:
        if 'rl_chains' in t.root.input.chain_break:
            rl_chains = t.root.input.chain_break.rl_chains[:]
        else:
            rl_chains = None

    print 'This program is an ugly hack, and your simulation may give very bad results.'
    print 'If you are lucky, the results will be only a little bad.'
    print
    print 'Breaking chain at residues %s' % args.chain_first_residue
    print

    pot = t.root.input.potential
    chain_starts = np.array(args.chain_first_residue)*3
    print chain_starts
    n_chains = len(chain_starts)

    # Setting Jump Sampler params
    if args.remove_pivot:
        t.remove_node("/input", "pivot_moves", recursive=True)

    if not args.no_jump:
        # Need to add one atom past the last atom so that the last chain is processed
        chain_starts_plus = np.append(chain_starts, len(t.root.input.sequence)*3)

        if rl_chains is None:
            # range covers each individual chain
            jump_atom_range = np.array([[chain_starts_plus[i], chain_starts_plus[i+1]] for i in xrange(n_chains)], dtype='int32')
        else:
            # Add ranges of all chains in receptor and ligand for collective jumps
            # if rl_chains[0] > 1:
                # jump_atom_range = np.append(jump_atom_range, [[chain_starts_plus[0], chain_starts_plus[rl_chains[0]]]], axis=0)
            jump_atom_range = np.array([[chain_starts_plus[0], chain_starts_plus[rl_chains[0]]]])
            # if rl_chains[1] > 1:
            jump_atom_range = np.append(jump_atom_range, [[chain_starts_plus[rl_chains[0]], chain_starts_plus[-1]]], axis=0)

        jump_sigma_trans = np.array([args.jump_length_scale]*len(jump_atom_range), dtype='float32')
        jump_sigma_rot = np.array([args.jump_rotation_scale*np.pi/180.]*len(jump_atom_range), dtype='float32') # Converts to radians

        print "jump atom_range:\n{}\nsigma_trans:\n{}\nsigma_rot:\n{}\n".format(jump_atom_range, jump_sigma_trans, jump_sigma_rot)

        jump_grp = t.create_group("/input","jump_moves","JumpSampler Params")
        t.create_array(jump_grp, "atom_range", jump_atom_range, "First, last atom num demarking each chain")
        t.create_array(jump_grp, "sigma_trans", jump_sigma_trans, "Translational gaussian width")
        t.create_array(jump_grp, "sigma_rot", jump_sigma_rot, "Rotational gaussian width")

    # FIXME don't attempt pivots across chain breaks

    # Breaking interactions
    multicut(chain_starts, pot, 'angle_spring', 'id', 'equil_dist spring_const'.split())
    multicut(chain_starts, pot, 'dihedral_spring', 'id', 'equil_dist spring_const'.split())
    multicut(chain_starts, pot, 'dist_spring', 'id', 'equil_dist spring_const bonded_atoms'.split(), pot.dist_spring.bonded_atoms[:].astype('bool'))

    if 'infer_H_O' in pot: 
        print 'Checking %s' % pot.infer_H_O
        print
        is_bad =(any(find_multichain_terms(pot.infer_H_O.donors   .id,chain_starts)) or
                 any(find_multichain_terms(pot.infer_H_O.acceptors.id,chain_starts)))

        if is_bad:
            print '!!! Error: You must use --hbond-excluded-residues in upside_config to produce chain breaks for the hbonds !!!'
            print

    unsupported(pot, 'basin_correlation_pot')

    if 'rama_coord' in pot:
        tbl = pot.rama_coord.id
        multichain_locs = find_multichain_terms(tbl[:], chain_starts).nonzero()[0]
        print 'Editing %s (%i rows)' %(pot.rama_coord, len(multichain_locs))
        for loc in multichain_locs:
            ids = tbl[loc]
            assert ids.shape == (5,)
            chain_num = (ids[:,None]>=chain_starts).sum(axis=-1)
            if not (chain_num[1] == chain_num[2] == chain_num[3] and (chain_num[0]==chain_num[1] or chain_num[3]==chain_num[4])):
                if not args.ignore_weird_rama: 
                    raise ValueError("Weird rama_coord %i, unable to proceed" % loc)
                else:
                    print "WARNING: ignoring weird rama_coord. Not understood effects of doing this."

            if chain_num[0]==chain_num[1]: 
                tbl[loc,4] = -1  # cut psi
            else: 
                tbl[loc,0] = -1  # cut phi

    if args.tether_fragments:
        ch_first_res = t.root.input.chain_break.chain_first_residue[:]
        chain_counts = t.root.input.chain_break.chain_counts[:] 

        imp_break_pairs = []
        ch_imp = 0
        for ch_real, memb_count in enumerate(chain_counts):
            if memb_count > 1:
                for ch in xrange(memb_count-1):
                    imp_idx = ch_imp + ch
                    print imp_idx
                    res_after = ch_first_res[imp_idx]
                    atom_after = res_after*3
                    atom_before = atom_after - 1
                    imp_break_pairs.append([atom_before, atom_after])
            ch_imp += memb_count
        imp_break_pairs = np.array(imp_break_pairs)
        n_restraints = len(imp_break_pairs)
        if n_restraints > 0:
            print "Adding springs between improper break atoms:", imp_break_pairs

            xyz = t.root.input.pos[:,:,0]
            pair_dists = []
            for pair in imp_break_pairs:
                pair_dists.append(xyz[pair[1]] - xyz[pair[0]])
            pair_dists = np.linalg.norm(pair_dists, axis=1)

            pair_spring_consts = np.full(n_restraints, 48.)

            grp = t.root.input.potential.dist_spring

            idx = grp.id[:]
            equil_dist = grp.equil_dist[:]
            spring_const = grp.spring_const[:]
            bonded_atoms = grp.bonded_atoms[:]

            grp.id._f_remove()
            grp.equil_dist._f_remove()
            grp.spring_const._f_remove()
            grp.bonded_atoms._f_remove()

            create_array(t, grp, 'id',           obj=np.concatenate((idx,          imp_break_pairs), axis=0))
            create_array(t, grp, 'equil_dist',   obj=np.concatenate((equil_dist,   pair_dists), axis=0))
            create_array(t, grp, 'spring_const', obj=np.concatenate((spring_const, pair_spring_consts), axis=0))
            create_array(t, grp, 'bonded_atoms', obj=np.concatenate((bonded_atoms, np.zeros(n_restraints, dtype='int')), axis=0))

    t.close()


if __name__ == '__main__':
    main()
