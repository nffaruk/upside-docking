''' Very opinionated convenience script for running upside jobs on midway '''
import collections
import numpy as np
import subprocess as sp
import os, sys
import json,uuid
import IPython.display as disp

params_dir = os.path.expanduser('~/upside-parameters/')
upside_dir = os.path.expanduser('~/upside/')


def upside_config(fasta, output, n_system, dimer=False, backbone=True, rotamer=True, 
                  sidechain=False, hbond=None, sheet_mix_energy=None, environment=None,
                  sidechain_scale=None, inverse_scale=0., inverse_radius_scale=None, init=None, rama_pot=True,
                  reference_rama=None, restraint_groups=[], restraint_spring=None, hbond_coverage_radius=None,
                  rotamer_interaction_param='/home/jumper/optimized_param4_env.h5'):
    
    args = [upside_dir + 'src/upside_config.py', '--fasta=%s'%fasta, '--output=%s'%output, '--n-system=%i'%n_system]

    if init:
        args.append('--initial-structures=%s'%init)
    if rama_pot:
        args.append('--rama-library=%s'%(params_dir+'rama_libraries.h5'))
    if sheet_mix_energy is not None:
        args.append('--rama-sheet-mixing-energy=%f'%sheet_mix_energy)
    if dimer:
        args.append('--dimer-basin-library=%s'%(params_dir+'TCB_count_matrices.pkl'))
    if not backbone:
        args.append('--no-backbone')
    if hbond:
        args.append('--hbond-energy=%f'%hbond)
    if reference_rama:
        args.append('--reference-state-rama=%s'%reference_rama)
    for rg in restraint_groups:
        args.append('--restraint-group=%s'%rg)
    if restraint_spring is not None:
        args.append('--restraint-spring-constant=%f'%restraint_spring)
        
    if rotamer or environment:
        args.append('--rotamer-placement=%s'%(params_dir+'rotamer-MJ-1996-with-direc.h5'))

    if rotamer:
        args.append('--rotamer-interaction=%s'%rotamer_interaction_param)
    if environment:
        args.append('--environment=%s'%environment)
    
    if sidechain:
        args.append('--sidechain-radial=%s'%(params_dir+'radial-MJ-1996.h5'))
    if sidechain_scale is not None: 
        args.append('--sidechain-radial-scale-energy=%f'%sidechain_scale)
    if inverse_scale: 
        args.append('--sidechain-radial-scale-inverse-energy=%f'%inverse_scale)
        args.append('--sidechain-radial-scale-inverse-radius=%f'%inverse_radius_scale)
        
    # print ' '.join(args)

    return ' '.join(args) + '\n' + sp.check_output(args)


def compile():
    return sp.check_output(['/bin/bash', '-c', 'cd %s; make -j4'%(upside_dir+'obj')])


UpsideJob = collections.namedtuple('UpsideJob', 'job config output'.split())


def run_upside(queue, config, duration, frame_interval, n_threads=1, hours=36, temperature=1., seed=None,
               replica_interval=None, max_temp=None, pivot_interval=None, time_step = None, 
               log_level='basic'):
    if isinstance(config,str): config = [config]
    
    upside_args = [upside_dir+'obj/upside', '--duration', '%f'%duration, 
            '--frame-interval', '%f'%frame_interval, '--temperature', '%f'%temperature] + config
    if replica_interval is not None:
        upside_args.extend(['--replica-interval', '%f'%replica_interval])
    if max_temp is not None:
        upside_args.extend(['--max-temperature', '%f'%max_temp])
    if pivot_interval is not None:
        upside_args.extend(['--pivot-interval', '%f'%pivot_interval])
    upside_args.extend(['--log-level', log_level])
    
    if time_step is not None:
        upside_args.extend(['--time-step', str(time_step)])

    upside_args.extend(['--seed','%li'%(seed if seed is not None else np.random.randint(1<<31))])
    
    output_path = config[0]+'.output'

    if queue == '': 
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(n_threads)
        output_file = open(output_path,'w')
        job = sp.Popen(upside_args, stdout=output_file, stderr=output_file)
    elif queue == 'srun':
        # set num threads carefully so that we don't overwrite the rest of the environment
        # setting --export on srun will blow away the rest of the environment
        # afterward, we will undo the change

        old_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)

        try:
            os.environ['OMP_NUM_THREADS'] = str(n_threads)
            args = ['srun', '--ntasks=1', '--nodes=1', '--cpus-per-task=%i'%n_threads, 
                    '--slurmd-debug=0', '--output=%s'%output_path] + upside_args
            job = sp.Popen(args, close_fds=True)
        finally:
            if old_omp_num_threads is None:
                del os.environ['OMP_NUM_THREADS']
            else:
                os.environ['OMP_NUM_THREADS'] = old_omp_num_threads
    else:
        args = ['sbatch', '-p', queue, '--time=0-%i'%hours, '--ntasks=1', 
                '--cpus-per-task=%i'%n_threads, '--export=OMP_NUM_THREADS=%i'%n_threads,
                '--output=%s'%output_path, '--parsable', '--wrap', ' '.join(upside_args)]
        job = sp.check_output(args).strip()

    return UpsideJob(job,config,output_path)


def status(job):
    try:
        job_state = sp.check_output(['/usr/bin/env', 'squeue', '-j', job.job, '-h', '-o', '%t']).strip()
    except sp.CalledProcessError:
        job_state = 'FN'
        
    if job_state == 'PD':
        status = ''
    else:
        status = sp.check_output(['/usr/bin/env','tail','-n','%i'%1, job.output])[:-1]
    return '%s %s' % (job_state, status)


def read_hb(tr):
    n_res = tr.root.input.pos.shape[0]/3
    don_res =  tr.root.input.potential.infer_H_O.donors.id[:,1] / 3
    acc_res = (tr.root.input.potential.infer_H_O.acceptors.id[:,1]-2) / 3
    
    n_hb = tr.root.output.hbond.shape[2]
    hb_raw   = tr.root.output.hbond[:,0]
    hb = np.zeros((hb_raw.shape[0],n_res,2,2))

    hb[:,don_res,0,0] =    hb_raw[:,:len(don_res)]
    hb[:,don_res,0,1] = 1.-hb_raw[:,:len(don_res)]

    hb[:,acc_res,1,0] =    hb_raw[:,len(don_res):]
    hb[:,acc_res,1,1] = 1.-hb_raw[:,len(don_res):]
    
    return hb

def read_constant_hb(tr, n_res):
    don_res = tr.root.input.potential.infer_H_O.donors.residue[:]
    acc_res = tr.root.input.potential.infer_H_O.acceptors.residue[:]
    
    n_hb = tr.root.output.hbond.shape[2]
    hb_raw   = tr.root.output.hbond[:,0]

    hb = np.zeros((hb_raw.shape[0],n_res,2,3))

    hb[:,don_res,0,0] = hb_raw[:,:len(don_res),0]
    hb[:,don_res,0,1] = hb_raw[:,:len(don_res),1]
    hb[:,don_res,0,2] = 1.-hb_raw[:,:len(don_res)].sum(axis=-1)

    hb[:,acc_res,1,0] = hb_raw[:,len(don_res):,0]
    hb[:,acc_res,1,1] = hb_raw[:,len(don_res):,1]
    hb[:,acc_res,1,2] = 1.-hb_raw[:,len(don_res):].sum(axis=-1)
    
    return hb
    


def rmsd_transform(target, model):
    assert target.shape == model.shape == (model.shape[0],3)
    base_shift_target = target.mean(axis=0)
    base_shift_model  = model .mean(axis=0)
    
    target = target - target.mean(axis=0)
    model = model   - model .mean(axis=0)

    R = np.dot(target.T, model)
    U,S,Vt = np.linalg.svd(R)
    if np.linalg.det(np.dot(U,Vt))<0.:
        Vt[:,-1] *= -1.  # fix improper rotation
    rot = np.dot(U,Vt)
    shift = base_shift_target - np.dot(rot, base_shift_model)
    return rot, shift


def structure_rmsd(a,b):
    rot,trans = rmsd_transform(a,b)
    diff = a - (trans+np.dot(b,rot.T))
    return np.sqrt((diff**2).sum(axis=-1).mean(axis=-1))


def traj_rmsd(traj, native):
    return np.array([structure_rmsd(x,native) for x in traj])


def vmag(x):
    assert x.shape[-1] == 3
    return np.sqrt(x[...,0]**2 + x[...,1]**2 + x[...,2]**2)


def vhat(x):
    return x / vmag(x)[...,None]


def compact_sigmoid(x, sharpness):
    y = x*sharpness;
    result = 0.25 * (y+2) * (y-1)**2
    result = np.where((y< 1), result, np.zeros_like(result))
    result = np.where((y>-1), result, np.ones_like (result))
    return result


def compute_topology(t):
    seq = t.root.input.sequence[:]
    infer = t.root.input.potential.infer_H_O
    n_donor = infer.donors.id.shape[0]
    n_acceptor = infer.acceptors.id.shape[0]
    id = np.concatenate((infer.donors.id[:],infer.acceptors.id[:]), axis=0)
    bond_length = np.concatenate((infer.donors.bond_length[:],infer.acceptors.bond_length[:]),axis=0)
    
    def augment_pos(pos, id=id, bond_length=bond_length):
        prev = pos[id[:,0]]
        curr = pos[id[:,1]]
        nxt  = pos[id[:,2]]
        
        virtual = curr + bond_length[:,None] * vhat(vhat(curr-nxt) + vhat(curr-prev))
        new_pos = np.concatenate((pos,virtual), axis=0)
        return json.dumps([map(float,x) for x in new_pos])  # convert to json form
    
    n_atom = 3*len(seq)
    backbone_names = ['N','CA','C']
    
    backbone_atoms = [dict(name=backbone_names[i%3], residue_num=i/3, element=backbone_names[i%3][:1]) 
                      for i in range(n_atom)]
    virtual_atoms  = [dict(name=('H' if i<n_donor else 'O'), residue_num=int(id[i,1]/3), 
                           element=('H' if i<n_donor else 'O'))
                     for i in range(n_donor+n_acceptor)]
    backbone_bonds = [[i,i+1] for i in range(n_atom-1)]
    virtual_bonds  = [[int(id[i,1]), n_atom+i] for i in range(n_donor+n_acceptor)]
    
    topology = json.dumps(dict(
        residues = [dict(resname=str(s), resid=i) for i,s in enumerate(seq)],
        atoms = backbone_atoms + virtual_atoms,
        bonds = backbone_bonds + virtual_bonds,
    ))
    
    return topology, augment_pos


def display_structure(topo_aug, pos, size=(600,600)):
    id_string = uuid.uuid4()
    return disp.Javascript(lib='/files/js/protein-viewer.js', 
                    data='render_structure(element, "%s", %i, %i, %s, %s);'%
                       (id_string, size[0], size[1], topo_aug[0], topo_aug[1](pos))), id_string