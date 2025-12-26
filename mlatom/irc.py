from . import data, constants
import sys, os, shutil, tempfile
import numpy as np 
from datetime import datetime
from typing import List, Optional

TASK_ROOT = "irc"; TRAJ_ROOT = "traj"; PLOT_ROOT = "plot"; CALC_ROOT = "calc"
DUMP_FILENAME = "irc_traj"; DUMP_FORMAT = ["json", "xyz"]; PLOT_FILENAME = "irc_plot"

# ==============================================================
# The main IRC class used by MLatom
# ==============================================================

class irc():

    '''
    Provide the intrinsic reaction coordinates (IRC) given transition state. IRC is the mimial energy path under the mass weighted coordinates. Path direction of each point on IRC follows the negative energy gradients (by definition).

    Arguments:
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): 
            Model or method to provide energy and energy derivatives for IRC generation.
        molecule (:class:`data.molecule`): 
            The transtion state molecule to start with.
        model_predict_kwargs (dict, optional): 
            Additional keywords used for prediction.
        program (str, optional): 
            The program to be used for generating IRC. Available options are ``geometric``, ``gaussian``, ``pysisyphus`` and ``mlatom``. Default: ``None``, i.e., the built-in implementation in mlatom
        program_kwargs (dict, optional): 
            Control the behavior of the program. Default: ``None``
        forward (bool, optional): 
            Whether to generate forward trajectory. Default: ``None``
        backward (bool, optional): 
            Whether to generate backward trajectory. Default: ``None``
        working_directory (str, optional): 
            The working directory for IRC calculation. Default: current directory 
        overwrite (bool, optional):
            Whether to overwrite the $task_directory if it exits. Default: ``True``
        plot (bool, optional): 
            Whether to plot the energy changes with respect to the reaction coordinates. Default: ``True``
        plot_filename (str, optional): 
            The file name to save the plot. Default: ``irc_plot`` (it will dump ``irc_plot.png`` and ``irc_plot.txt`` with the raw data for this plot)
        dump (bool, optional): 
            Whether to dump the obtained IRC trajectory file. Default:``True``
        dump_filename (str, optional): 
            The file name without extension to save the IRC trajectory. Default: ``irc_traj`` (the extension will be added based on the format, see below)
        dump_format (List[str], str, optional): 
            The format of the file to save the IRC trajectory. Default: ``['json', 'xyz]``, i.e., the molecular database in .json format and .xyz
        verbose (bool, optional): 
            Whether to print information. Default: ``True``


    **Example:**

    .. code-block:: python

        mol = ml.molecule.from_xyz_file('init_ts.xyz')
        method = ml.models.methods(method='AIQM2')

        # optimize TS
        geomopt = ml.optimize_geometry(model=method, initial_molecule=mol, ts=True)
        optmol = geomopt.optimized_molecule
        _ = ml.freq(model=method, molecule=optmol, program='pyscf')

        # run IRC
        irc_results = ml.irc(
            model=method,
            molecule=optmol)
        
        print(irc_results)
        '''
    
    def __init__(
           self,
           model = None, 
           molecule:data.molecule = None, 
           model_predict_kwargs:dict = None,
           trajdb: data.molecular_database = None,
           ftrajdb: data.molecular_database = None,
           btrajdb: data.molecular_database = None, 
           program:Optional[str] = None, 
           program_kwargs:Optional[dict] = None,
           forward:Optional[bool] = None, 
           backward:Optional[bool] = None, 
           working_directory:Optional[str] = None, 
           overwrite:Optional[bool] = None,
           plot:Optional[bool] = None, 
           plot_filename:Optional[str] = None,
           dump:Optional[bool] = None, 
           dump_filename:Optional[str] = None, 
           dump_format:Optional[List[str]] = None,
           verbose:Optional[bool] = None):

        # --- check whether irc traj exists --- #
        self.trajdb = trajdb; self.ftrajdb = ftrajdb; self.btrajdb = btrajdb
        if self.trajdb is not None:
            return 

        # --- Check settings --- #
        assert model is not None, "Please provide model or method for IRC generation"
        assert molecule is not None, "Please provide transition state molecule for IRC generation"
        
        if verbose: highlight_print(f"Parse settings for IRC calculation")
        # working directory
        if working_directory is None: 
            working_directory = os.path.abspath(os.path.join(os.getcwd(), TASK_ROOT))
        else: working_directory = os.path.abspath(working_directory)
        if overwrite is None: overwrite = True
        if os.path.exists(working_directory):
            if overwrite: 
                print(f"IRC working directory {working_directory} found and overwrite was set to True. The working directory will be recreated."); sys.stdout.flush()
                shutil.rmtree(working_directory)
            else:
                print(f"IRC working directory {working_directory} found but overwrite was set to False. Please choose another working directory to save IRC results."); sys.stdout.flush()
                return 
            
        os.makedirs(working_directory)
        model.working_directory = os.path.join(working_directory, CALC_ROOT)
        
        if program is None: program = 'mlatom'
        if program_kwargs is None: program_kwargs = {}
        if model_predict_kwargs is None: model_predict_kwargs = {}
        if plot is None: plot = True
        if dump is None: dump = True
        if verbose is None: verbose = True

        if plot:
            if plot_filename is None: plot_filename = PLOT_FILENAME
        else: plot_filename = None
        if dump:
            if dump_filename is None: dump_filename = DUMP_FILENAME
            if dump_format is None: dump_format = DUMP_FORMAT
        else: dump_filename = None; dump_format = None

        # direction
        if not forward and not backward: forward = True; backward = True
        if program.lower() == 'geometric': 
            if not forward or not backward:
                print("geomeTRIC can not specify direction. Both Directions will be propagated."); sys.stdout.flush()
            forward = True; backward = True

        if verbose:        
            settings_print("Calculation", locals())
            settings_print("Program", program_kwargs)
            sys.stdout.flush()


        # --- choose program and initialize --- #
        if program.lower() == 'gaussian': 
            irc_generator = irc_gaussian; x_unit = None
        elif program.lower() == 'geometric': 
            irc_generator = irc_geometric; x_unit = "Index"
        elif program.lower() == 'mlatom': 
            irc_generator = irc_mlatom; x_unit = None
        elif program.lower() == 'pysisyphus': 
            irc_generator = irc_pysisyphus; x_unit = None
        else:
            raise ValueError('Please provide correct program name for generating IRC. The supported programs are gaussian, geometric, pysisiphus and mlatom built-in implementation if left None.')
        

        # --- check frequency inside molecule --- #
        if verbose: highlight_print("Start checking frequencies inside molecule object")
        if program.lower() == 'gaussian':
            if verbose:
                print("\nGaussian will perform hessian calculations by default. Skip molecule checking\n")
        else:
            has_freq = check_molecule(molecule) 
            if not has_freq:
                print(f"Start frequency calculation"); sys.stdout.flush()
                from .simulations import freq
                freq(model=model, molecule=molecule, model_predict_kwargs=model_predict_kwargs)
            else:
                if verbose:
                    print(f"\nFrequencies found and validated as transition state!")
                    print(f"The found imaginary frequency: {molecule.frequencies[0]} cm-1")
                
                sys.stdout.flush()
            
            # check if TS
            check_molecule(molecule)
        if verbose: highlight_print("Finish checking frequencies inside molecule")
                

        # --- generate IRC --- #
        if verbose: highlight_print(f"Start IRC calculation with {program.lower()}")

        if forward and backward: direction = 'both'
        elif forward: direction = 'forward'
        elif backward: direction = 'backward'

        trajdb = irc_generator(
                model=model, molecule=molecule, model_predict_kwargs=model_predict_kwargs, 
                working_directory=working_directory, 
                direction=direction, program_kwargs=program_kwargs, verbose=verbose)
        self.trajdb = trajdb
        if self.ts_index != 0:
            if self.ftrajdb is None or self.btrajdb is None: 
                self.split()
        self.add_comment(direction)
        
        if verbose: highlight_print(f"Finish IRC calculation with {program.lower()}")

        # --- generate figure and dump trajectory --- #
        if plot: 
            self.plot(
                working_directory=os.path.join(working_directory, PLOT_ROOT), 
                filename=plot_filename, 
                x_unit=x_unit, verbose=verbose)
        if dump: 
            self.dump(os.path.join(working_directory, TRAJ_ROOT), dump_filename, dump_format, verbose)
    
    def __str__(self):
        forward = False; backward = False
        printstrs = [highlight_print(text="Results from IRC calculation", return_string=True)]
        if self.ftrajdb is not None: forward = True
        if self.btrajdb is not None: backward = True

        if forward:
            printstrs += ["  IRC in forward direction "]
            printstrs += [f"    Number of steps {len(self.ftrajdb)}"]
            dE = (self.ftrajdb[0].energy - self.ftrajdb[-1].energy) * constants.Hartree2kcalpermol
            printstrs += [f"    Energy decrease {dE:.4} kcal/mol"]
            printstrs += [f"    Structure of final molecule:"]
            printstrs += [self.ftrajdb[-1].info(properties='xyz_coordinates',return_string=True)]
        if backward:
            printstrs += ["  IRC in backward direction "]
            printstrs += [f"    Number of steps {len(self.btrajdb)}"]
            dE = (self.btrajdb[0].energy - self.btrajdb[-1].energy) * constants.Hartree2kcalpermol
            printstrs += [f"    Energy decrease {dE:.4} kcal/mol"]
            printstrs += [f"    Structure of final molecule:"]
            printstrs += [self.btrajdb[-1].info(properties='xyz_coordinates',return_string=True)]
        
        printstrs += [f"  Index of transition state: {self.ts_index} (starting from zero)"]
        printstr = '\n'.join(printstrs)
        return printstr

    @classmethod
    def irc_filename(cls, working_directory=None, prefix=None, direction=None, ext=None):
        if working_directory is None: 
            working_directory = os.path.join(os.getcwd(), TASK_ROOT)
        else: working_directory = os.path.abspath(working_directory)
        if prefix is None: prefix = DUMP_FILENAME
        if direction is None: direction = ""
        if ext is None: ext = "json"
        return os.path.join(working_directory, prefix+direction+'.'+ext)
    
    @property
    def ts_index(self):
        if "_ts_index" not in self.__dict__:
            assert self.trajdb is not None, "IRC trajectory database not found!"
            rcoords = self.trajdb.get_properties("reaction_coordinates")
            return np.argwhere(rcoords==0).flatten()[0].item()
        else: return self._ts_index 

    @ts_index.setter
    def ts_index(self, value):
        if value is not None: self._ts_index = value

    @property 
    def rcoords(self):
        assert self.trajdb is not None, "IRC trajectory database not found!"
        return self.trajdb.get_properties("reaction_coordinates")
    
    @property
    def ts_molecule(self):
        assert self.trajdb is not None, "IRC trajectory database not found!"
        return self.trajdb[self.ts_index]

    @property
    def traj(self):
        if self.trajdb is None: return None
        return self.trajdb.to_molecular_trajectory()
    
    @traj.setter
    def traj(self, value):
        if value is not None:
            self.trajdb = value.to_database()

    @property
    def ftraj(self):
        if self.ftrajdb is None: return None
        return self.ftrajdb.to_molecular_trajectory()
    
    @ftraj.setter
    def ftraj(self, value):
        if value is not None:
            self.ftrajdb = value.to_database()
    
    @property
    def btraj(self):
        if self.btrajdb is None: return None
        return self.btrajdb.to_molecular_trajectory()
    
    @btraj.setter
    def btraj(self, value):
        if value is not None:
            self.btrajdb = value.to_database()

    @classmethod
    def load(cls, working_directory=None, dump_filename=None):
        '''load IRC results from the traj folder'''
        
        if os.path.exists(os.path.join(working_directory, TRAJ_ROOT)):
            working_directory = os.path.join(working_directory, TRAJ_ROOT)
        trajdb_json = cls.irc_filename(working_directory, dump_filename, None, 'json')
        trajdb_xyz = cls.irc_filename(working_directory, dump_filename, None, 'xyz')

        if os.path.exists(trajdb_json):
            trajdb = data.molecular_database.load(trajdb_json, format='json')
            inst = cls(trajdb=trajdb)
        elif os.path.exists(trajdb_xyz):
            trajdb = data.molecular_database.from_xyz_file(trajdb_xyz, format='json')
            inst = cls(trajdb=trajdb)
            inst.read_comment()
        else:
            raise ValueError(f"IRC results not found in {trajdb_json} or {trajdb_xyz}")

        ts_index = inst.ts_index
        if ts_index != 0: inst.split()

        return inst

    def add_comment(self, direction):
        """add information of IRC to comment"""
        for mol in self.trajdb:
            rcoord = mol.reaction_coordinates
            rcoord_str = f"{mol.reaction_coordinates:.6}"
            energy = f"{mol.energy:.10}"
            if rcoord != 0:
                if direction.lower() in ['forward', 'backward']:
                    mol.comment = " ".join([direction, rcoord_str, energy])
                elif rcoord > 0:
                    mol.comment = " ".join(['forward', rcoord_str, energy])
                else: mol.comment = " ".join(['backward', rcoord_str, energy])
            else: mol.comment = " ".join(["TS", rcoord_str, energy])

    def read_comment(self):
        """read comment to add information to IRC traj"""
        comments = self.trajdb.get_properties('comment')
        rcoords = []; energies = []
        for com in comments:
            direction, rcoord, energy = com.split(" ")
            rcoords.append(float(rcoord))
            energies.append(float(energy))
        self.trajdb.add_scalar_properties(rcoord, "reaction_coordinates")
        self.trajdb.add_scalar_properties(energies, "energy")

    @classmethod
    def combine(cls, ftrajdb, btrajdb):
        """Combine forward results and backward results"""

        btrajdb.molecules = btrajdb.molecules[1:][::-1]
        brcoords = btrajdb.get_properties('reaction_coordinates')
        btrajdb.add_scalar_properties(-brcoords, "reaction_coordinates")

        trajdb = data.molecular_database(
            molecules=btrajdb.molecules + ftrajdb.molecules)
        return trajdb 
    
    def split(self):
        """Split the trajdb into ftrajdb and btrajdb according to reaction coord"""
        # if self.ftrajdb is not None: print("Forward trajectory database found and will be overritten")
        # if self.btrajdb is not None: print("Backward trajectory database found and will be overritten")

        ftrajdb = self.trajdb[self.ts_index:]
        btrajdb = self.trajdb[:self.ts_index+1][::-1].copy()
        btrajdb.add_scalar_properties(
            -btrajdb.get_properties("reaction_coordinates"), "reaction_coordinates")
        self.ftrajdb = ftrajdb 
        self.btrajdb = btrajdb
    
    def plot(self, working_directory=None, filename=None, x_unit=None, verbose=True):
        import matplotlib.pyplot as plt 

        if working_directory is None: working_directory = os.getcwd()
        if filename is None: filename = PLOT_FILENAME
        plot_filename = os.path.join(working_directory, filename+'.png')
        data_filename = os.path.join(working_directory, filename+'.txt')
        if not os.path.exists(working_directory):
            os.makedirs(working_directory, exist_ok=True)
        if x_unit is None: x_unit = r'$\sqrt{amu}*Å$'

        energies = self.trajdb.get_properties('energy')
        reaction_coordinates = self.trajdb.get_properties('reaction_coordinates')

        # dump data as .txt file
        with open(data_filename, 'w') as f:
            for ee, rc in zip(energies, reaction_coordinates):
                f.write(f'{ee:.8}\t{rc:.8}\n')
        
        # plot
        ts_energy = self.trajdb[self.ts_index].energy
        energies -= ts_energy
        energies = energies*data.constants.Hartree2kcalpermol
        if verbose:
            print(f"Path to plot: \n{plot_filename}\n{data_filename}"); sys.stdout.flush()
        
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(reaction_coordinates,energies,marker='.')
        
        # add text above each point
        # hhover = (energies[1]-energies[0])/4
        # for rc, ee in zip(reaction_coordinates, energies):
        #     plt.text(rc, ee+hhover, f"{ee:.2f}", ha='center')
        ax.set_xlabel(f'Intrinsic reaction coordinate / '+ x_unit)
        ax.set_ylabel('Total energy relative to TS (kcal/mol)')
        ax.set_title('Total energy along IRC (' + r'$E_\text{TS}$' + f'={ts_energy:.6} Ha)')
        # plt.ticklabel_format(axis='y', useOffset=False)
        # plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        # plt.grid()
        ax.grid(True)
        fig.savefig(plot_filename, dpi=150)

    def dump(self, working_directory=None, filename=None, format=None, verbose=True):

        paths = []
        if working_directory is None: working_directory = os.path.abspath(os.getcwd())
        if filename is None: filename = DUMP_FILENAME
        if format is None: format = DUMP_FORMAT
        filename = os.path.join(working_directory, filename)
        if not os.path.exists(working_directory):
            os.makedirs(working_directory, exist_ok=True)

        if isinstance(format, str): format = [format]

        for ff in format:
            if ff == 'json':
                self.trajdb.dump(filename+'.json', format='json')
                paths.append(filename+'.json')
                if self.ftrajdb is not None: 
                    self.ftrajdb.dump(filename+'_forward.json', format='json')
                    paths.append(filename+'_forward.json')
                if self.btrajdb is not None: 
                    self.btrajdb.dump(filename+'_backward.json', format='json')
                    paths.append(filename+'_backward.json')
            elif ff == 'xyz':
                self.trajdb.write_file_with_xyz_coordinates(filename+'.xyz')
                paths.append(filename+'.json')
                if self.ftrajdb is not None: 
                    self.ftrajdb.write_file_with_xyz_coordinates(filename+'_forward.xyz')
                    paths.append(filename+'_forward.xyz')
                if self.btrajdb is not None: 
                    self.btrajdb.write_file_with_xyz_coordinates(filename+'_backward.xyz')
                    paths.append(filename+'_backward.xyz')
            else:
                print(f"Format {ff} is not supported for dumping IRC trajectories. Skip this format")
                sys.stdout.flush()
        if verbose:
            paths_string = '\n'.join(paths)
            print(f"The saved traj files:")
            print(paths_string)
            sys.stdout.flush()


# ==============================================================
# Supported programs for IRC
# ==============================================================

def irc_geometric(**kwargs):
    from .interfaces import geometric_interface
    ftrajdb, btrajdb = geometric_interface.generate_irc(**kwargs)
    trajdb = irc.combine(ftrajdb, btrajdb)
    return trajdb

def irc_pysisyphus(**kwargs):
    from .interfaces import pysisyphus_interface 
    trajdb = pysisyphus_interface.generate_irc(**kwargs)
    return trajdb

def irc_gaussian(model, molecule, model_predict_kwargs, working_directory, direction, program_kwargs, verbose):
    from .interfaces import gaussian_interface
    if 'number' in molecule.__dict__.keys(): suffix = f'_{molecule.number}'
    else: suffix = ''
    filename = f'gaussian{suffix}'

    # gaussian.com and .log go to $working_directory/calcs
    working_directory = os.path.join(working_directory, CALC_ROOT)
    os.makedirs(working_directory, exist_ok=True)
    model.dump(filename=os.path.join(working_directory, 'model.json'), format='json')

    # make a copy of kwargs for safety purpose
    program_kwargs_copy = program_kwargs.copy()
    
    # IRC keywords
    irc_keywords = ['CalcFC']
    if direction.lower() == 'forward': irc_keywords.append("Forward")
    elif direction.lower() == 'backward': irc_keywords.append("Reverse")
    if 'irc_keywords' in program_kwargs: 
        for kk in program_kwargs['irc_keywords']:
            if kk not in irc_keywords: irc_keywords.append(kk)
    program_kwargs_copy['irc_keywords'] = irc_keywords

    # Run Gaussian
    gaussian_interface.run_gaussian_job(
        filename=f'{filename}.com', molecule=molecule, external_task='irc', 
        working_directory=working_directory, model_predict_kwargs=model_predict_kwargs,**program_kwargs_copy)

    # Get results
    moleculetmp = molecule.copy(atomic_labels=[], molecular_labels=[])
    gaussian_interface.parse_gaussian_output(filename=os.path.join(working_directory, f'{filename}.log'), molecule=moleculetmp)

    if "error_message" in moleculetmp.__dict__:
        raise ValueError(f"IRC with Gaussian terminated with the following error message:\n\n{moleculetmp.error_message}")
    
    # filter unconverged steps
    trajdb = moleculetmp.molecular_database.filter_by_property('n_path')
    trajdb[0].reaction_coordinates = 0.00 # TS

    rcoord = trajdb.get_properties('reaction_coordinates')
    trajdb.add_scalar_properties(rcoord * constants.Bohr2Angstrom, 'reaction_coordinates')

    if direction.lower() == 'both':
        ftrajdb = data.molecular_database(molecules=[trajdb[0]] + [mol for mol in trajdb[1:] if mol.n_path==1])
        btrajdb = data.molecular_database(molecules=[trajdb[0]] + [mol for mol in trajdb[1:] if mol.n_path==2])
        trajdb = irc.combine(ftrajdb, btrajdb)

    return trajdb

def irc_mlatom(model, molecule, model_predict_kwargs, working_directory, direction, program_kwargs, verbose):

    avail_integrator = ['lqa']; 
    avail_hess_est = ['bofill', 'sr1', 'psb', False]
    program_kwargs_copy = program_kwargs.copy()

    # --- parse settings --- #
    # IRC settings
    n_steps = program_kwargs_copy.pop('n_steps', 10)
    step_size = program_kwargs_copy.pop('step_size', 5e-2) # mass deweighted
    init_length = program_kwargs_copy.pop('init_length', 5e-2) # mass deweighted
    thred = program_kwargs_copy.pop('thred', 1e-5)

    # integrator
    algorithm = program_kwargs_copy.pop('algorithm', 'lqa')
    if algorithm.lower() not in avail_integrator:
        raise ValueError(f"{algorithm} is not available from built-in implementation of IRC in MLatom. Please choose from {avail_integrator}")
    if verbose: print(f'Type of Integrator: {algorithm}')

    # --- decide hessian estimator --- #
    hess_est = program_kwargs_copy.pop('hess_est', 'bofill')
    if hess_est:
        if hess_est.lower() not in avail_hess_est:
            raise ValueError(f"{hess_est} is not available from built-in implementation of IRC in MLatom. Please choose from {avail_hess_est}")
        if verbose: print(f'Type of Hessian estimator: {hess_est}')
    else:
        if verbose: print(f'Hessian estimator turned off. Hessian will be calculated at each step.')

    # --- initialize integrator and hessian estimator --- #
    if algorithm.lower() == 'lqa': integrator = integrator_lqa(**program_kwargs_copy)

    if hess_est.lower() == 'bofill': hess_est = bofill
    elif hess_est.lower() == 'sr1': hess_est = sr1 
    elif hess_est.lower() == 'psb': hess_est = psb

    # --- propagate IRC --- #
    def irc_gen(direction):
        
        if verbose:
            highlight_print(f"Start IRC in {direction} direction.")

        prev_mwhessian = to_mwhess(molecule.nuclear_masses, molecule.hessian)
        # get gradients of TS for later hessian estimation
        if 'energy_gradients' not in molecule.atoms[0].__dict__:
            model.predict(molecule=molecule, calculate_energy_gradients=True, **model_predict_kwargs)
        prev_mwgrad = to_mwgrad(molecule.nuclear_masses, molecule.get_energy_gradients())
        prev_mwcoord = to_mwcoord(molecule.nuclear_masses, molecule.xyz_coordinates)

        # store results
        trajdb = [molecule]
        mwrcoord = 0.00; mwrcoords = [mwrcoord]

        ##############
        # first step #
        ##############
        # here I directly use normal mode from frequency calculation and apply some fixed scaling. 
        # There are some other schemes and we can implement later.
        # need to make sure the mass weighted transition vector is normalized here
        if verbose:
            print(f"Generate initial displacement in {direction} direction."); sys.stdout.flush()
        assert 'normal_mode' not in molecule.atoms[0].__dict__, "Please perform frequency calculation before IRC"
        trans_vec = np.array([aa.normal_modes[0] for aa in molecule.atoms]) # normal modes are normalized from MLatom
        trans_vec_mw = to_mwcoord(molecule.nuclear_masses, trans_vec)
        trans_fac = np.linalg.norm(trans_vec_mw)/np.linalg.norm(trans_vec)
        trans_vec_mw = trans_vec_mw/np.linalg.norm(trans_vec_mw)

        step_size_mw = step_size * trans_fac # convert step size to mass weighted

        init_length_mw = trans_fac * init_length
        current_mwstep = init_length_mw * trans_vec_mw
        if direction.lower() == 'backward': current_mwstep = -current_mwstep

        current_mwcoord = prev_mwcoord + current_mwstep
        mwrcoord += init_length_mw
        current_molecule = upd_coords(molecule, current_mwcoord)

        trajdb.append(current_molecule)
        mwrcoords.append(init_length_mw)

        prev_mwstep = current_mwstep
        ##############
        # start IRC  #
        ##############
        for ii in range(0, n_steps):
            ptime = f"{datetime.now().strftime('%D %H:%M:%S')}"
            model.predict(
                molecule=current_molecule, calculate_energy=True, calculate_energy_gradients=True, **model_predict_kwargs)
            current_grad = current_molecule.energy_gradients
            current_mwgrad = to_mwgrad(molecule.nuclear_masses, current_grad)

            # only check modulus of the mass unweighted gradients
            if np.linalg.norm(current_grad) < thred:
                print('Gradient threshold reached and iteration stops.'); sys.stdout.flush()
                break

            # print summary
            # if ii > 0:
            dE = (trajdb[-1].energy - trajdb[-2].energy)*constants.Hartree2kcalpermol
            if verbose:
                print(f"{ptime} | Step {ii} in {direction} direction."); sys.stdout.flush()
                print(f"    Reaction coordinate {mwrcoords[-1]:.4} in sqrt(amu)*Å")
                print(f"    Energy changed {dE:.4} kcal/mol")
                sys.stdout.flush()

            # perform sp for the last step
            if ii == n_steps-1: break
            
            if not hess_est:
                # get gradients and true hessian
                model.predict(molecule=current_molecule, calculate_hessian=True, **model_predict_kwargs)
                current_hessian = current_molecule.hessian
                current_mwhessian = to_mwhess(molecule.nuclear_masses, current_hessian)
            else:
                # estimate hessian from old hessian and gradients from both current and previous steps
                delta_mwhessian = hess_est(prev_mwhessian, prev_mwgrad, current_mwgrad, prev_mwstep)
                current_mwhessian = prev_mwhessian + delta_mwhessian

            # reaction path integration
            current_mwstep = integrator(step_size=step_size_mw, grad=current_mwgrad, hessian=current_mwhessian)
            
            # update reaction coordinates
            mwrcoord += np.linalg.norm(current_mwstep).item(); mwrcoords.append(mwrcoord)
            prev_mwgrad = current_mwgrad; prev_mwhessian = current_mwhessian
            prev_mwstep = current_mwstep

            # update coordinates
            current_mwcoord = current_mwcoord + current_mwstep
            
            # update molecule
            current_molecule = upd_coords(molecule, current_mwcoord)
            trajdb.append(current_molecule)

        trajdb = data.molecular_database(molecules=trajdb)
        trajdb.add_scalar_properties(np.array(mwrcoords), "reaction_coordinates")
        return trajdb

    if direction == 'both':
        ftrajdb = irc_gen('forward'); btrajdb = irc_gen('backward')
        trajdb = irc.combine(ftrajdb, btrajdb)
    else:
        trajdb = irc_gen(direction)
    return trajdb
 

# ==============================================================
# Reaction path integrators
# ==============================================================
# Only LQA is implemented

class integrator_lqa():

    def __init__(self, n_euler=5000):
        self.n_euler = n_euler 
         
    def __call__(self, step_size, grad=None, hessian=None):
        '''
        Implementation of local quadratic approximation (LQA) to obtain IRC path. 
        DOI: 10.1063/1.459634

        dx(s)/ds = -(g_0 + H_0\Delta x)/|g_0 + H_0\Delta x|

        In Pysisyphus, they didn't integrate to the desired step size,
        but to the limit of numer of Euler steps. In my implementation, 
        I use step size to control euler integration instead of step numbers.
        '''

        eigvals, eigvec = np.linalg.eigh(hessian)
        n_eigval = len(eigvals)

        def euler_integral(step_size, grad, eigvals, eigvec):
            '''
            Get the independent parameter t by applying euler integral on arc length s
            '''
            dt = 1/self.n_euler * step_size / np.linalg.norm(grad)
            s = 0; t = dt
            while s < step_size: 
                s += dt * euler_integral_step(grad, eigvals, eigvec, t)
                t += dt
            return t, s

        def euler_integral_step(grad, eigvals, eigvec, t):
            '''
            ds/dt
            '''
            g_prime = eigvec.T @ grad.reshape(-1, 1)
            euler_step = 0
            for ii in range(n_eigval):
                euler_step += g_prime[ii]**2 * np.exp(-2 * eigvals[ii] * t)
            euler_step = euler_step ** 0.5
            return euler_step
        
        def A(t, eigvals, eigvec):
            alpha_mat = np.zeros((n_eigval, n_eigval))
            for ii in range(n_eigval):
                alpha_mat[ii][ii] = (np.exp(-eigvals[ii]*t) - 1)/eigvals[ii] 
            return eigvec @ alpha_mat @ eigvec.T

        t, s = euler_integral(step_size, grad, eigvals, eigvec)
        lqa_step = A(t, eigvals, eigvec) @ grad.reshape(-1,1)
        lqa_step = lqa_step.reshape(-1,3)
        
        return lqa_step


# ==============================================================
# Hessian estimating functions
# ==============================================================
# Currently, PSB, Bofill and SR1 are implemented

def psb(prev_hessian, prev_grad, current_grad, prev_step):
    '''
    Or Powell updating
    DOI: 10.1007/BF01584071
    '''

    delta_grad = (current_grad - prev_grad).reshape(-1,1)
    prev_step = prev_step.reshape(-1,1)
    xi = delta_grad - prev_hessian @ prev_step

    delta_hessian = -(prev_step.T @ xi)/((prev_step.T @ prev_step)**2) * prev_step@prev_step.T + \
                    (xi.dot(prev_step.T) + prev_step.dot(xi.T))/(prev_step.T @ prev_step)
    return delta_hessian
    
def sr1(prev_hessian, prev_grad, current_grad, prev_step):
    '''
    Murtagh B A & Sargent RWH. 
    Computational experience with quadratically convergent minimisation methods. 
    Computer J. 13:185-94, 1970
    '''
    delta_grad = (current_grad - prev_grad).reshape(-1,1)
    prev_step = prev_step.reshape(-1,1)

    xi = delta_grad - prev_hessian @ prev_step

    delta_hessian = (xi @ xi.T) / (xi.T @ prev_step)
    return delta_hessian
    
def bofill(prev_hessian, prev_grad, current_grad, prev_step):
    '''
    DOI: 10.1002/jcc.540150102

    dH = (1-psi) * dH_MS + psi * dH_psb
    '''

    delta_hessian_ms = sr1(prev_hessian, prev_grad, current_grad, prev_step)
    delta_hessian_psb = psb(prev_hessian, prev_grad, current_grad, prev_step)

    delta_grad = (current_grad - prev_grad).reshape(-1,1)
    prev_step = prev_step.reshape(-1,1)
    
    xi = delta_grad - prev_hessian @ prev_step
    psi = 1 - (prev_step.T @ xi) ** 2 / ((prev_step.T @ prev_step) * (xi.T @ xi))
    psi = psi.item() 
    delta_hessian = (1-psi) * delta_hessian_ms + psi * delta_hessian_psb
    return delta_hessian


# ==============================================================
# Utilities
# ==============================================================

def check_molecule(molecule:data.molecule):

    # check if frequency exist
    if "frequencies" not in molecule.__dict__:
        print('\nFrequencies and normal modes not found in molecule. Start frequency calculations with default settings ...\n'); sys.stdout.flush()
        return False

    if molecule.frequencies[0] > 0:
        raise ValueError("No imaginary frequency found! IRC calculations stopped.\n")
    elif molecule.frequencies[1] < 0:
        raise ValueError("More than one imaginary frequency found! IRC calculations stopped.\n")
    
    if molecule.frequencies[0] > -200:
        print("Imaginary frequency less than 200 cm-1 detected. IRC calculations will proceed. Please check the geometry."); sys.stdout.flush()
    return True

def to_mwcoord(nuclear_masses, xyz_coordiantes):
    nuclear_masses = nuclear_masses[:,None]
    return xyz_coordiantes * (nuclear_masses**0.5)

def to_mwgrad(nuclear_masses, gradients):
    nuclear_masses = nuclear_masses[:,None]
    return gradients / (nuclear_masses**0.5)

def to_mwhess(nuclear_masses, hessian):
    hess_scale = np.repeat(nuclear_masses**-0.5, 3).reshape(hessian.shape[0], -1)
    return hessian * np.outer(hess_scale, hess_scale)

def upd_coords(molecule, mass_weighted_xyz_coordinates):
    updated_molecule = molecule.copy(atomic_labels=[], molecular_labels=['charge', 'multiplicity'])
    updated_molecule.xyz_coordinates = mass_weighted_xyz_coordinates/molecule.nuclear_masses[:, None]**0.5
    return updated_molecule

def highlight_print(text:str, length:int=72, return_string=False):
    ptime = f"{datetime.now().strftime('%D %H:%M:%S')}"
    nchar = len(text); ntime = len(ptime)
    corner = '+'; hor = '-'; ver = '|'
    ubound = hor*((length-len(ptime))//2) + ptime + hor*((length-len(ptime))//2+1)
    lbound = hor*length
    htext = f"\n{ubound}\n{text}\n{lbound}"
    if return_string: return htext
    else: print(htext); sys.stdout.flush()

def settings_print(name:str, settings:dict):
    from typing import Callable
    print(f"{name} settings:")
    if len(settings) == 0:
        print("\tNone")
        return 
    for kk, vv in settings.items():
        if kk == 'program_kwargs' or kk == 'self' or 'trajdb' in kk:
            continue
        if kk == 'model' and "method" in vv.__dict__: print(f"\t{kk:<20} : {vv.method}")
        if not isinstance(vv, Callable) and kk != "molecule": print(f"\t{kk:<20} : {vv}")