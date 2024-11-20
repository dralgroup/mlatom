#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! spectra: Module for working with spectra                                  ! 
  ! Implementations by: Yi-Fan Hou, Fuchun Ge, Bao-Xin Xue, Pavlo O. Dral     !
  !---------------------------------------------------------------------------! 
'''

import os, copy
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance
from . import constants
from . import data
from .stopper import stopMLatom

class spectrum():
    def __init__(self, x=None, y=None, xy_pairs=None, xy=None, sort=False):
        self.set_data(x=x, y=y, xy_pairs=xy_pairs, xy=xy, sort=sort)

    def set_data(self, x=None, y=None, xy_pairs=None, xy=None, sort=False):
        if x is not None:
            self.x = data.array(x) ; self.x=self.x.astype(np.float64)
        if y is not None:
            self.y = data.array(y) ; self.y=self.y.astype(np.float64)
            if x is not None:
                self.xy = np.array([x, y])
                self.xy_pairs = self.xy.T
        if xy_pairs is not None:
            self.xy_pairs = data.array(xy_pairs) ; self.xy_pairs = self.xy_pairs.astype(np.float64)
            self.xy = self.xy_pairs.T
            self.x,self.y = self.xy
        if xy is not None:
            self.xy = data.array(xy) ; self.xy = self.xy.astype(np.float64)
            self.xy_pairs = self.xy.T
            self.x,self.y = self.xy
        if sort:
            self.xy_pairs =  self.xy_pairs[np.argsort(self.xy_pairs[:,0])]
            self.xy = self.xy_pairs.T
            self.x,self.y = self.xy

    def interpolate(self, method="linear"):
        if method.casefold() == "linear".casefold():
            interpolation_function = interp1d(self.x,self.y,kind='linear')
        elif method.casefold() == "cubic".casefold():
            interpolation_function = interp1d(self.x,self.y,kind='cubic')
        self.interpolation_function = interpolation_function

    # Normalize the Riemann sum
    def normalize(self,method='average'):
        x,y = self.x,self.y
        dx = x[1:]-x[:-1]
        if method.casefold() == 'average'.casefold():
            Riemann_sum = np.sum(y[:-1]*dx)
            self.xy_pairs[:,1] /= Riemann_sum
            self.xy = self.xy_pairs.T 
            self.x = self.xy[0]
            self.y = self.xy[1]
        elif method.casefold() == 'sqrt'.casefold():
            squared_sum = np.sum(y[:-1]**2*dx)
            self.xy_pairs[:,1] /= np.sqrt(squared_sum)
            self.xy = self.xy_pairs.T 
            self.x = self.xy[0]
            self.y = self.xy[1]
        elif method.casefold() == 'max'.casefold():
            self.xy_pairs[:,1] /= np.max(y)
            self.xy = self.xy_pairs.T 
            self.x = self.xy[0]
            self.y = self.xy[1]
        elif method.casefold() == 'msc'.casefold():
            summ = np.sum(y)
            self.xy_pairs[:,1] = np.sqrt(self.xy_pairs[:,1]) / np.sqrt(summ)
            self.xy = self.xy_pairs.T 
            self.x = self.xy[0]
            self.y = self.xy[1]
        else:
            stopMLatom(f"Unsupported method for spectrum normalization: {method}")

    # Interpolate and sample the orginal spectrum 
    def sample(self, list, method="linear"):
        self.interpolate(method)
        intensities = self.interpolation_function(data.array(list))
        new_spectrum = copy.deepcopy(self)
        new_spectrum.set_data(x=data.array(list),y=intensities)
        return new_spectrum
    
    # Convolute the line spectrum
    @classmethod
    def broaden(cls,line_spectrum=None,spectrum_range=None,broadening_func="normalized_Gaussian",broadening_func_kwargs={}):
        if type(broadening_func) == str:
            if broadening_func.casefold() == "normalized_Gaussian".casefold():
                broadening_func = cls.normalized_Gaussian_function
            elif broadening_func.casefold() == "Gaussian".casefold():
                broadening_func = cls.Gaussian_function
            elif broadening_func.casefold() == 'Lorentzian'.casefold():
                broadening_func = cls.Lorentzian_line_shape_function
            else:
                stopMLatom(f"Unsupported broadening function type: {broadening_func}")            

        new_spectrum = cls()
        new_spectrum.x = data.array(spectrum_range)
        for ii in range(len(line_spectrum)):
            if ii==0:
                new_spectrum.y = broadening_func(line_spectrum[ii][0],line_spectrum[ii][1],new_spectrum.x,**broadening_func_kwargs)
            else:
                new_spectrum.y += broadening_func(line_spectrum[ii][0],line_spectrum[ii][1],new_spectrum.x,**broadening_func_kwargs)
        new_spectrum.set_data(x=new_spectrum.x, y=new_spectrum.y)
        return new_spectrum
    
    # Broadening functions
    # Normalized Gaussian function
    @classmethod
    def normalized_Gaussian_function(cls, mu, a, x, sigma=20):
        """
        Normalized Gaussian function

        Arguments:
            mu (float): peak position of Gaussian function
            a (float): scaling factor of the normalized Gaussian function
            x (float, np.darray): values of Gaussian function to be calculated at x
            sigma (float): standard deviation
            
        """
        f = ( a * (1 / (sigma * np.sqrt(2 * np.pi) ) )
          *   np.exp( -(x - mu) ** 2 / (2 * sigma ** 2) ) )
        return f
    
    # Gaussian function
    @classmethod
    def Gaussian_function(cls, mu, fmax, x, sigma=20): 
        """
        Gaussian function

        Arguments:
            mu (float): peak position of Gaussian function
            fmax (float): peak height of Gaussian function (maximum value at x=mu)
            x (float, np.darray): values of Gaussian function to be calculated at x
            sigma (float): standard deviation,

        """
        f = ( fmax
          *   np.exp( -(x - mu) ** 2 / (2 * sigma ** 2) ) )
        return f

    # Lorentzian line shape function
    @classmethod
    def Lorentzian_line_shape_function(cls,mu,fmax,x,w=30):
        """
        Lorentzian line shape function

        Arguments:
            mu (float): peak position
            fmax (float): peak intensity 
            x (float, np.darray): values to be calculated at x
            w (float): full width at half-maximum 

        """
        f = fmax / (1.0 + ((mu-x)/w*2)**2)
        return f

    def get_range(self, lb=-np.inf, ub=np.inf):
        idx = (self.xy_pairs[:,0] > lb) & (self.xy_pairs[:,0] < ub)
        return self.xy_pairs[idx]
    
    def del_range(self, lb, ub):
        idx = (self.xy_pairs[:,0] > lb) & (self.xy_pairs[:,0] < ub)
        self.xy_pairs = np.delete(self.xy_pairs, idx, axis=0)
        self.xy = self.xy_pairs.T
        self.x,self.y = self.xy

    def update(self, new_xy_pairs, overwrite_range=True):
        if overwrite_range:
            lb = np.min(new_xy_pairs[:,0])
            ub = np.max(new_xy_pairs[:,0])
            self.del_range(lb, ub)
        self.set_data(np.concatenate((self.xy_pairs, new_xy_pairs), axis=0))

    def dump(self, filename, format='txt'):
        if format == 'npy':
            np.save(filename, self.xy_pairs)
        elif format == 'txt':
            np.savetxt(filename, self.xy_pairs)
        else:
            print(' unsupported format')

    def plot(self,filename,xaxis_caption='',yaxis_caption='',title='',invert_xaxis=False,invert_yaxis=False):
        from .plot import plot
        spectrum_plot = plot()
        spectrum_plot.savein=filename
        spectrum_plot.plottype='linechart_without_points'
        spectrum_plot.xs.append(self.x)
        spectrum_plot.ys.append(self.y)
        spectrum_plot.xaxis_caption = xaxis_caption
        spectrum_plot.yaxis_caption = yaxis_caption
        spectrum_plot.title = title
        spectrum_plot.invert_xaxis = invert_xaxis
        spectrum_plot.invert_yaxis = invert_yaxis
        spectrum_plot.make_figure()

    def _load(self, filename, format='txt', sort=False):
        if format == 'npy':
            xy_pairs = np.load(filename)
        elif format == 'txt':
            xy_pairs = np.loadtxt(filename)
        else:
            print(' unsupported format')
        self.set_data(xy_pairs=xy_pairs, sort=sort)

    @classmethod
    def load(cls, filename, format='txt',sort=False):
        return_value = cls()
        return_value._load(filename,format=format,sort=sort)
        return return_value
    
class uvvis(spectrum):
    '''
    UV/Vis absorption spectrum class
    
    Arguments:
        x (float, np.ndarray):                 range of spectra (e.g., wavelength in nm, recommended, or energies in eV)
        y (float, np.ndarray):                 user-provided intensities (e.g., molar absorpbance, recommended, or cross section)
        It is better to provide spectrum information explicitly so that the correct conversions to different units are done:
        wavelengths_nm (float, np.ndarray):    range of wavelengths in nm
        energies_eV (float, np.ndarray):       range of energies in eV
        molar_absorbance (float, np.ndarray):  molar absorbance (extinction coefficients) in M^-1 cm^-1
        cross_section (float, np.ndarray):     cross section in A^2/molecule
        Also, the user is encouraged to provide the meta-data:
        meta_data (str):                       meta data such as solvent, references, etc.

    Example:
        uvvis = mlatom.spectra.uvvis(
            wavelengths_nm    = np.array(...),
            molar_absorbance  = np.array(...),
            meta_data         = 'solvent: benzene, reference: DOI...'
            )
        # spectral properties can be accessed as:
        # uvvis.x is equivalent to what is provided by the user, e.g., wavelengths_nm or energies_eV
        # uvvis.y is equivalent to what is provided by the user, e.g., molar_absorbance or cross_section
        # wavelength range (float, np.ndarray) in nm 
        uvvis.wavelengths_nm
        # molar absorbance (extinction coefficients) (float, np.ndarray) in M^-1 cm^-1
        uvvis.molar_absorbance
        # energies corresponding to the wavelength range (float, np.ndarray), in eV
        uvvis.energies_eV
        # absorption cross-section (float, np.ndarray) in A^2/molecule
        uvvis.cross_section
    '''
    def __init__(self,
                 x=None, y=None,
                 wavelengths_nm=None, energies_eV=None,
                 molar_absorbance=None, cross_section=None, 
                 meta_data=None):
        if x is not None:
            self.x = x
        elif wavelengths_nm is not None:
            self.x = wavelengths_nm
            self.wavelengths_nm = wavelengths_nm
            self.energies_eV = constants.nm2eV(self.wavelengths_nm)
        elif energies_eV is not None:
            self.x = energies_eV
            self.energies_eV = energies_eV
            self.wavelengths_nm = constants.eV2nm(self.energies_eV)
        if y is not None:
            self.y = y
        elif molar_absorbance is not None:
            self.y = molar_absorbance
            self.molar_absorbance = molar_absorbance
            self.cross_section = molar_absorbance * 3.82353e-5
        elif cross_section is not None:
            self.y = cross_section
            self.cross_section = cross_section
            self.molar_absorbance = cross_section / 3.82353e-5
        if 'x' in self.__dict__ and 'y' in self.__dict__:
            self.xy = np.array([self.x, self.y])
            self.xy_pairs = self.xy.T
        if meta_data is not None:
            self.meta_data = meta_data            

    def plot(self,filename=None,xaxis_caption='Wavelength (nm)',yaxis_caption='Extinction coefficient (M$^-1$ cm$^-1$)',title='UV-Vis spectrum'):
        plot_uvvis(spectra=[self], filename=filename, xaxis_caption=xaxis_caption, yaxis_caption=yaxis_caption, title=title)

    @classmethod
    def spc(cls, molecule=None,
                 band_width=0.3, shift=0.0, refractive_index=1.0):
        '''
        Single-point convolution (SPC) approach for obtaining UV/vis spectrum
        via calculating the exctinction coefficient (and absorption cross section)
        from the single-point excited-state simulations
        for a single geometry
        Implementation follows http://doi.org/10.1007/s00894-020-04355-y

        Arguments:
            molecule (:class:`mlatom.data.molecule`): molecule object with
                                                   excitation_energies (in Hartree, not eV!)
                                                   and oscillator_strengths
            wavelengths_nm (float, np.ndarray):    range of wavelengths in nm (default: np.arange(400, 800))
            band_width (float):                    band width in eV (default: 0.3 eV)
            shift (float):                         shift of excitation energies, eV (default: 0 eV)
            refractive_index (float):              refractive index (default: 1)
        
        Example:
            uvvis = mlatom.spectra.uvvis.spc(
                        molecule=mol,
                        wavelengths_nm=np.arange(100, 200),
                        band_width=0.3)
            # spectral properties can be accessed as:
            # uvvis.x is equivalent to uvvis.wavelengths_nm
            # uvvis.y is equivalent to uvvis.molar_absorbance
            # wavelength range (float, np.ndarray) in nm 
            uvvis.wavelengths_nm
            # molar absorbance (extinction coefficients) (float, np.ndarray) in M^-1 cm^-1
            uvvis.molar_absorbance
            # energies corresponding to the wavelength range (float, np.ndarray), in eV
            uvvis.energies_eV
            # absorption cross-section (float, np.ndarray) in A^2/molecule
            uvvis.cross_section
            # quick plot
            uvvis.plot(filename='uvvis.png')
        '''
        excitation_energies = molecule.excitation_energies * constants.hartree2eV
        wavelengths_nm = np.arange(constants.eV2nm(np.max(excitation_energies) + 3*band_width),
                                   constants.eV2nm(np.min(excitation_energies) - 3*band_width),
                                   0.2
                                   )
        new_spectrum = cls.broaden(line_spectrum=np.array([excitation_energies,
                                                     molecule.oscillator_strengths
                                                    ]).T,
                                    spectrum_range=wavelengths_nm,
                                    broadening_func = cls.spc_broadening_func,
                                    broadening_func_kwargs={'band_width':       band_width,
                                                            'shift':            shift,
                                                            'refractive_index': refractive_index}
                                  )
        new_spectrum.wavelengths_nm = new_spectrum.x # wavelengths in nm
        new_spectrum.molar_absorbance = new_spectrum.y # extinction coefficients in M^-1 cm^-1
        new_spectrum.energies_eV = constants.nm2eV(new_spectrum.wavelengths_nm)
        new_spectrum.cross_section = new_spectrum.molar_absorbance * 3.82353e-5  # absorption cross-section in A^2/molecule
        return new_spectrum
    
    # SPC function
    @classmethod
    def spc_broadening_func(cls, DeltaE, ff, wavelength_range, band_width, refractive_index=1, shift=0.0): 
        # http://doi.org/10.1007/s00894-020-04355-y
        """
        Spectrum convolution function

        Arguments:
            band_width (float): width of band
            DeltaE (float): vertical excitation energy, eV
            ff (float): oscillator strength
            wavelength_range (float, np.ndarray): range of wavelengths
            refractive_index (float): refractive index
            shift (float): peak shift
        
        Returns:
            (float, np.ndarray): extinction coefficients in M^-1 cm^-1
            
        """
        f = ( 0.619 * refractive_index * ff / (band_width * 3.82353e-5)
          *   np.exp( -(constants.nm2eV(wavelength_range) - DeltaE + shift) ** 2 / (band_width ** 2) ) )
        return f

    @classmethod
    def nea(cls, molecular_database=None,
                 wavelengths_nm=None,
                 broadening_width=0.05):
        '''
        Nuclear ensemble approach (NEA) for obtaining UV/vis spectrum.
        Implementation follows Theor. Chem. Acc. 2012, 131, 1237.

        Arguments:
            molecular_database (:class:`mlatom.data.molecular_database`): molecular_database object
                                                   with molecules containing
                                                   excitation_energies (in Hartree, not eV!)
                                                   and oscillator_strengths
            wavelengths_nm (float, np.ndarray):    range of wavelengths in nm (default: determined automatically)
            broadening_width (float):              broadening factor in eV (default: 0.05 eV)
        
        Example:
            uvvis = mlatom.spectra.uvvis.nea(molecular_database=db,
                                             wavelengths_nm=wavelengths_nm,
                                             broadening_width=0.02)
            # spectral properties can be accessed as:
            # uvvis.x is equivalent to uvvis.wavelengths_nm
            # uvvis.y is equivalent to uvvis.molar_absorbance
            # wavelength range (float, np.ndarray) in nm 
            uvvis.wavelengths_nm
            # molar absorbance (extinction coefficients) (float, np.ndarray) in M^-1 cm^-1
            uvvis.molar_absorbance
            # energies corresponding to the wavelength range (float, np.ndarray), in eV
            uvvis.energies_eV
            # absorption cross-section (float, np.ndarray) in A^2/molecule
            uvvis.cross_section
            # quick plot
            uvvis.plot(filename='uvvis.png')
        '''

        from ctypes import c_double, CDLL, c_long 

        npoints = len(molecular_database)
        nexcitations = molecular_database[0].nstates-1

        # calculate required prefactors
        nref = 1 # ratio
        prefactor = np.pi * constants.electron_charge**2 / (2 * constants.electron_mass * constants.speed_of_light * constants.eps0 * nref)  # m^2/s  unit
        hplanck = constants.planck_constant * constants.J2hartree * constants.hartree2eV
        prefactor = prefactor * hplanck / (2 * np.pi) * 1E20 # Angstrom^2*eV
        exp_prefactor = 1 / (broadening_width * (np.pi / 2) ** 0.5)

        if wavelengths_nm is None:
            min_es = min([np.min(molecular_database[ipoint].excitation_energies) for ipoint in range(npoints)]) * constants.hartree2eV
            max_es = max([np.max(molecular_database[ipoint].excitation_energies) for ipoint in range(npoints)]) * constants.hartree2eV
            wavelengths_nm = np.arange(constants.eV2nm(max_es + 3*broadening_width),
                                       constants.eV2nm(min_es - 3*broadening_width),0.2)
        new_spectrum = cls()
        new_spectrum.wavelengths_nm = data.array(wavelengths_nm)
        new_spectrum.x = new_spectrum.wavelengths_nm
        new_spectrum.energies_eV = constants.nm2eV(new_spectrum.wavelengths_nm)
        n_spectra_points = len(new_spectrum.energies_eV)

        # Convert to C-type
        c_excitation_energies_eV = ((c_double * npoints) * nexcitations)() # C-type list with excitation energies in eV for all points
        c_oscillator_strengths = ((c_double * npoints) * nexcitations)() # C-type list with oscillator strengths for all points
        for iex in range(nexcitations):
            for ipoint in range(npoints):
                c_excitation_energies_eV[iex][ipoint] = molecular_database[ipoint].excitation_energies[iex] * constants.hartree2eV
                c_oscillator_strengths[iex][ipoint] = molecular_database[ipoint].oscillator_strengths[iex]
        c_broadening_width = c_double(broadening_width)
        c_exp_prefactor = c_double(exp_prefactor)
        c_prefactor = c_double(prefactor)
        c_n_spectra_points = c_long(n_spectra_points)
        c_energies_eV = (c_double* n_spectra_points)()
        for ii in range(n_spectra_points):
            c_energies_eV[ii] = new_spectrum.energies_eV[ii]
        c_cross_section = (c_double * n_spectra_points)()

        # calculate cross section
        py_script_path   = os.path.abspath(__file__[:__file__.rfind('/')])
        c_calculate_cross_section = CDLL(os.path.join(py_script_path, 'cs.so'))
        _ = c_calculate_cross_section.cs_calc(c_excitation_energies_eV, c_oscillator_strengths, 
                    nexcitations, npoints,
                    c_broadening_width, c_exp_prefactor, c_n_spectra_points, c_prefactor,
                    c_cross_section, c_energies_eV)
        new_spectrum.cross_section = data.array(c_cross_section)   # absorption cross-section in A^2/molecule
        new_spectrum.molar_absorbance = new_spectrum.cross_section / 3.82353e-5 # extinction coefficients in M^-1 cm^-1
        new_spectrum.y = new_spectrum.molar_absorbance
        return new_spectrum

class ir(spectrum):

    def __init__(self,
                x=None, y=None,
                frequencies=None,infrared_intensities=None,
                meta_data=None):
        if x is not None:
            self.x = x 
        elif frequencies is not None:
            self.x = frequencies 
            self.frequencies = frequencies
        if y is not None:
            self.y = y 
        elif infrared_intensities is not None:
            self.y = infrared_intensities 
            self.infrared_intensities = infrared_intensities
        if 'x' in self.__dict__ and 'y' in self.__dict__:
            self.xy = np.array([self.x,self.y])
            self.xy_pairs = self.xy.T 
        if meta_data is not None:
            self.meta_data = meta_data 
        else:
            self.meta_data = None

    def plot(self,filename,xaxis_caption='Wavenumber (cm$^{-1}$)',yaxis_caption='Intensity (km/mol)',title='IR spectrum'):
        super().plot(filename=filename,xaxis_caption=xaxis_caption,yaxis_caption=yaxis_caption,title=title,invert_xaxis=True)

    @classmethod 
    def lorentzian(cls,molecule=None,fwhm=30,spectrum_range=np.arange(500,4001)):
        frequencies = molecule.frequencies
        infrared_intensities = molecule.infrared_intensities 
        new_spectrum = cls.broaden(line_spectrum=np.array([frequencies,infrared_intensities]).T,
                                   spectrum_range=spectrum_range,
                                   broadening_func='Lorentzian',
                                   broadening_func_kwargs={'w':fwhm})
        new_spectrum.frequencies = new_spectrum.x 
        new_spectrum.infrared_intensities = new_spectrum.y 
        return new_spectrum

class spectrum_comparison():
    def __init__(self):
        pass 

    @classmethod
    def spectrum_comparison(cls,spectrum1, spectrum2, metric, align_method_dict={},metric_arg_dict={}, line_up=True):
        # By default, spectrum1 is the reference spectrum
        # Align two spetra before comparison
        
        if line_up:
            _spec1,_spec2 = cls.line_up_spectra(spectrum1,spectrum2,**align_method_dict)
            spec1 = _spec1.y
            spec2 = _spec2.y
        else:
            spec1 = spectrum1.y
            spec2 = spectrum2.y

        # Pearson correlation coefficient (PCC)
        if metric.casefold() == 'PCC'.casefold():
            return cls.pearson_coefficient(spec1,spec2,**metric_arg_dict)
        # Spearman correlation coefficient (SCC)
        elif metric.casefold() == 'SCC'.casefold():
            return cls.spearman_coeffient(spec1,spec2,**metric_arg_dict)
        # Tanimoto coefficient 
        elif metric.casefold() == 'Tanimoto'.casefold():
            return cls.tanimoto_coefficient(spec1,spec2,**metric_arg_dict)
        # Kullback-Leibler Divergence (KLD)
        elif metric.casefold() == 'KLD'.casefold():
            return cls.KL_divergence_transformed(spec1,spec2,**metric_arg_dict)
        # Jeffrey Divergence (JD)
        elif metric.casefold() == 'JD'.casefold():
            return cls.jeffery_divergence_transformed(spec1,spec2,**metric_arg_dict)
        # Jensen-Shannon divergence (JSD)
        elif metric.casefold() == 'JSD'.casefold():
            return cls.JS_divergence_transformed(spec1,spec2,**metric_arg_dict)
        # Earth-Mover distance (EMD)
        elif metric.casefold() == 'EMD'.casefold():
            return cls.earth_mover_distance_transformed(spec1,spec2,**metric_arg_dict)
        # Mean square error (MSE)
        elif metric.casefold() == 'MSE'.casefold():
            return cls.mse_transformed(spec1,spec2,**metric_arg_dict)
        # Root mean square error (RMSE)
        elif metric.casefold() == 'RMSE'.casefold():
            return cls.rmse_transformed(spec1,spec2,**metric_arg_dict)
        # Mean absolute error (MAE)
        elif metric.casefold() == 'MAE'.casefold():
            return cls.mae_transformed(spec1,spec2,**metric_arg_dict)
        # Spectral information similarity (SIS)
        elif metric.casefold() == 'SIS'.casefold():
            return cls.spectral_information_similarity(spec1,spec2,**metric_arg_dict)
        # Root mean spuare deviation (RMSD)
        elif metric.casefold() == 'RMSD'.casefold():
            return cls.rmsd_transformed(spec1,spec2,**metric_arg_dict)
        # Euclidean distance 
        elif metric.casefold() == 'Euclidean'.casefold():
            return cls.euclidean_transformed(spec1,spec2,**metric_arg_dict)
        # Cosine
        elif metric.casefold() == 'cosine'.casefold():
            return cls.cosine(spec1,spec2,**metric_arg_dict)
        # Absolute difference value search (ADV)
        elif metric.casefold() == 'ADV'.casefold():
            return cls.absolute_difference_value_search(spec1,spec2,**metric_arg_dict)
        # Relative integral change (RIC)
        elif metric.casefold() == 'RIC'.casefold():
            return cls.relative_integral_change(spec1,spec2,**metric_arg_dict)
        else:
            stopMLatom(f'Unrecognized metric: {metric}')

    @classmethod
    def line_up_spectra(cls,spectrum1,spectrum2,interpolate_method="linear",align_list=None):
        range1 = spectrum1.x
        range2 = spectrum2.x
        range_lb = np.max([np.min(range1),np.min(range2)])
        range_ub = np.min([np.max(range1),np.max(range2)])
        if align_list == None:
            align_list = np.linspace(range_lb,range_ub,int((range_ub-range_lb)*2+1))
        else:
            align_list = data.array(align_list)
            align_list = align_list[align_list>=range_lb]
            align_list = align_list[align_list<=range_ub]
        spec1 = spectrum1.sample(align_list,method=interpolate_method)
        spec2 = spectrum2.sample(align_list,method=interpolate_method)
        return spec1, spec2
        
    """
    Functions measuring similarity between spectra. Implemented by Yangtao Chen & Yifan Hou
    """
    @classmethod
    def loss2similarity(cls,loss,alpha=0.2):
        """
        Transform loss value to similarity(range from 0 to 1) by the formula: similarty = exp(-alpha * loss), where alpha is a hyperparameter.
        
        Parameters
        ----------
        loss : float
            The loss value between two spectra. The smaller the loss, the more similar the two spectra.
        """
        return np.exp(-alpha * loss)

    @classmethod 
    def pearson_coefficient(cls,spectra_ref,spectra_pred):
        # https://pubs.acs.org/doi/10.1021/acs.jctc.0c01279?ref=PDF
        """
        Calculates the Pearson correlation coefficient between two spectra.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        r_pearson: float
            Pearson correlation coefficient range from [-1, 1]
        """
        spectra_ref_mean = np.mean(spectra_ref)
        spectra_pred_mean = np.mean(spectra_pred)
        spectra_ref_var = np.sum((spectra_ref-spectra_ref_mean)**2)
        spectra_pred_var = np.sum((spectra_pred-spectra_pred_mean)**2)
        r_pearson = np.sum((spectra_ref-spectra_ref_mean)*(spectra_pred-spectra_pred_mean))/np.sqrt(spectra_ref_var*spectra_pred_var)
        return r_pearson

    @classmethod
    def spearman_coeffient(cls,spectra_ref,spectra_pred):
        """
        Calculates the  Spearman rank correlation coefficient between two spectra.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        r_spearman: float
            nonlinear Spearman rank correlation coefficient
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        k = len(spectra_ref)
        rank_true = {xi: i+1 for i, xi in enumerate(sorted(spectra_ref))}
        rank_pred = {yi: i+1 for i, yi in enumerate(sorted(spectra_pred))}
        d = [rank_true[xi] - rank_pred[yi] for xi, yi in zip(spectra_ref, spectra_pred)]
        d_square_sum = sum(d_i ** 2 for d_i in d)
        r_spearman = 1 - (6 * d_square_sum) / (k * (k ** 2) -1 )
        return r_spearman

    @classmethod
    def tanimoto_coefficient(cls,spectra_ref,spectra_pred):
        """
        Calculates the  Tanimoto coefficient between two spectra.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        t: float
            Tanimoto coefficient
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        t = (np.sum(spectra_ref * spectra_pred)) / (np.sum(spectra_ref ** 2) + np.sum(spectra_pred ** 2) - np.sum(spectra_ref * spectra_pred))
        return t
    
    @classmethod
    def KL_divergence_transformed(cls,spectra_ref, spectra_pred):
        """
        Calculates the transformed Kullback-Leibler divergence between two spectra. Note that the KL divergence is not symmetric.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        KL_divergence_transformed
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        KL_divergence = np.sum(spectra_ref * np.log(spectra_ref / (spectra_pred + 1e-8)))
        return cls.loss2similarity(KL_divergence)

    @classmethod 
    def jeffery_divergence_transformed(cls,spectra_ref, spectra_pred):
        """
        Calculates the transformed Jeffery divergence between two spectra. Note that the Jeffery divergence is not symmetric.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        jeffery_divergence_transformed
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        jeffery_divergence = np.sum((spectra_ref - spectra_pred) * np.log(spectra_ref / (spectra_pred + 1e-8)))
        return cls.loss2similarity(jeffery_divergence)

    @classmethod 
    def JS_divergence_transformed(cls,spectra_ref, spectra_pred):
        """
        Calculates the transformed Jensen–Shannon divergence between two spectra. Note that the Jensen–Shannon divergence is symmetric.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        JS_divergence_transformed
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        spectra_mean = (spectra_ref + spectra_pred) / 2
        JS_divergence = 1/2 * np.sum(spectra_ref * np.log(spectra_ref / (spectra_mean + 1e-8))) + 1/2 * np.sum(spectra_pred * np.log(spectra_pred / (spectra_mean + 1e-8)))
        return cls.loss2similarity(JS_divergence, 0.8)

    @classmethod
    def earth_mover_distance_transformed(cls,spectra_ref, spectra_pred):
        """
        Calculates the transformed Wasserstein distance(earth mover distance) between two spectra implemented by scipy. Note that this value can be used between two spectra containing different number of wavelength.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        wasserstein_distance_transformed
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        return cls.loss2similarity(wasserstein_distance(spectra_ref, spectra_pred))

    @classmethod
    def mse_transformed(cls,spectra_ref, spectra_pred):
        """
        Calculates the transformed Mean Square Error(MSE) between two spectra.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        mse_transformed
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        mse = np.mean((spectra_ref - spectra_pred) ** 2)
        return cls.loss2similarity(mse)

    @classmethod
    def rmse_transformed(cls,spectra_ref, spectra_pred):
        """
        Calculates the transformed Root Mean Square Error(RMSE) between two spectra.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        rmse_transformed
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        rmse = np.sqrt(np.mean((spectra_ref - spectra_pred) ** 2))
        return cls.loss2similarity(rmse)

    @classmethod
    def mae_transformed(cls,spectra_ref, spectra_pred):
        """
        Calculates the transformed Mean Absolute Error(MAE) between two spectra.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        mae_transformed
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        mae = np.mean(np.abs(spectra_ref - spectra_pred))
        return cls.loss2similarity(mae)

    @classmethod
    def spectral_information_similarity(cls,spectra_ref, spectra_pred, threshold=1e-10, std_dev=10):
        # reference: https://github.com/gfm-collab/chemprop-IR
        """
        Calculates the spectral_information_similarity(SIS) between two spectra.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        sim
        """
        # spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        # length = len(spectra_ref)
        # frequencies = list(range(400, 400 + length))
        # gaussian=[(1/(2*np.pi*std_dev**2)**0.5)*np.exp(-1*((frequencies[i])-frequencies[0])**2/(2*std_dev**2)) for i in range(length)]
        # conv_matrix=np.empty([length,length])
        # for i in range(length):
        #     for j in range(length):
        #         conv_matrix[i,j]=gaussian[abs(i-j)]
        # nan_mask=np.isnan(spectra_ref)+np.isnan(spectra_pred)
        # # print(length,conv_matrix.shape,spectra_ref.shape,spectra_pred.shape)
        # assert length == len(spectra_pred), "compared spectra are of different lengths"
        # assert length == len(frequencies), "compared spectra are a different length than the frequencies list, which can be specified"
        # spectra_ref[spectra_ref<threshold]=threshold
        # spectra_pred[spectra_pred<threshold]=threshold
        # spectra_ref[nan_mask]=0
        # spectra_pred[nan_mask]=0
        # # print(spectra_ref.shape,spectra_pred.shape)
        # spectra_ref=np.expand_dims(spectra_ref,axis=0)
        # spectra_pred=np.expand_dims(spectra_pred,axis=0)
        # # print(spectra_ref.shape,spectra_pred.shape)
        # conv1=np.matmul(spectra_ref,conv_matrix)
        # # print(conv1[0,1000])
        # conv2=np.matmul(spectra_pred,conv_matrix)
        # conv1[0,nan_mask]=np.nan
        # conv2[0,nan_mask]=np.nan
        # # print(conv1.shape,conv2.shape)
        # sum1=np.nansum(conv1)
        # sum2=np.nansum(conv2)
        # norm1=conv1/sum1
        # norm2=conv2/sum2
        # distance=norm1*np.log(norm1/norm2)+norm2*np.log(norm2/norm1)
        # sim=1/(1+np.nansum(distance))
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        spectra_ref[spectra_ref<threshold]=threshold
        spectra_pred[spectra_pred<threshold]=threshold

        norm1 = spectra_ref 
        norm2 = spectra_pred

        distance=norm1*np.log(norm1/norm2)+norm2*np.log(norm2/norm1)
        sim=1/(1+np.nansum(distance))
        return sim
    
    @classmethod 
    def rmsd_transformed(cls,spectra_ref,spectra_pred):
        # https://pubs.acs.org/doi/10.1021/acs.jctc.0c01279?ref=PDF
        """
        Calculates the transformed Root Mean Square Deviation (RMSD) between two spectra.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        rmsd_transformed
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        rmse = np.sqrt(np.mean((spectra_ref - spectra_pred) ** 2))
        return cls.loss2similarity(rmse)

    @classmethod 
    def euclidean_transformed(cls,spectra_ref,spectra_pred):
        # https://pubs.acs.org/doi/10.1021/acs.jctc.0c01279?ref=PDF
        """
        Calculates the transformed Euclidean distance between two spectra.

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        euclidean_transformed
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        euclidean = np.sqrt(np.sum((spectra_ref-spectra_pred)**2))
        return 1.0 / (1.0+euclidean)


    @classmethod 
    def cosine(cls,spectra_ref,spectra_pred):
        """
        Calculates the cosine value of the angle between two spectra (vectors).

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        cosine
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        return np.sum(spectra_ref*spectra_pred)/np.sqrt(np.sum(spectra_ref**2)*np.sum(spectra_pred**2))

    @classmethod 
    def absolute_difference_value_search(cls,spectra_ref,spectra_pred):
        # https://journals.pan.pl/dlibra/publication/122822/edition/107074/content
        """
        Calculates the Absolute difference value search (ADV) between two spectra (vectors).

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        absolute_difference_value_search
        """
        spectra_ref, spectra_pred = data.array(spectra_ref), data.array(spectra_pred)
        absolute_difference = np.sum(np.abs(spectra_ref-spectra_pred))
        reference = np.sum(np.abs(spectra_ref))
        return 1.0 - absolute_difference / reference

    @classmethod 
    def relative_integral_change(cls,spectra_ref,spectra_pred,align_list):
        """
        Calculates the relative integral change (RIC) between reference and predicted spectra

        Parameters
        ----------
        spectra_ref : list or np.ndarray data of shape(n_wavelength, )
            True spectrum.
        spectra_pred : list or np.ndarray data of shape(n_wavelength, )
            Predicted spectrum.

        Returns
        -------
        1.0 - relative_integral_change
        """
        align_list = data.array(align_list)
        dx = align_list[1:]-align_list[:-1]
        diff = np.abs(spectra_ref - spectra_pred)
        Riemann_sum = np.sum(diff[:-1]*dx)
        return 1.0 - Riemann_sum / np.sum(spectra_ref[:-1]*dx)
    
def align_spectra(spectrum1,spectrum2,interpolate_method="linear",align_list=None):
    pass

def plot_spectra(spectra=None, linespectra=None,
                filename=None, 
                title=None,
                xaxis_caption='', yaxis_caption='', y2axis_caption='',
                labels=[],
                colors=[],
                normalize=False,
                shift=False, shiftby=None,
                plotstart=None, plotend=None,
                ):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['axes.linewidth'] = 2

    spectra = copy.deepcopy(spectra)
    linespectra = copy.deepcopy(linespectra)
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.85)
    lines = []

    if colors == []:
        colors = ['k', 'r'] #None for xx in spectra]
    if labels == []:
        labels = [None for xx in spectra]
    
    if normalize:
        for ii in range(len(spectra)):
            ymax = max(spectra[ii].y)
            spectra[ii].y = [zz / ymax for zz in spectra[ii].y]
            
    if shift:
        # Superimposes global maxima
        ix1m = np.argmax(spectra[0].y)
        ix2m = np.argmax(spectra[1].y)
        
        #cls.delta = cls.xxs[0][ix1m] - cls.x2ys[0][ix2m]
        if not shiftby:
            shiftby = spectra[0].x[ix1m] - spectra[1].x[ix2m]
        print('Theoretical spectrum is shifted by %.2f nm' % shiftby)
        spectra[1].x += shiftby
        if linespectra is not None:
            linespectra[0].x += shiftby
    
    for ii in range(len(spectra)):
        if ii < len(labels): label=labels[ii]
        else: label=None
        if ii < len(colors): color=colors[ii]
        else: color=None
        lines.append(ax.plot(spectra[ii].x, spectra[ii].y, color=color, label=label,
                        linewidth=None, marker=None, markersize=10, mfc='none')[0])

    if title != None: plt.title(title, fontsize=18)
    plt.xlabel(xaxis_caption, fontsize=18)
    plt.ylabel(yaxis_caption, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,3))#, labelsize=18)
    ax.xaxis.get_offset_text().set_fontsize(18)
    ax.yaxis.get_offset_text().set_fontsize(18)

    if linespectra is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel(y2axis_caption,
                        fontsize=18, color='red')
        if normalize:
            y2max = 1.0
        else:
            y2max = max(linespectra[0].y) * 1.1
        ax2.set_ylim(ymin=-0.05, ymax=y2max)
        ax2.tick_params(axis='both', colors='red')
        for ii in range(len(linespectra[0].x)):
            ax2.plot((linespectra[0].x[ii], linespectra[0].x[ii]),
                        (-0.05, linespectra[0].y[ii]), 'r', linewidth=1)
        zed = [tick.set_fontsize(18)
                for tick in ax2.yaxis.get_ticklabels()]

    if plotstart:
        plt.xlim(left=plotstart)
    if plotend:
        plt.xlim(right=plotend)

    if not all(label == None for label in labels):
        plt.legend(lines, [ll.get_label() for ll in lines],
                    frameon=False, fontsize=18, loc='best')

    ax.set_ylim(bottom=-0.05)
    if filename:
        plt.savefig('%s' % filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_uvvis( spectra=None,
                linespectra=None,
                molecule=None, oscillator_strength=True,
                spc=False, band_width=0.3, band_width_slider=False,
                filename=None, 
                title='UV-Vis spectrum',
                xaxis_caption='Wavelength (nm)', yaxis_caption='Extinction coefficient (M$^{-1}$ cm$^{-1}$)', y2axis_caption='Oscillator strength $f$',
                labels=[],
                colors=[],
                normalize=False,
                shift=False, shiftby=None,
                plotstart=None, plotend=None,):
    if spectra is None:
        spectra = []
    else:
        spectra = copy.deepcopy(spectra)
    linespectra = copy.deepcopy(linespectra)
    if molecule is not None and oscillator_strength:
        if 'oscillator_strengths' in molecule.__dict__:
            linespectra = copy.deepcopy([spectrum(x=constants.hartree2nm(molecule.excitation_energies), y=molecule.oscillator_strengths)])
    
    normalize = normalize
    if normalize and yaxis_caption == 'Extinction coefficient (M$^{-1}$ cm$^{-1}$)':
        yaxis_caption='Normalized extinction'
    shift = shift
    shiftby = shiftby
    
    plotstart = plotstart
    plotend = plotend

    filename = filename

    title=title
    xaxis_caption = xaxis_caption
    yaxis_caption = yaxis_caption
    y2axis_caption = y2axis_caption
    labels = labels
    colors = colors
    spc_calls = []
    def spc_broaden(band_width):
        if spc:
            spc_calls.append(1)
            spc_spectrum = uvvis.spc(molecule=molecule,
                        #wavelengths_nm=np.arange(plotstart, plotend),
                        band_width=band_width)
            if len(spc_calls) > 1:
                spectra[-1] = spc_spectrum
            else:
                spectra.append(spc_spectrum)
        plot_spectra(spectra=spectra, linespectra=linespectra,
                    filename=filename, 
                    title=title,
                    xaxis_caption=xaxis_caption, yaxis_caption=yaxis_caption,
                    y2axis_caption=y2axis_caption,
                    labels=labels,
                    colors=colors,
                    normalize=normalize,
                    shift=shift, shiftby=shiftby,
                    plotstart=plotstart, plotend=plotend,)
    if spc and band_width_slider:
        import ipywidgets
        _ = ipywidgets.interact(spc_broaden,
                                band_width=ipywidgets.FloatSlider(
        value=band_width,
        min=0.01,
        max=0.5,
        step=0.01,
        description='width (eV):',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
         ))
    else:
        spc_broaden(band_width)

def plot_ir(spectra=None,
            linespectra=None,
            molecule=None,
            lorentzian=False,fwhm=30,band_width_slider=False,
            peak_highlight_slider=False,
            spectrum_range=np.arange(500,4001),
            filename=None,
            title='IR spectrum',
            xaxis_caption='Wavenumber (cm$^{-1}$)',yaxis_caption='Intensity (km/mol)',y2axis_caption='Intensity (km/mol)',
            labels=[],
            colors=[],
            normalize=False,
            scaling_factor=None,
            plotstart=None,plotend=None,):
    spectra = copy.deepcopy(spectra)
    linespectra = copy.deepcopy(linespectra)
    
    if normalize:
        yaxis_caption = 'Normalized intensity'

    lorentzian_calls = []
    def plot_ir_spectra(molecule=None,
                    spectra=None,linespectra=None,
                    filename=None,
                    title=None,
                    xaxis_caption='', yaxis_caption='', y2axis_caption='',
                    highlight=None,
                    labels=[],
                    colors=[],
                    normalize=False,
                    scaling_factor=False,
                    plotstart=None,plotend=None):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams['axes.linewidth'] = 2

        spectra = copy.deepcopy(spectra)
        linespectra = copy.deepcopy(linespectra)
        
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.85)
        lines = []

        if colors == []:
            colors = ['k', 'r'] #None for xx in spectra]
        if labels == []:
            labels = [None for xx in spectra]
        if normalize:
            for ii in range(len(spectra)):
                spectra[ii].normalize(method=normalize)
        
        if scaling_factor:
            print('Theoretical spectrum is scaled by %.3f' % scaling_factor)
            spectra[1].x *= scaling_factor 
            if linespectra is not None:
                linespectra[0].x *= scaling_factor
        for ii in range(len(spectra)):
            if ii < len(labels): label=labels[ii]
            else: label=None
            if ii < len(colors): color=colors[ii]
            else: color=None
            lines.append(ax.plot(spectra[ii].x, spectra[ii].y, color=color, label=label,
                            linewidth=None, marker=None, markersize=10, mfc='none')[0])

        if title != None: plt.title(title, fontsize=18)
        plt.xlabel(xaxis_caption, fontsize=18)
        plt.ylabel(yaxis_caption, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,3))#, labelsize=18)
        ax.xaxis.get_offset_text().set_fontsize(18)
        ax.yaxis.get_offset_text().set_fontsize(18)
        

        ymax1 = 0
        if linespectra is not None:
            ymax2 = max(linespectra[0].y)
            ymax1 = max([max(each.y) for each in spectra])
            ax2 = ax.twinx()
            ax2.set_ylabel(y2axis_caption,
                            fontsize=18, color='red')
            ax2.set_ylim(ymin=-0.05*ymax2, ymax=max(linespectra[0].y) * 1.1)
            ax2.tick_params(axis='both', colors='red')
            for ii in range(len(linespectra[0].x)):
                
                ax2.plot((linespectra[0].x[ii], linespectra[0].x[ii]),
                            (-0.05*ymax2, linespectra[0].y[ii]), 'r', linewidth=1)
            zed = [tick.set_fontsize(18)
                    for tick in ax2.yaxis.get_ticklabels()]
            if highlight is not None:
                ax2.plot((linespectra[0].x[highlight-1], linespectra[0].x[highlight-1]),
                            (-0.05*ymax2, linespectra[0].y[highlight-1]), 'b', linewidth=2)
                molecule.view(normal_mode=highlight-1,slider=False)
        if plotstart:
            plt.xlim(left=plotstart)
        if plotend:
            plt.xlim(right=plotend)

        if not all(label == None for label in labels):
            plt.legend(lines, [ll.get_label() for ll in lines],
                        frameon=False, fontsize=18, loc='best')

        ax.set_ylim(bottom=-ymax1*0.05)
        ax.invert_xaxis()
        if filename:
            plt.savefig('%s' % filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def lorentzian_broaden(fwhm_internal,highlight_internal):
        if lorentzian:
            lorentzian_calls.append(1)
            lorentzian_specturm = ir.lorentzian(molecule=molecule,fwhm=fwhm_internal,spectrum_range=spectrum_range)
            if len(lorentzian_calls) > 1:
                spectra[-1] = lorentzian_specturm
            else:
                spectra.append(lorentzian_specturm)
        plot_ir_spectra(molecule=molecule,
                     spectra=spectra,linespectra=linespectra,
                     filename=filename,
                     title=title,
                     xaxis_caption=xaxis_caption,yaxis_caption=yaxis_caption,y2axis_caption=y2axis_caption,
                     highlight=highlight_internal,
                     labels=labels,
                     colors=colors,
                     normalize=normalize,
                     scaling_factor=scaling_factor,
                     plotstart=plotstart,plotend=plotend
                     )
    if (lorentzian and band_width_slider) or peak_highlight_slider:
        import ipywidgets
        if lorentzian and band_width_slider:
            slider_for_band_width = ipywidgets.FloatSlider(
                value=fwhm,
                min=10,
                max=50,
                step=1,
                description='width/cm^-1:',
                disabled=False,
                continuous_update=True,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
            )
        else:
            slider_for_band_width = ipywidgets.fixed(fwhm)
        if peak_highlight_slider:
            
            slider_for_peak_highlight = ipywidgets.IntSlider(
                value=1,
                min=1,
                max=len(molecule.frequencies),
                step=1,
                description='peak',
                disabled=False,
                continuous_upate=True,
                orientation='horizontal',
                readout=True,
            )
        else:
            slider_for_peak_highlight = ipywidgets.fixed(None)
        _ = ipywidgets.interact(lorentzian_broaden,fwhm_internal=slider_for_band_width,highlight_internal=slider_for_peak_highlight)
    else:
        lorentzian_broaden(fwhm,None)


if __name__ == '__main__':
    print(__doc__)
