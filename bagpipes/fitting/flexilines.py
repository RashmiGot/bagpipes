"""
Adds flexibe line component fitting to Bagpipes
"""

import msaexp
import grizli
import numpy as np

from grizli import utils
from msaexp.resample_numba import sample_gaussian_line_numba as SAMPLE_LINE_FUNC

def em_line_list(halpha_prism=['Ha+NII']):

    """ List of emission lines """
    
    hlines = ["Hb", "Hg", "Hd"]
    hlines += halpha_prism + ["NeIII-3968"]
    oiii = ["OIII-4959", "OIII-5007"]
    hene = ["HeII-4687", "NeIII-3867", "HeI-3889"]
    o4363 = ["OIII-4363"]
    oii_7320 = ["OII-7325"]
    sii = ['SII']
    list = [
        *hlines,
        *oiii,
        *o4363,
        "OII",
        *hene,
        *sii,
        *oii_7320,
        "ArIII-7138",
        "ArIII-7753",
        "SIII-9068",
        "SIII-9531",
        "OI-6302",
        "PaD",
        "PaG",
        "PaB",
        "PaA",
        "HeI-1083",
        "CI-9850",
        # "SiVI-19634",
        "BrA",
        "BrB",
        "BrG",
        "BrD",
        "BrE",
        "BrF",
        "PfB",
        "PfG",
        "PfD",
        "PfE",
        "Pa8",
        "Pa9",
        "Pa10",
        "HeI-5877",
        # *fuv,
        "CIII-1906",
        "NIII-1750",
        "Lya",
        "MgII",
        "NeV-3346",
        "NeVI-3426",
        "HeI-7065",
        "HeI-8446",
        # *extra,
        # *extra_lines,
        ]
    
    return list

def emission_line_templ(
        spec_wobs,
        spec_R_fwhm,
        line_um,
        line_flux=1.0,
        velocity_sigma=100,
        ):
    """
    Makes an emission line template centered at a given wavelength
    Adapted from msaexp function "fast_emission_line"; https://github.com/gbrammer/msaexp/blob/b852919bafbc2f20c15e24075c2be71e03e0d397/msaexp/spectrum.py#L500
    
    Parameters
    ----------
    spec_wobs : array-like
        Spectrum wavelengths

    spec_R_fwhm : array-like
        Spectral resolution `wave/d(wave)`, FWHM

    line_um : float
        Emission line central wavelength, in microns

    line_flux : float
        Normalization of the line

    velocity_sigma : float
        Kinematic velocity width, km/s

    Returns
    -------
    res : array-like
    Gaussian emission line sampled at the spectrum wavelengths
    """

    res = SAMPLE_LINE_FUNC(
        spec_wobs = spec_wobs,
        spec_R_fwhm = spec_R_fwhm,
        line_um = line_um,
        line_flux = line_flux,
        velocity_sigma = velocity_sigma,
    )

    return res

# def add_msaexp_emission_components(obj, line_names_all=None):
def add_msaexp_emission_components(obj):
    """
    Adds single/multiple emission line components from msaexp
    
    Parameters
    ----------
    obj : fitted_model object in bagpipes; format=object

    Returns
    -------
    _A : design matrix of line components; format=2D numpy array
    """

    # redshift
    z = obj.model_components["redshift"]
    
    # prism resolution curve interpolated onto wavelength grid
    R_curve_interp = obj.galaxy.R_curve_interp

    # emission line names
    lw, lr = utils.get_line_wavelengths() # list of all line names and wavelengths
    line_names_all = obj.galaxy.msa_line_components

    _A = [] # initialise design matrix
    skipped_lines = []

    for li in line_names_all:

        if li not in lw:
            skipped_lines.append(li)
            continue

        # handle multiple components per line (e.g., doublets)
        line_total = None
        
        for i, (lwi0, lri) in enumerate(zip(lw[li], lr[li])):
            # convert rest wavelength to observed wavelength in microns
            lwi = lwi0 * (1 + z) / 1.0e4

            line_i = emission_line_templ(
                                spec_wobs = obj.galaxy.spectrum[:, 0]/10000,
                                spec_R_fwhm = R_curve_interp,
                                line_um = lwi,
                                line_flux = lri / np.sum(lr[li]),  # normalize component flux
                                velocity_sigma = obj.fit_instructions["veldisp"],
                                )

            if line_total is None:
                line_total = line_i
            else:
                line_total += line_i

        _A.append(line_total / 1.0e4) 

    if len(skipped_lines) > 0:
        print(f"Warning: Skipped {len(skipped_lines)} lines not in grizli database: {skipped_lines}")

    if len(_A) == 0:
        raise ValueError("No valid emission lines found for fitting")
        
    _A = np.vstack(_A)
        
    return (_A)


def msa_line_model(obj, model, noise=None):
    """
    Makes line models with msaexp

    Parameters
    ----------
    obj : fitted_model object in bagpipes; format=object
    model : bagpipes model spectrum; format=numpy array
    noise : noise object in bagpipes; format=object

    Returns
    -------
    msa_model : flexible line model; format=numpy array
    lsq_coeffs : least-squares coefficients for line components; format=numpy array
    """

    # design matrix of line components
    A = add_msaexp_emission_components(obj)

    # spectrum data
    spec_fluxes = obj.galaxy.spectrum[:,1]

    # inverse variance weighting
    if noise is None:
        spec_ivar = 1.0 / obj.galaxy.spectrum[:,2]**2  # convert errors to inverse variance
    else:
        spec_ivar = noise.inv_var

    # residuals between observed spectrum and bagpipes model
    model_diff = spec_fluxes - model 

    # weighted least squares fitting
    try:
        # weight the design matrix and residuals
        A_weighted = (A * spec_ivar).T
        residuals_weighted = model_diff * spec_ivar

        # solve for line coefficients
        lsq_coeffs = np.linalg.lstsq(A_weighted, residuals_weighted, rcond=None)
            
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error in line fitting: {e}")
        # return zero model if fitting fails
        msa_model = np.zeros_like(model)
        lsq_coeffs = [np.zeros(A.shape[0]), None, None, None]
        return msa_model, lsq_coeffs

    # construct the line model from fitted coefficients
    msa_model = (A.T).dot(lsq_coeffs[0])

    return msa_model, lsq_coeffs


def calc_flux_in_filter(spec_wavs, spec_flux, filter_int, filt_norm, valid):
    """
    Calculate flux in a filter given a spectrum and filter throughput
    
    Parameters
    ----------
    spec_wavs : wavelengths of the spectrum; format=numpy array
    spec_flux : flux values of the spectrum; format=numpy array
    filter_int : interpolated filter throughput array; format=numpy array
    filt_norm : normalization value for the filter; format=float
    valid : boolean for valid spectrum pixels; format=numpy array

    Returns
    -------
    flux : flux in the filter; format=float
    """

    # trapezoid rule steps
    trapz_dx = utils.trapz_dx(spec_wavs)

    flux = (
            (filter_int / filt_norm * trapz_dx * spec_flux / spec_wavs)[
                valid
            ]
        ).sum()
    
    return flux


def calc_phot_fluxes(spec_wavs, spec_flux, filter_int_array, filt_norm_array, valid):
    """
    Calculate fluxes in multiple filters given a spectrum and filter throughputs

    Parameters 
    ----------
    spec_wavs : wavelengths of the spectrum; format=numpy array
    spec_flux : flux values of the spectrum; format=numpy array
    filter_int_array : array of filter throughput arrays; format=2D numpy array
    filt_norm_array : array of filter normalization values; format=1D numpy array
    valid : boolean array for valid spectrum pixels; format=numpy array

    Returns
    -------
    phot : list of fluxes in each filter; format=list of floats
    """

    phot = [calc_flux_in_filter(spec_wavs, spec_flux, filter_int, filt_norm, valid) for filter_int, filt_norm in zip(filter_int_array, filt_norm_array)]
    
    return phot
