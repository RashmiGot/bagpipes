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
    filt_list : array of paths to filter files, each element is a string

    Returns
    -------
    eff_wavs : effective wavelengths of the input filters, format=numpy array
    """

    # redshift
    z = obj.model_components["redshift"]
    
    # prism resolution curve interpolated onto wavelength grid
    R_curve_interp = obj.galaxy.R_curve_interp

    # emission line names
    lw, lr = utils.get_line_wavelengths() # list of all line names and wavelengths
    line_names_all = obj.galaxy.msa_line_components

    _A = [] # initialise design matrix

    for li in line_names_all:

        if li not in lw:
            continue

        lwi = lw[li][0] * (1 + z)

        for i, (lwi0, lri) in enumerate(zip(lw[li], lr[li])):
            lwi = lwi0 * (1 + z) / 1.0e4

            line_i = emission_line_templ(
                                spec_wobs = obj.galaxy.spectrum[:, 0]/10000,
                                spec_R_fwhm = R_curve_interp,
                                line_um = lwi,
                                line_flux = lri / np.sum(lr[li]), 
                                velocity_sigma = obj.fit_instructions["veldisp"],
                                )

            if i == 0:
                line_0 = line_i
            else:
                line_0 += line_i

        _A.append(line_0 / 1.0e4) # append line to design matrix

    _A = np.vstack(_A)
        
    return (_A)


def msa_line_model(obj, model, noise=None):
    """
    Makes line models with msaexp
    """

    # design matrix of line components
    A = add_msaexp_emission_components(obj)

    # !!!!! UPDATE !!!!!
    spec_fluxes = obj.galaxy.spectrum[:,1]

    if noise is None:
        spec_sivar = 1/obj.galaxy.spectrum[:,2]
    else:
        spec_sivar = np.sqrt(noise.inv_var)

    model_diff = spec_fluxes - model # differences between spectrum and bagpipes model

    lsq_coeffs = np.linalg.lstsq((A*spec_sivar).T, model_diff*spec_sivar, rcond=None)

    msa_model = (A.T).dot(lsq_coeffs[0])

    return msa_model, lsq_coeffs

