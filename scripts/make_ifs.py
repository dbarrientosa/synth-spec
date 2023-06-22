#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:35:12 2022

Code to simulate an IFS observation using SKIRT output files emulating a 
specific setup. 

The initial cube is:
    PSF convolved (Moffat profile)
    Re-binned if needed
    Perturbed with a constant noise value
    
Arguments passed to the function are:
    SKIRT output file 
    Inclination of simulated galaxy
    Effective radius in pixels     
    

User specified parameters are:
    PSF: FWHM of the observed PSF to match (arcsec)
    NBINS: number bins for rebinning if needed 
    SNe: S/N at 1 Re

The output cubes are stored as fits files named:
    original_filename_PSF_NBINS_SNe.fits

@author: dbarrien
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

from astropy.io import fits
from astropy.convolution import convolve_fft

from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler, SplineInterpolatedResampler
from scipy.ndimage import gaussian_filter1d

from astropy import units as u
from astropy.constants import c

import ppxf.ppxf_util as util
#------------------------------ USEFUL METHODS --------------------------------

# Moffat profile
def moffat(x, y, FWHM, beta=4.765):
    """
    Implementation of moffat profile in 2D.
    
    x, y [pix]: arrays with position to calculate profile
    FWHM [pix]: full width at half maximum of the profile
    beta [pix?]: seeing dependent parameter
    
    returns ndarray with profile values at (x,y)
    """
    alpha = FWHM / (2 * np.sqrt(np.power(2, 1/beta) - 1))
    return (beta - 1) / (np.pi * alpha ** 2) * (1 + ((x**2 + y**2) / alpha ** 2)) ** -beta

# Convolution
def convolve_moffat_fft(data, FWHM, **kwargs):
    """
    Convolution of moffat profile for a 3D array.
    
    data : data cube with fluxes to covolve. shape must be [wavelentgh, x,y]
    FWHM [pix] : full width at half maximum of the profile
    returns data cube convolved with the moffat profile.
    """
    x = np.arange(-100, 101)
    x, y = np.meshgrid(x, x)
    moffat_kernel = moffat(x, y, FWHM, **kwargs)

    wsize = data.shape[0]
    data_convovled = np.zeros_like(data)

    for i in range(wsize):
        data_convovled[i,:,:] = convolve_fft(data[i,:,:], moffat_kernel)

    return data_convovled

# Spatial binning for a 3D cube 
def binning_3D(array, binning):
    """
    Spatial binning of an IFS cube (3 dim).
    
    array: data cube (wave, x, y) size (N, M, M)
    int binning: number of pixels to merge together
    
    returns binned datacube of size (N, M/binning, M/binning)
    """
    
    # Assuming the shape is (wave, x, y)
    size = array.shape[-1]
    wsize = array.shape[0]
    nbins = int(size / binning)

    return array.reshape(-1, size, nbins, binning).sum(axis=3).reshape(wsize, 
                        nbins, binning, -1).sum(axis=2)

# Spatial binning for a 2D array
def binning_2D(array, binning, bool=True):
    """
    Spatial binning of an image (2 dim).
    
    array: data cube (x, y) size (M, M)
    int binning: number of pixels to merge together
    
    returns binned datacube of size (M/binning, M/binning)
    """
    # Assuming the shape is (wave, x, y)
    size = array.shape[-1]
    nbins = int(size / binning)

    binned = array.reshape(-1, nbins, binning).sum(axis=2).reshape(nbins, 
                          binning, -1).sum(axis=1)

    if bool:
        binned = binned > 0

    return binned


# Linear interpolation 
def log_to_linear(lin_wave, log_wave, flux, sampler):
    """
    lin_wave: new linear wavelength 
    log_wave: original log spaced wavelength 
    flux: original flux
    sampler: sampler object (from specutils.manipulation)
    
    returns spectrum at new linear wavelength
    """
    input_spec = Spectrum1D(spectral_axis=log_wave * u.AA, flux=flux * u.Lsun / u.AA)
    
    return sampler(input_spec, lin_wave).flux.value


#------------------------------ MAIN FUNCTION ---------------------------------
    
def make_ifs(halonum, ifu_folder, inc, inc_num, Re, PSF=1, NBINS=1, SNe=5, 
             out_folder='', SAMI=False, kind='total', N = 5e9, Rmax=0.1, 
             spec_res=1808., LSF=2.65):
    """
    Transfrom SKIRT output data cube into IFS from a specific survey. 
    This involves PSF convolution, binning and noise adittion. 
    
    halonum (str): number of halo
    ifu_file (str): file name of the skirt output datacube
    ifu_folder (str): path of the skirt output files
    inc (str): inclination in file name
    inc_num (float): inclination
    Re (float): effective radius of the galaxy in kpc
    PSF (float = 1): PSF FWHM of the survey in arcsec, if (=0) convolution is not performed
    NBINS (int = 1): Number of bins to merge, if (=0) binning is not performed
    SNe (float = 5): SN of the annuli at 1 Re, if (=0) noise addition is minimal
    out_folder (''): path of the folder to write output files
    SAMI (bool = False): if SAMI=True, pixel size are taken from a different file
    kind (str = 'total'): source for SKIRT (total: w/ extinction, transparent: only stars)
    N (float = 5e9): Number of photon packets 
    Rmax (float = 0.1) : minimum value for R statistic
    spec_res (float = 1808.) : spectral resolution of desired mock obs, default SAMI blue
    LSF (float = 2.65 : FWHM of LSF of obs, default SAMI blue
    """
    
    ############################ LOAD DATA ####################################
    ifu_file = 'hx_fsps_hr_IFU_v_dust_IFU_{}_{}.fits'.format(inc, kind)

    #ifu_file = 'hx_harmoni_z3_slit_{}_{}.fits'.format(inc, kind)
    
    ifu = fits.open(ifu_folder + ifu_file)
    data_cube = ifu[0].data
    pix_to_arcsec = ifu[0].header['CDELT1']  # Arcsec
    pix_to_kpc = 0.5   # 0.5
    wave = ifu[1].data['GRID_POINTS'] * 1e4  # Angstrom
    
    # For SAMI only
    if SAMI==True:
        sami_size = fits.open('../aux_files/size_sami_total.fits')
        pix_to_arcsec = sami_size[0].header['CDELT1']
    sidex = ifu[0].header['NAXIS1']
    sidey = ifu[0].header['NAXIS2'] 
    Re_pix = Re / pix_to_kpc
    extent_og = (-sidex/2*pix_to_kpc, sidex/2*pix_to_kpc, 
                 -sidey/2*pix_to_kpc, sidey/2*pix_to_kpc)
    # R statistic
    # Calculate R and VOV

    s1 = fits.open(ifu_folder + 'hx_fsps_hr_IFU_v_dust_IFU_{}_stats1.fits'.format(inc))[0].data.astype('float64')
    s2 = fits.open(ifu_folder + 'hx_fsps_hr_IFU_v_dust_IFU_{}_stats2.fits'.format(inc))[0].data.astype('float64')

    #s1 = fits.open(ifu_folder + 'hx_harmoni_z3_slit_{}_stats1.fits'.format(inc))[0].data.astype('float64')
    #s2 = fits.open(ifu_folder + 'hx_harmoni_z3_slit_{}_stats2.fits'.format(inc))[0].data.astype('float64')

    R = np.sqrt(s2/s1**2 - 1/N)
    mask = R.mean(axis=0) <= Rmax#0.1
    
    image = data_cube.sum(axis=0)
    image[image==0] = np.nan   #Replace pixels with no flux with nan
    
    
    ############################ INSTRUMENTALS ################################
    # Spectral binning
    if spec_res != 0:
        
        # Get new wavelength range
        spec_int = data_cube[:,mask].sum(axis=(1))
        velscale_ifs = c.to('km/s').value / spec_res
        
        spec_int_bin, wave_bin, _ = util.log_rebin([wave[0], wave[-1]], 
                                                   spec_int, 
                                                   velscale=velscale_ifs)

        new_wave = np.exp(wave_bin)
        
        # Convolve w/ LSF
        fsps_lsf = np.loadtxt('../aux_files/LSF-Config_fsps.txt') # FSPS Spec res (SKIRT)
        fwhm_gal = np.interp(new_wave, fsps_lsf[:,0], fsps_lsf[:,1])
        fwhm_dif = np.sqrt(LSF**2 - fwhm_gal**2)

        dw_new = np.exp(wave_bin[1:]) - np.exp(wave_bin[:-1])
        sigma_new = fwhm_dif/2.355/np.append(dw_new, dw_new[-1])     # Sigma difference in pixels
        
        
        # Re-bin whole cube
        samp = SplineInterpolatedResampler()
        #samp = FluxConservingResampler()

        spec_lsf = np.full((len(new_wave), sidex, sidey), np.nan)
        for i in range(sidex):
            for j in range(sidey):
                if mask[i,j]:
                    input_spec = Spectrum1D(spectral_axis=wave*u.AA, 
                                            flux=data_cube[:,i,j]*u.MJy) 
                    spec_lsf[:,i,j] = gaussian_filter1d(samp(input_spec, 
                            new_wave*u.AA).flux.value, np.mean(sigma_new))
        
                    
        data_cube = spec_lsf  # Replace original cube with convolved one
        wave = new_wave
        print('##################################################################')
        print('#########        LSF CONVOLUTION: DONE                  ##########')
        print('##################################################################')
        
    
    
    # PSF convolution
    
    if PSF != 0:
        PSF_pix = PSF / pix_to_arcsec
        data_cube = convolve_moffat_fft(data_cube, PSF_pix) 

        print('##################################################################')
        print('#########        PSF CONVOLUTION: DONE                  ##########')
        print('##################################################################')
    
          
    # Binning
    if NBINS != 0:
        data_cube = binning_3D(data_cube, NBINS)
        sidex /= NBINS
        sidey /= NBINS
        Re_pix /= NBINS
        pix_to_kpc *= NBINS
        mask = binning_2D(mask, NBINS)
        sidex = int(sidex)
        sidey = int(sidey)
        print('##################################################################')
        print('#########              REBINNING: DONE                  ##########')
        print('##################################################################')
          
          
    ############################ NOISE ADDITION ############################### 
    
    x = np.linspace(-sidex/2, sidex/2, int(sidex))
    y = np.linspace(-sidey/2, sidey/2, int(sidey))
    
    if SNe != 0:
        # Noise addittion
        xx, yy = np.meshgrid(x, y/np.cos(inc_num * np.pi/180))
        r = np.sqrt(xx**2 + yy**2)
        r_in = Re_pix * 0.8
        r_out = Re_pix * 1.2
        # Select pixels around Re
        re_ellipse = np.full_like(data_cube.sum(axis=0), False, dtype='bool')  
        re_ellipse[(r > r_in) & (r <= r_out)] = True
        flux_re = data_cube.mean(axis=0)[re_ellipse].mean()   # Mean flux around Re
        sigma_flux = flux_re / SNe
        # Preturb datacube
        signal = np.random.normal(data_cube, sigma_flux)
        
    if SNe == 0:
        signal = data_cube
        sigma_flux = 1e-5    # Set a very low value so that SN condition is met everywhere
        re_ellipse = np.full_like(data_cube.sum(axis=0), False, dtype='bool')
        
    print('##################################################################')
    print('#########              ADD NOISE: DONE                  ##########')
    print('##################################################################')
  

          
    ################################# SAVE ####################################      
    
    # Create new fits file
    new_fits = fits.PrimaryHDU(signal, header=ifu[0].header)
    
    # Update sizes and add noise value
    new_fits.header['CDELT1'] = pix_to_arcsec * NBINS
    new_fits.header['CDELT2'] = pix_to_arcsec * NBINS
    new_fits.header['NAXIS1'] = sidex
    new_fits.header['NAXIS2'] = sidey
    
    new_fits.header['NOISE'] = (sigma_flux, 'flux noise')
    new_fits.header['PSF'] = (PSF, 'psf FWHM [arcsec]')
    
    new_mask = fits.ImageHDU(mask.astype('int'))
    
    wave = fits.Column(name='GRID_POINTS', array=wave, format='E')
    table = fits.BinTableHDU.from_columns([wave])
    
    # Save file
    new_name = ifu_file.split('.')[0] + '_{}_{}_{}_0{}.fits'.format(PSF,NBINS,
                             SNe,int(Rmax*10))
    hdul = fits.HDUList([new_fits, table, new_mask])
    hdul.writeto(out_folder+new_name)
    
    ################################# PLOT #################################### 
    
    a, b = np.unravel_index(signal.sum(axis=0).argmax(), (sidex,sidey))
    vmin, vmax = np.nanpercentile(image.flatten(), [0.5, 99.5])
     
    fig, ax = plt.subplots(figsize=(12,5), ncols=3, sharex=True, sharey=True)
    
    # Original 
    ax[0].imshow(image, origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax), 
      extent=extent_og)
    ax[0].plot(x[a], y[b], 'k*')
    # SN_map with contours
    SN_map = signal.mean(axis=0) / sigma_flux
    ax[1].imshow(SN_map, extent=extent_og)
    ax[1].contour(SN_map, levels=[3, 5], origin='lower', extent=extent_og)
    ax[1].imshow(re_ellipse, extent=extent_og, alpha=0.3)
    
    # Resulting flux with Rmask
    ax[2].imshow(signal.sum(axis=0)/4, norm=LogNorm(vmin=vmin, vmax=vmax), 
      origin='lower', extent=extent_og)
    ax[2].contour(mask, origin='lower', extent=extent_og)
    
    # Change inclination string with _ to -
    if len(inc) > 3:
        incs = inc.split('_')
        inc = incs[0] + '-' + incs[1]
    
    for j in range(3):
        e1 = Ellipse(xy=(0,0), width=2*Re, height=2*Re*np.cos(inc_num * np.pi/180), 
                     fill=False, ls='dashed')
        e2 = Ellipse(xy=(0,0), width=4*Re, height=4*Re*np.cos(inc_num * np.pi/180), 
                     fill=False, ls='dashed')
        ax[j].add_artist(e1)
        ax[j].add_artist(e2)
        
    title1 = 'Halo {} {} \n i={}'.format(halonum, kind, inc)
    title2 = 'PSF={}, BIN={}, SN={}, Rmax={}'.format(PSF,NBINS,SNe, Rmax)
    fig.suptitle(title1+r'$^{\circ}$, '+title2 , fontsize=15, y=0.92)
    fig.tight_layout()
    
    plt.savefig(out_folder+'h{}_{}_{}_{}_{}_{}_0{}_flux.png'.format(halonum,
                kind,inc,PSF,NBINS,SNe,int(Rmax*10)))
    
    
    
    return None
    
    
#------------------------------- PARAMETERS -----------------------------------
    
psf = 0       #0.01#2#4    # Arcsec
rebin = 0     #2
sn = 0        #5
Rmax = 0.1    #threshold for R statistic
kind = 'total'
spec_res = 1808.      # 1808.
#halonum = 1          # 1, 3, 6, 7, 13, 14, 17, 27, 28, 30 




# Example: run multiple halos at multiple inclinations 
halos = [30]#[1, 3, 6, 14, 27, 28]
for halonum in halos:
    print('##################################################################')
    print('#########              HALO: {}                         ##########'.format(halonum))
    print('##################################################################')
  

    ifu_folder = 'FOLDER_WITH_SKIRT_OUTPUT/'  
    au_table = pd.read_csv('../aux_files/auriga_table.txt', delim_whitespace=True)
    Re = 1.68 * au_table['Rd'].iloc[halonum - 1]

    inclinations = ['25', '45', '60', '72_0', '72', '72_90']  
    inclinations_num = [25.84, 45.57, 60.0, 72.54, 72.54, 72.54]

    out_folder = 'FOLDER_TO_STORE_NEW_DATA/'


    for j, inc in enumerate(inclinations):
        inc_num = inclinations_num[j]
        make_ifs(halonum, ifu_folder, inc, inc_num, Re, PSF=psf, NBINS=rebin, SNe=sn, 
             out_folder=out_folder, SAMI=True, kind=kind, Rmax=Rmax, 
             spec_res=spec_res, LSF=2.65)
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



