#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:47:27 2022

Code to simulate a long slit observation using SKIRT output files emulating a 
specific setup. 

The initial cube is:
    PSF convolved (Moffat profile)
    Re-binned to a slit
    Perturbed with a noise array
    
Arguments passed to the function are:
    SKIRT output file 
    LEGA-C noise file

User specified parameters are:
    PSF: FWHM of the observed PSF to match (arcsec)
    SNe: S/N at 1 Re

The output cubes are stored as fits files named:
    original_filename_PSF_SNe.fits
    
@author: dbarrien
"""
import numpy as np

from astropy.io import fits
from astropy.constants import c
from astropy import units as u
from astropy.convolution import convolve_fft
from scipy import interpolate


#------------------------------ USEFUL METHODS --------------------------------

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



#------------------------------ MAIN FUNCTION ---------------------------------

def make_slit(inc, kind, ifs_folder, legac_file, PSF, SNe, z_sim, z_legac, 
              out_folder='', N=1e9):
    """
    
    
    """
    
    # Load data
    ifs_file = 'hx_legac_z08_slit_{}_{}.fits'.format(inc, kind)
    spec = fits.open(ifs_folder + ifs_file)
    wave = spec[1].data['GRID_POINTS'] * u.micron
    pix_to_arcsec = spec[0].header['CDELT1']    # arcsec per pix
    
    # Mask bad spaxels
    s1 = fits.open(ifs_folder+'hx_legac_z08_slit_{}_stats1.fits'.format(inc))[0].data.astype('float64')
    s2 = fits.open(ifs_folder+'hx_legac_z08_slit_{}_stats2.fits'.format(inc))[0].data.astype('float64')

    R = np.sqrt(s2/s1**2 - 1/N)
    mask = R.mean(axis=0) <= 0.1
    mask_1d = mask.prod(axis=0).astype('bool')   #Mask with pixels to use (along slit)
    
    
    # Unit change (Fnu to Flam)
    fnu = spec[0].data * u.MJy    # MJy
    flam = (fnu.T * c / wave**2).to('erg/s/cm2/Angstrom').T #erg/s/cm2/Angstrom
    
    # Convolve
    PSF_pix = PSF / pix_to_arcsec
    spec_conv_ = convolve_moffat_fft(flam.value[:,:,mask_1d], PSF_pix)
    spec_conv = np.zeros_like(spec[0].data)
    spec_conv[:,:,mask_1d] = spec_conv_
    print('##################################################################')
    print('#########        PSF CONVOLUTION: DONE                  ##########')
    print('##################################################################')
          
    # Make slit
    slit_2d = spec_conv[4:,0,:] + spec_conv[3:-1,1,:] + spec_conv[2:-2,2,:] + spec_conv[1:-3,3,:] + spec_conv[:-4,4,:]
    wave_2d = wave[2:-2].to('Angstrom')   # Some wavelentgh elemnts lost to slit mixing
    print('##################################################################')
    print('#########            SLIT MIXING: DONE                  ##########')
    print('##################################################################')

    # Add Noise
    legac_w1d = fits.open(legac_file)
    h = legac_w1d[0].header
    wave_legac = np.arange(h['CRVAL1'], h['CRVAL1'] + h['CD1_1'] * h['NAXIS1'], 
                           h['CD1_1'])
    
    wave_2d_rest = wave_2d.value / (1 + z_sim)
    wave_legac_rest = wave_legac / (1 + z_legac)
    noise_1d = 1/np.sqrt(legac_w1d[0].data[legac_w1d[0].data != 0]) 
    wave_noise = wave_legac_rest[legac_w1d[0].data != 0]     #Filter nans

    # Find range in simulated data inside legac range
    wmin = wave_noise.min()
    wmax = wave_noise.max()
    mask_wave = (wave_2d_rest > wmin) & (wave_2d_rest < wmax)
    wave_spec = wave_2d_rest[mask_wave]
    slit_2d_spec = slit_2d[mask_wave, :]

    # Interpolate noise to skirt wavelength
    f = interpolate.interp1d(wave_noise, noise_1d, kind='nearest')
    noise_1d_interp = f(wave_spec)

    # Compute noise normalization
    # Combined 5 central pixels should have fixed SN
    N_sigma = np.median(slit_2d_spec[:,11:17].sum(axis=1) / (SNe * noise_1d_interp))

    # Apply 1d noise to all slit
    slit_2d_noise = np.zeros_like(slit_2d_spec)
    for i in range(slit_2d_noise.shape[-1]):
        slit_2d_noise[:,i] = np.random.normal(slit_2d_spec[:,i], noise_1d_interp * N_sigma)#noise[:,i])
    
    print('##################################################################')
    print('#########              ADD NOISE: DONE                  ##########')
    print('##################################################################')
          
    # Create new fits file
    new_fits = fits.PrimaryHDU(slit_2d_noise, header=spec[0].header)
    
    # Update sizes and add noise value
    c2 = fits.Column(name='noise', array=N_sigma * noise_1d_interp, format='E')
    c1 = fits.Column(name='wave', array=wave_spec, format='E')
    new_noise = fits.BinTableHDU.from_columns([c1, c2])
    new_fits.header['PSF'] = (PSF, 'psf FWHM [arcsec]')
    
    # Add table for mask
    c3 = fits.Column(name='r-mask', array=mask_1d, format='E')
    new_mask = fits.BinTableHDU.from_columns([c3])
    
    
    # Save file
    new_name = ifs_file.split('.')[0] + '_{}_{}.fits'.format(PSF, SNe)
    hdul = fits.HDUList([new_fits, new_noise, new_mask])
    hdul.writeto(out_folder+new_name)
    
    return None


def make_slit_simple(inc, kind, ifs_folder, z_sim, out_folder='', N=1e9):
    """
    
    
    """
    
    # Load data
    ifs_file = 'hx_legac_z08_slit_{}_{}.fits'.format(inc, kind)
    spec = fits.open(ifs_folder + ifs_file)
    wave = spec[1].data['GRID_POINTS'] * u.micron
    
    # Mask bad spaxels
    s1 = fits.open(ifs_folder+'hx_legac_z08_slit_{}_stats1.fits'.format(inc))[0].data.astype('float64')
    s2 = fits.open(ifs_folder+'hx_legac_z08_slit_{}_stats2.fits'.format(inc))[0].data.astype('float64')

    R = np.sqrt(s2/s1**2 - 1/N)
    mask = R.mean(axis=0) <= 0.1
    mask_1d = mask.prod(axis=0).astype('bool')   #Mask with pixels to use (along slit)
    
    # Unit change (Fnu to Flam)
    fnu = spec[0].data * u.MJy    # MJy
    flam = (fnu.T * c / wave**2).to('erg/s/cm2/Angstrom').T #erg/s/cm2/Angstrom

          
    # Make slit
    slit_2d = flam.value.sum(axis=1)[2:-2,:]
    wave_2d = wave[2:-2].to('Angstrom')   # Some wavelentgh elemnts lost to slit mixing
    print('##################################################################')
    print('#########            SLIT MIXING: DONE                  ##########')
    print('##################################################################')
          
    # Create new fits file
    new_fits = fits.PrimaryHDU(slit_2d, header=spec[0].header)
    
    # Update sizes and add noise value
    c2 = fits.Column(name='noise', array=np.zeros_like(slit_2d[:,0]), format='E')
    c1 = fits.Column(name='wave', array=wave_2d.value / (1 + z_sim), format='E')
    new_noise = fits.BinTableHDU.from_columns([c1, c2])
    
    # Add table for mask
    c3 = fits.Column(name='r-mask', array=mask_1d, format='E')
    new_mask = fits.BinTableHDU.from_columns([c3])
    
    
    # Save file
    new_name = ifs_file.split('.')[0] + '_2d.fits'
    hdul = fits.HDUList([new_fits, new_noise, new_mask])
    hdul.writeto(out_folder+new_name)
    
    return None





#------------------------------- PARAMETERS -----------------------------------
PSF = 0    # arcsec
SNe = 20
kind = 'transparent'

ifs_folder = '/Users/dbarrien/Phd/Auriga/velocity_maps/skirt_results/high_z/h6_z08_1e9/'


id_legac = 123292
legac_mask = 8
legac_dir = '/Users/dbarrien/Dropbox/legac_team_share/spectra/'
legac_file = legac_dir + 'legac_M{}_v3.11_wht1d_{}.fits'.format(legac_mask, id_legac)
z_sim = 0.8
z_legac = 0.803533


out_folder = '/Users/dbarrien/Phd/Auriga/synth-spec/legac_files/'
inclinations = ['25', '45', '60', '60_20', '80', '85', 'xy', 'xz']  #

#inc = '60'    # '60', '60_20' '80' '85' 'xy' 'xz'

for inc in inclinations:
    if PSF != 0:
        make_slit(inc, kind, ifs_folder, legac_file, PSF, SNe, z_sim, z_legac, 
              out_folder=out_folder)
        
    if PSF == 0:
        make_slit_simple(inc, kind, ifs_folder, z_sim, out_folder=out_folder)




#for inc in inclinations:
#    ifs_file = 'hx_legac_z08_slit_{}_total.fits'.format(inc)
#    









