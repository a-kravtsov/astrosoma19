try:
    import numpy as np
except ImportError:
    print("numpy is required to run this code! but does not seem to be available...")


try:
    from astropy.io import fits
except ImportError:
    print("astropy.io is required to read fits file! but does not seem to be available...")
    
    

def read_meert_catalog(datadir, phot_type=None):
    """
    Loader for the Meert et al. 2015 catalog of improved photometric measurements
    for galaxies in the SDSS DR7 main galaxy catalog 
    
    Parameters: 
    -----------
        datadir: string; path to the UPenn data directory
        phot_type: integer (1 to 5) corresponding to the photometry model fit type from the catalog
        1=best fit, 2=deVaucouleurs, 3=Sersic, 4=DeVExp, 5=SerExp
    
    Returns:
    --------
    
    arrays containing different catalog information in r and g bands 
    """

    if (phot_type < 1) or (phot_type > 5):
        raise Exception('unsupported type of Meert et al. photometry: %d, choose number between 1 and 5')

    datameertnonpar  = datadir + 'UPenn_PhotDec_nonParam_rband.fits'
    datameertnonparg = datadir + 'UPenn_PhotDec_nonParam_gband.fits'
    datameert        = datadir + 'UPenn_PhotDec_Models_rband.fits'
    datasdss         = datadir + 'UPenn_PhotDec_CAST.fits'
    datasdssmodels   = datadir + 'UPenn_PhotDec_CASTmodels.fits'
    datameertg       = datadir + 'UPenn_PhotDec_Models_gband.fits'
    datamorph        = datadir + 'UPenn_PhotDec_H2011.fits' # morphology probabilities from Huertas-Company et al. 2011

    # mdata tables: 1=best fit, 2=deVaucouleurs, 3=Sersic, 4=DeVExp, 5=SerExp
    mdata    = fits.open(datameert)[phot_type].data
    mdatag   = fits.open(datameertg)[phot_type].data
    mnpdata  = fits.open(datameertnonpar)[1].data
    mnpdatag = fits.open(datameertnonparg)[1].data
    sdata    = fits.open(datasdss)[1].data
    phot_r   = fits.open(datasdssmodels)[1].data
    morph    = fits.open(datamorph)[1].data

    # eliminate galaxies with bad photometry
    fflag = mdata['finalflag']
    print("%d galaxies in Meert et al. sample initially"%np.size(fflag))

    def isset(flag, bit):
        """Return True if the specified bit is set in the given bit mask"""
        return (flag & (1 << bit)) != 0
        
    # use minimal quality cuts and flags recommended by Alan Meert
    igood = [(phot_r['petroMag'] > 0.) & (phot_r['petroMag'] < 100.) & (mnpdata['kcorr'] > 0) &
             (mdata['m_tot'] > 0) & (mdata['m_tot'] < 100) &
             (isset(fflag, 1) | isset(fflag, 4) | isset(fflag, 10) | isset(fflag, 14))]

    igood = tuple(igood)
    sdata = sdata[igood]; phot_r = phot_r[igood]; mdata = mdata[igood]
    mnpdata = mnpdata[igood]; mdatag = mdatag[igood]; mnpdatag = mnpdatag[igood]; morph = morph[igood]

    return sdata, mdata, mnpdata, phot_r, mdatag, mnpdatag, morph

