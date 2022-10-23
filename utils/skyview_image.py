"""
Author: Kevin Luke
Date: Created 23 OKT 2022
"""

import argparse
import astropy
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
import astropy.coordinates as coord

def skyview_image(name_loc, zoom_deg, surveyname=None):
    """
    Make a skyview search from a survey for a given region
    """
    
    if surveyname is None:
        # Query for images centered on target name
        hdu = SkyView.get_images(name_loc, survey='DSS',radius= 1*u.deg)[0][0]
    else:
        print(SkyView.survey_dict)
        print('Enter name of survey:')
        survey_name=str(input())
        # Query for images centered on target name
        hdu = SkyView.get_images(name_loc, survey=survey_name,
                                 radius= zoom_deg*u.deg)[0][0]
        
    # Tell matplotlib how to plot WCS axes
    wcs = WCS(hdu.header)
    fig = plt.figure(figsize=(20, 10), dpi=100)
    ax = fig.gca(projection=wcs)
    # Plot the image
    ax.imshow(hdu.data)
    ax.set(xlabel="RA", ylabel="Dec")
    fig.savefig('skyimage.jpeg')
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('name_loc', type=str,
                       help='Input the RA, DEC of the region. eg: 34.45,56,78')
    parser.add_argument('--surveyname',dest='surveyname',
                        default=None, action='store_true',
                       help='If this flag used then user can choose her own survey name for search')

    args = parser.parse_args()
    name_loc=args.name_loc
    surveyname=args.surveyname
    
    print('Enter the radius to be zoomed in degree')
    zoom_deg=float(input())
    
    #Skyview image query
    if surveyname is None:
        skyview_image(name_loc, zoom_deg,surveyname=None)
    else:
        skyview_image(name_loc, zoom_deg, surveyname=surveyname)
 
