"""
Author: Kevin Luke
Date: Created 23 OKT 2022
"""

import argparse
import astropy
import matplotlib.pyplot as plt
import astropy.units as u
from astroquery.simbad import Simbad
import astropy.coordinates as coord

def simbad_query_for_skyview_image(zoom_deg,name_loc):
    """
    For the region entered do a simbad query to list all objects.
    """
    #Conver the string of RA AND DEC into suitable floats
    name_loc.split(',')
    RA=float(name_loc[0])
    DEC=float(name_loc[1])
    #Feed them into the location variable for the coordinates generation
    location=coord.SkyCoord(ra=RA, dec=DEC,unit=(u.deg, u.deg), frame='icrs')
    #The query and then the table is displayed
    result_table = Simbad.query_region(location,radius=zoom_deg*u.deg)
    print(result_table)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('name_loc', type=str,
                       help='Input the RA, DEC of the region. eg: 34.45,56,78')
    parser.add_argument('--simbad_search',dest='simbad_search',
                        default=None, action='store_true',
                       help='If this flag used then user can do a simbad search of the given region')
    
    args = parser.parse_args()
    name_loc=args.name_loc
    simbad_search=args.simbad_search
    
    print('Enter the radius to be zoomed in degree')
    zoom_deg=float(input())
    if simbad_search is not None:
        simbad_query_for_skyview_image(zoom_deg,name_loc)
