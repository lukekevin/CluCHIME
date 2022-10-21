import argparse
import astropy
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('name_loc', type=str)
    parser.add_argument('--surveyname',dest='surveyname',
                        default=None, action='store_true')

    args = parser.parse_args()
    
    name_loc=args.name_loc
    surveyname=args.surveyname
    
    if surveyname is None:
        # Query for images centered on target name
        hdu = SkyView.get_images(name_loc, survey='DSS')[0][0]
    else:
        print(SkyView.survey_dict)
        print('Enter name of survey:')
        survey_name=str(input())
        # Query for images centered on target name
        hdu = SkyView.get_images(name_loc, survey=survey_name)[0][0]
        
    # Tell matplotlib how to plot WCS axes
    wcs = WCS(hdu.header)
    fig = plt.figure(figsize=(20, 10), dpi=100)
    ax = fig.gca(projection=wcs)
    # Plot the image
    ax.imshow(hdu.data)
    ax.set(xlabel="RA", ylabel="Dec")
    fig.savefig('skyimage.jpeg')
