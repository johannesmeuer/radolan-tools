import argparse
import sys

import cartopy
import cartopy.crs as ccrs
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray
from cartopy.feature import ShapelyFeature
from matplotlib.colors import ListedColormap
from scipy import ndimage

from utils import get_ndat, parse_slops
import cartopy.io.shapereader as shpreader


def parse_title(title):
    if title is None:
        return None
    elif title == "None":
        return ""
    else:
        return title.replace("_", " ")


def str_list(arg):
    if isinstance(arg, str):
        return arg.split(',')
    else:
        return arg


def int_list(arg):
    return list(map(int, arg.split(',')))


def float_list(arg):
    if isinstance(arg, str):
        return list(map(float, arg.split(',')))
    else:
        return arg


def ncplot(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", type=str, help="Filename of the netCDF file", nargs="+")
    parser.add_argument("--masknames", type=str, help="Filename of the netCDF file", nargs="+")
    parser.add_argument("-i", "--indices", type=int_list, default=None, help="List of selected date by string or index")
    parser.add_argument("-v", "--selvars", type=str_list, default=[[None]], help="Comma separated selected variables",
                        nargs="+")
    parser.add_argument("--dim-names", type=str_list, default=None,
                        help="Comma seprated list of dimension names (lon,lat,time)")
    parser.add_argument("-m", "--maskval", type=float_list, default=None,
                        help="Mask the values between val1 and val2 (val1,val2)")
    parser.add_argument("--fps", type=int, default=5, help="Number of rows for the 2d plots")
    parser.add_argument("--hide-cbar", action='store_true', help="Hide colorbars")
    parser.add_argument("--hide-axes", action='store_true', help="Hide axes")
    parser.add_argument("--skip-checks", action='store_true', help="Skip the checks")
    parser.add_argument("--create-vid", action='store_true', help="Skip the checks")
    parser.add_argument("--cblabel", type=parse_title, default="mm/h", help="Label of the colorbar")
    parser.add_argument("--title", type=parse_title, default=None, help="Title of the plot")
    parser.add_argument("--exp", type=str, default=None, help="Export as filename")
    args = parser.parse_args(argv)
    slops_dict = parse_slops(argv)

    images = get_ndat(args.fnames, selvars=args.selvars, dim_names=args.dim_names, maskval=args.maskval,
                      lib="xarray", slops_dict=slops_dict,
                      skip_checks=args.skip_checks)
    if args.masknames:
        masks = get_ndat(args.masknames, selvars=args.selvars, dim_names=args.dim_names, maskval=args.maskval,
                         verbose=args.verbose, lib="xarray", slops_dict=slops_dict,
                         skip_checks=args.skip_checks)

    map_proj = ccrs.Stereographic(
        true_scale_latitude=60.0, central_latitude=90.0, central_longitude=10.0
    )

    nws_precip_colors = [
        "#04e9e7",  # 0.01 - 0.10 inches
        "#019ff4",  # 0.10 - 0.25 inches
        "#0300f4",  # 0.25 - 0.50 inches
        "#02fd02",  # 0.50 - 0.75 inches
        "#01c501",  # 0.75 - 1.00 inches
        "#008e00",  # 1.00 - 1.50 inches
        "#fdf802",  # 1.50 - 2.00 inches
        "#e5bc00",  # 2.00 - 2.50 inches
        "#fd9500",  # 2.50 - 3.00 inches
        "#fd0000",  # 3.00 - 4.00 inches
        "#d40000",  # 4.00 - 5.00 inches
        "#bc0000",  # 5.00 - 6.00 inches
        "#f800fd",  # 6.00 - 8.00 inches
        "#9854c6",  # 8.00 - 10.00 inches
        "#1f0a2e",  # 10.00+
    ]
    precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)
    levels = [0.13, 0.2, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
              6.0, 8.0, 10.]
    #levels = [20*level for level in levels]
    norm = matplotlib.colors.BoundaryNorm(levels, 15)
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.family': "serif"})

    if args.indices:
        indices = args.indices
    else:
        indices = range(images.ds.pr.shape[0])

    for i in indices:
        fig = plt.figure(figsize=(10, 8))

        image = images.ds.pr.isel(time=i)
        mask = np.isnan(image)
        new_mask = ndimage.binary_dilation(mask, iterations=4)
        mask = np.logical_xor(new_mask, mask)

        image = image.fillna(0)
        image = xarray.where(image < 0, 0, image)
        image = image.where(mask == False)

        if args.masknames:
            image = image.where(masks.ds.pr.isel(time=0) == 0)

        cmap = precip_colormap
        new_cmap = cmap(np.arange(cmap.N))
        new_cmap[:, -1] = np.ones(cmap.N)
        new_cmap[:, -1][0] = 0
        new_cmap = ListedColormap(new_cmap)
        new_cmap.set_bad('black', 1.)
        cmap.set_bad('black', 1.)

        if args.hide_cbar:
            pm = image.plot(subplot_kws=dict(projection=map_proj), add_colorbar=False,
                        cmap=new_cmap, norm=norm, cbar_kwargs={'label': args.cblabel})
        else:
            pm = image.plot(subplot_kws=dict(projection=map_proj), add_colorbar=True,
                        cmap=new_cmap, norm=norm)
        ax = plt.gca()

        # Read Natural Earth data
        shpfilename = shpreader.natural_earth(resolution='10m',
                                              category='cultural',
                                              name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        surround_countries = ["Belgium", "Netherlands", "Denmark", "Italy", "Switzerland", "Poland", "Czech Republic",
                              "Luxembourg", "France", "Sweden", "Austria", "Slovenia"]
        surround_country_records = []
        germany = None
        for record in reader.records():
            if "Germany" in record.attributes["NAME_LONG"]:
                germany = record
            for country in surround_countries:
                if country in record.attributes["NAME_LONG"]:
                    surround_country_records.append(record)
        # Display Kenya's shape
        shape_feature = ShapelyFeature([germany.geometry], ccrs.PlateCarree(), facecolor="white", edgecolor='darkgray',
                                       lw=1, alpha=0.0)
        ax.add_feature(shape_feature)

        for record in surround_country_records:
            shape_feature = ShapelyFeature([record.geometry], ccrs.PlateCarree(), facecolor="gray", edgecolor='darkgray',
                                           lw=1, alpha=0.2)
            ax.add_feature(shape_feature)
        ax.add_feature(cartopy.feature.OCEAN)

        #plt.axes(projection=ccrs.PlateCarree())

        ahrweiler_latitude = 50.54
        ahrweiler_longitude = 7.09

        # Convert lon-lat to the specified projection
        lon_lat_point = map_proj.transform_point(ahrweiler_longitude, ahrweiler_latitude, ccrs.PlateCarree())

        # Add a marker at the converted lon-lat point
        #ax.scatter(lon_lat_point[0], lon_lat_point[1], marker='o', color='red', label='Marker Label')

        plt.scatter(lon_lat_point[0], lon_lat_point[1], marker='o', color='black', label='Marker Label', s=200)
        plt.text(lon_lat_point[0] + 30000, lon_lat_point[1], 'Ahrweiler')

        ax.add_feature(cartopy.feature.COASTLINE, edgecolor="darkgray", linewidth=1)
        ax.add_feature(cartopy.feature.BORDERS, color="darkgray", linewidth=1)

        if args.title:
            ax.set_title(args.title)
        else:
            ax.set_title("")

        plt.savefig("{}{}.png".format(args.exp, i), bbox_inches='tight', dpi=300)

    if args.create_vid:
        with imageio.get_writer('{}.gif'.format(args.exp), mode='I', fps=args.fps) as writer:
            for i in range(images.ds.pr.shape[0]):
                image = imageio.imread('{}{}.png'.format(args.exp, str(i)))
                writer.append_data(image)

if __name__ == "__main__":
    ncplot(sys.argv[1:])
