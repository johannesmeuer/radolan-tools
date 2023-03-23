import argparse
import os
import sys

import numpy as np
import wradlib as wrl
import netCDF4 as nc
from cftime import date2num


def convert(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str, help="Directory of binary input files")
    parser.add_argument("--output", type=str, default="output.nc", help="Filename of the output netCDF file")
    args = parser.parse_args(argv)

    # Create the netCDF file
    ncfile = nc.Dataset(args.output, 'w', format='NETCDF4')

    files = sorted(os.listdir(args.indir))

    counter = 0
    for file in files:
        filepath = args.indir + file
        filehandle = wrl.io.get_radolan_filehandle(filepath)
        header = wrl.io.read_radolan_header(filehandle)
        attrs = wrl.io.parse_dwd_composite_header(header)

        # Read the binary data
        data, rxattrs = wrl.io.read_radolan_composite(filepath)

        # Define the dimensions
        if counter == 0:
            time = ncfile.createDimension('time', None)
            y = ncfile.createDimension('y', data.shape[0])
            x = ncfile.createDimension('x', data.shape[1])

            # time
            var_t = ncfile.createVariable('time', 'double', ('time'))
            var_t.setncattr('units', "hours since 2001-01-01 00:00:00")
            var_t.setncattr('standard_name', 'time')
            var_t.setncattr('calendar', 'proleptic_gregorian')
            var_t.setncattr('axis', 'T')

            # lats
            lats = ncfile.createVariable('lat', 'float', ('y', 'x'))
            lats.setncattr('standard_name', 'longitude')
            lats.setncattr('long_name', 'longitude')
            lats.setncattr('units', 'degrees_east')
            lats.setncattr('_CoordinateAxisType', 'Lon')

            # lons
            lons = ncfile.createVariable('lon', 'float', ('y', 'x'))
            lons.setncattr('standard_name', 'latitude')
            lons.setncattr('lat_name', 'latitude')
            lons.setncattr('units', 'degrees_north')
            lons.setncattr('_CoordinateAxisType', 'Lat')

            # x
            var_x = ncfile.createVariable('x', 'float', ('x'))
            var_x.setncattr('standard_name', 'projection_x_coordinate')
            var_x.setncattr('units', 'm')
            var_x.setncattr('axis', 'X')

            # y
            var_y = ncfile.createVariable('y', 'float', ('y'))
            var_y.setncattr('standard_name', 'projection_y_coordinate')
            var_y.setncattr('units', 'm')
            var_y.setncattr('axis', 'Y')

            # projection
            crs = ncfile.createVariable('crs', 'int')
            crs.setncattr('grid_mapping_name', 'Projection')

            # precipitation
            radar_data = ncfile.createVariable('pr', 'float', ('time', 'y', 'x'))
            radar_data.setncattr('standard_name', 'precipitation_flux')
            radar_data.setncattr('units', 'kg m-2 s-1')
            radar_data.setncattr('grid_mapping', 'crs')
            radar_data.setncattr('coordinates', 'lat lon')
            radar_data.setncattr('cell_methods', 'time: mean')

            times = []
            lat_lon = wrl.georef.get_radolan_grid(1100, 900, wgs84=True)
            lats[:] = lat_lon[:, :, 0]
            lons[:] = lat_lon[:, :, 1]

        # Assign values to the variables
        times.append(date2num(rxattrs['datetime'], var_t.units))

        data[data==-9999] = np.nan

        radar_data[counter, :, :] = data

        counter += 1

    times = np.array(times)
    var_t[:] = times
    # Close the netCDF file
    ncfile.close()


if __name__ == "__main__":
    convert(sys.argv[1:])