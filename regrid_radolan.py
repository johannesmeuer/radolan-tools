import argparse
import sys

import numpy as np
import xarray as xr
import xesmf as xe


def regrid_radolan(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Filename of the netCDF file")
    parser.add_argument("--out-name", type=str, help="Filename of the regridded netCDF file", default="out.nc")
    args = parser.parse_args(argv)

    ds = xr.open_dataset(args.datadir)

    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(46.5, 55.5, 0.140625)),
            "lon": (["lon"], np.arange(3.2, 17, 0.215625)),
        }
    )

    regridder = xe.Regridder(ds, ds_out, "bilinear")

    ds_out = regridder(ds)

    ds_out.to_netcdf(args.out_name)


if __name__ == "__main__":
    regrid_radolan(sys.argv[1:])
