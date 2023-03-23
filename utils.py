import numpy as np


def get_ndat(args, selvars=[None], dim_names=None, maskval=None, lib="netcdf", slops_dict=None, verbose=1,
             skip_checks=False):
    fnames, ops, onames, selvars = parse_args(args, selvars)
    nf = len(fnames)

    for i in range(nf):
        tmp = ncview(fnames[i])
        tmp.readncdf(selvar=selvars[i], dim_names=dim_names, lib=lib, verbose=verbose)
        if i == 0:
            ndat = tmp
            ndat.ndim0 = ndat.dat.ndim
        else:
            ndat.op_ncdf(tmp, ops[i], skip_checks)
        ndat.test_multi()

    ndat = multi_slops(ndat, slops_dict)

    if not maskval is None:
        ndat.dat[(ndat.dat >= maskval[0]) & (ndat.dat <= maskval[1])] = np.nan

    ndat.plot_type = 2
    if ndat.nlon == 1 and ndat.nlat == 1:
        ndat.plot_type = 1
        ndat.dat = ndat.dat.squeeze()
        if ndat.dat.ndim == 1:
            ndat.dat = [ndat.dat]
        else:
            ndat.multi_data = True

    if ndat.multi_data:
        ndat.nplots = len(ndat.dat)
    else:
        ndat.nplots = 1

    return ndat


def get_ncdf(fname, dim_names=None, lib="netcdf", verbose=1):
    if lib == "netcdf":
        from netCDF4 import Dataset
        ds = Dataset(fname, mode='r')
    elif lib == "xarray":
        import xarray as xr
        try:
            ds = xr.open_dataset(fname)
        except:
            ds = xr.open_dataset(fname, decode_times=False)

    ddims, vars = get_dimvar(ds, dim_names=dim_names, lib=lib, verbose=verbose)
    return ds, ddims, vars


def trans_ncdf(inpname, outname, selnames=None, dicvalues=None):
    from netCDF4 import Dataset

    with Dataset(inpname) as src, Dataset(outname, "w") as dst:
        # copy global attributes all at once via dictionary
        dst.setncatts(src.__dict__)
        # copy dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
        # copy all file data except for the excluded
        for name, variable in src.variables.items():
            if selnames is None or name in selnames:
                x = dst.createVariable(name, variable.datatype, variable.dimensions)
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].__dict__)
                if not dicvalues is None and name in dicvalues.keys():
                    dst[name][:] = dicvalues[name]
                else:
                    dst[name][:] = src[name][:]


def comp_ncdf(ds1, ds2):
    import numpy as np

    fnames = ds1.fname + " and " + ds2.fname
    if ds1.var != ds2.var:
        print("Warning! Inconsitent variable name for files {}:".format(fnames))
        # exit()

    if not ds1.multi_data and ds1.dat.shape != ds2.dat.shape or ds1.multi_data and ds1.dat[0].shape != ds2.dat.shape:
        print("Error! Inconsitent data shapes for files {}:".format(fnames))
        exit()

    for dim in ("lon", "lat", "time"):
        if not np.array_equal(getattr(ds1, dim), getattr(ds2, dim)):
            print("Warning! Inconsistent {} dimension for files {}:".format(dim, fnames))
            choice = input("Keep going? (y/n) > ")
            if not choice in ("y", "Y"):
                exit()


def get_longname(var, units):
    if var in ("tas", "temperature_anomaly"):
        lname = "Temperature anomaly (°C)"  # +units

    elif var in ("tas_mean", "temperature_anomaly_mean"):
        lname = "Near surface temperature anomaly (°C)"  # +units
    elif var in ("TXx"):
        lname = "Max Tmax (°C)"  # +units
    elif var in ("pr"):
        lname = "Precipitation"
    else:
        lname = ""

    return lname


def get_titles(fnames, onames):
    nf = len(fnames)
    assert nf == len(onames)

    titles = [fnames[0]]
    for i in range(1, nf):
        if onames[i] is None:
            titles.append(fnames[i])
        else:
            titles[-1] += " " + onames[i] + " " + fnames[i]
    return titles


class ncview():

    def __init__(self, fname):
        self.fname = fname
        self.ds = None

    def readncdf(self, selvar=None, dim_names=None, lib="netcdf", preps=None, verbose=1):

        from netCDF4 import num2date

        ds, ddims, vars = get_ncdf(self.fname, dim_names=dim_names, lib=lib, verbose=verbose)
        self.var = sel_var(vars, selvar)
        self.ds = ds
        for k in ddims.keys():
            if ddims[k] is None:
                if k == "time":
                    setattr(self, k, np.array([""]))
                else:
                    setattr(self, k, np.array([0.]))
            else:
                setattr(self, k, ds[ddims[k]])

        self.tnum = False
        if lib == "xarray":
            if hasattr(self.time, "values"):
                if self.time.values.dtype == object:
                    self.time = ds.indexes['time'].to_datetimeindex()
                if self.time.values.dtype == "datetime64[ns]":
                    import pandas as pd
                    self.time = pd.to_datetime(self.time.values)
                elif self.time.values.dtype == "int64":
                    self.time = np.array(self.time.values)
                    self.tnum = True
                else:
                    self.time = np.array(self.time.values, dtype=float)
                    self.tnum = True
        else:
            tunits = ds.variables["time"].units
            tunits = tunits.replace("months", "days")
            self.time = num2date(self.time, tunits)

        try:
            self.units = ds.variables[self.var].units
        except:
            self.units = None

        if verbose > 4:
            import pandas as pd
            df = pd.DataFrame({"date": self.time})
            df["count"] = 1
            df = df.groupby('date')["count"].count().reset_index()
            df.to_csv('dates_count.csv', sep=' ')

        for k in ddims.keys():
            if k == "time":
                setattr(self, "min" + k, min(getattr(self, k)))
                setattr(self, "max" + k, max(getattr(self, k)))
            else:
                setattr(self, "min" + k, np.nanmin(getattr(self, k)))
                setattr(self, "max" + k, np.nanmax(getattr(self, k)))
            setattr(self, "n" + k, len(getattr(self, k)))
            if verbose > 1:
                print("* min" + k + " :", getattr(self, "min" + k))
                print("* max" + k + " :", getattr(self, "max" + k))
                print("* n" + k + " = ", getattr(self, "n" + k))
            if verbose > 2:
                print("* {} : {}, {}, {} ... {}, {}, {}".format(k, *getattr(self, k)[:3], *getattr(self, k)[-3:]))

        for k in ("lon", "lat"):

            setattr(self, "mid" + k, (getattr(self, "max" + k) - getattr(self, "min" + k)) / 2.)

            if np.isnan(getattr(self, k)).any():
                print("Warning! NaN found in {} array.".format(k))
                setattr(self, k, np.nan_to_num(getattr(self, k), nan=getattr(self, "min" + k)))

            try:
                step = np.unique(np.gradient(getattr(self, k)))
            except:
                step = [0.]
            if len(step) != 1:
                print("Warning! Non-uniform grid.")
                choice = input("Keep going? (y/n) > ")
                if not choice in ("y", "Y"):
                    exit()
                step[0] = 0
            setattr(self, "step" + k, step[0])
            if verbose > 1:
                print("* step" + k + " :", getattr(self, "step" + k))

        self.extent = [self.minlon - self.steplon / 2., self.maxlon + self.steplon / 2.,
                       self.minlat - self.steplat / 2., self.maxlat + self.steplat / 2.]
        if verbose > 1:
            print("* extent :", self.extent)
        self.dat = ds.variables[self.var][:]

        if ddims["time"] is None:
            self.dat = np.expand_dims(self.dat, axis=0)

    def test_multi(self):
        if self.dat.ndim == self.ndim0:
            self.multi_data = False
        elif self.dat.ndim == self.ndim0 + 1:
            self.multi_data = True
        else:
            print("Error! Bad number of dimensions!")
            exit()

    def op_ncdf(self, ds, op, skip_checks=False):

        import numpy as np

        self.test_multi()
        if not skip_checks:
            comp_ncdf(self, ds)

        if op is None:
            if not self.multi_data:
                self.dat = np.expand_dims(self.dat, axis=0)
            self.dat = np.ma.concatenate((self.dat, np.expand_dims(ds.dat, axis=0)))
        else:
            if self.multi_data:
                self.dat[-1] = op(self.dat[-1], ds.dat)
            else:
                self.dat = op(self.dat, ds.dat)


def parse_args(args, selvars):
    fnames, ops, onames = get_opsnames(args)
    return parse_vars(fnames, ops, onames, selvars)


def get_opsnames(args):
    opdic = {'+': np.add, '-': np.subtract, '*': np.multiply, '/': np.divide, '%': np.mod}
    optest = True
    fnames = []
    onames = [None]
    ops = [None]
    for arg in args:
        if arg in opdic.keys():
            if optest:
                print("Error! Cannot define two operators in a row!")
                exit()
            optest = True
            onames.append(arg)
            ops.append(opdic[arg])
        else:
            if not optest:
                onames.append(None)
                ops.append(None)
            optest = False
            fnames.append(arg)

    return fnames, ops, onames


def parse_vars(fnames, ops, onames, selvars):
    nf = len(fnames)
    nv = len(selvars)

    if nv == 1:
        vars = [selvars[0] for i in range(nf)]
    elif nv == nf:
        vars = selvars
    else:
        print("Error! Inconsistent number of variables and files")
        exit()

    flist, olist, nlist, vlist = [], [], [], []
    for i in range(nf):
        flist += [fnames[i] for j in vars[i]]
        olist += [ops[i] for j in vars[i]]
        nlist += [onames[i] for j in vars[i]]
        vlist += [var for var in vars[i]]
    return flist, olist, nlist, vlist


def preproc(ds, prep=None, vars=None, lon_name="lon", lat_name="lat", keepdims=False):
    import numpy as np

    if prep is None:
        return ds
    elif prep == "smean":
        return ds.mean((lon_name, lat_name), skipna=True, keepdims=keepdims)
    elif prep == "sgrad":
        return np.abs(ds.diff((lon_name, lat_name), 1))
    elif prep == "wmean":
        if vars is None:
            dims = ds.dims
            if lon_name in dims and lat_name in dims:
                weights = np.cos(np.deg2rad(ds[lat_name]))
                weighted = ds.weighted(weights)
                ds = weighted.mean((lon_name, lat_name), skipna=True)
        else:
            for var in vars:
                dims = ds[var].dims
                if lon_name in dims and lat_name in dims:
                    weights = np.cos(np.deg2rad(ds[lat_name]))
                    weighted = ds[var].weighted(weights)
                    ds[var] = weighted.mean((lon_name, lat_name), skipna=True)
                    ds[lon_name] = [0.]
                    ds[lat_name] = [0.]
                    ds[var] = ds[var].expand_dims((lon_name, lat_name))
        return ds
    elif prep == "ymean":
        import pandas as pd
        ds = ds.groupby('time.year').mean(skipna=True).rename({"year": "time"})
        ds["time"] = pd.to_datetime(ds["time"], format='%Y')
        return ds


def parse_slops(argv):
    keys = [i.replace("--", "") for i in argv if "ops" in i]
    slops_dict = {}
    for key in keys:
        slops_dict[key] = argv[argv.index("--" + key) + 1].split(",")

    return slops_dict


def multi_slops(ndat, slops_dict):
    if ndat.dat.ndim == 3:
        nd = None
    else:
        nd = len(ndat.dat)

    if nd is None:
        shape = ndat.dat.shape
    else:
        shape = ndat.dat.shape[1:]

    if shape[0] == 1:
        ndat.ntime = 1
        ndat.time = np.array([ndat.time[0]])
    if shape[1] == 1:
        ndat.nlon == 1
        ndat.lon = np.array([0.])
    if shape[2] == 1:
        ndat.nlat == 1
        ndat.lat = np.array([0.])

    return ndat


def app_slops(dat, key, kop):
    print("* Performing {} {} operation...".format(key, kop))

    if key == "tops":
        axis = 0
    elif key == "sops":
        axis = (1, 2)

    if not kop is None:

        if kop == "sum":
            dat = np.sum(dat, axis=axis)
        elif kop == "mean":
            dat = np.mean(dat, axis=axis)
        elif kop == "rmse":
            dat = np.power(dat, 2)
            dat = np.mean(dat, axis=axis)
            dat = np.sqrt(dat)
        else:
            print("Error! Operation {} not yet implemented in {}!".format(kop, key))
            exit()

    return np.expand_dims(dat, axis=axis)


def get_dimvar(ds, dim_names=None, lib="netcdf", rev=False, verbose=1):
    if lib == "netcdf":
        vars = list(ds.variables.keys())
        dims = ['lon', 'lat', 'time']

    else:
        vars = list(ds.keys())
        dims = list(ds.dims)

    if verbose > 0:
        print("* List of variables:", vars)
        print("* List of dimensions:", dims)
    if verbose > 1:
        for dim in dims:
            print("* Size of {}: {}".format(dim, len(ds[dim])))

    ddims = {"lon": None, "lat": None, "time": None}

    i = -1
    for k in ddims.keys():
        i += 1
        if dim_names is None:
            for dim in dims:
                if k in dim:
                    ddims[k] = dim
        else:
            if not dim_names[i] in dims and not dim_names[i] in vars:
                print("Warning! Dimension name {} is not in the the dimension list of the netCDF file.".format(
                    dim_names[i]))
            ddims[k] = dim_names[i]

    for k in ddims.keys():
        for var in vars:
            if ddims[k] == var:
                vars.remove(var)

    if rev:
        ddims = {v: k for k, v in ddims.items()}

    return ddims, vars


def sel_var(vars, selvar):
    if len(vars) == 1:
        selvar = vars[0]
        print("* Selecting automatically {}...".format(selvar))
    else:
        if selvar is None:
            print("* Please, select one of the following variables:")
            for var in vars:
                print("  - {}".format(var))
            selvar = input(" > ")
        print("* Selecting variable {}...".format(selvar))

    return selvar
