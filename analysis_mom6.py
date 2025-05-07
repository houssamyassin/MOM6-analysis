# ----------------------------------------------------------------------
# analysis_mom6.py
# ----------------------------------------------------------------------
"""
analysis_mom6.py

MOM6Simulation: load, analyze and plot MOM6 ocean model output.
"""
# ----------------------------------------------------------------------

# Standard library
import re
from pathlib import Path

# Science python
import numpy as np
import xarray as xr

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import path as mpath
import matplotlib.colors as mcolors

# Cartopy
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


class MOM6_simulation:
    """
    Load, analyze, and plot MOM6 output.

    Parameters
    ----------
    base_path : str
        Root directory of your MOM6 archives (e.g. /glade/derecho/…).
    case_name : str
        Name of the simulation case (e.g. “my_experiment”).
    user : str
        Username for archive paths (default: 'yhoussam').
    computer : str
        Hostname or machine identifier for path logic (default: 'derecho').
    date : str, optional
        Initial date token in YYYY-MM-DD or custom format.
    token_date : str, optional
        Separator between year & month in filenames (default: '-').
    token_path : str, optional
        Suffix between path and date token (default: '.').
    loc : str, optional
        Which archive layout to use ('archive' or 'campaign').
    """
    def __init__(self,short_name,case_name,user='yhoussam',computer='derecho',
                 year=None,month=None,date=None,token_date='-',token_path='.',
                 loc='archive'):
        self.short_name = short_name
        self.case_name = case_name
        self.user = user
        self.computer = computer

        self.token_date = token_date
        self.token_path = token_path
        self.date = None
        self.set_date(year=year,month=month,date=date)

        self.native_end = 'h.native'
        self.surf_end = 'h.sfc'
        self.z_end =  'h.z'
        self.frc_end = 'h.frc'
        self.grid_end = 'h.static'
        self.ice_end = 'cice'
        
        self.set_base(loc=loc)
        self.set_paths()

        self.ds = None
        self.native_mode = 'native'
        self.surf_mode = 'surface'
        self.z_mode = 'z'
        self.ice_mode = 'ice'
        self.frc_mode = 'frc'
        self.ds_type = None # One of 'native', 'surface', 'z'

        self.depth = None
        self.ocn_data_path = self.base + self.case_name + '/ocn/hist/'
        self.ice_data_path = self.base + self.case_name + '/ice/hist/'


        self.regions = {
                    'North Atlantic': {'bounds': [-80, 20, 70, 10]},
                    'Kuroshio': {'bounds': [120, 180, 20, 50]},
                    'South Atlantic': {'bounds': [-60, 0, -30, -65]},
                    'Agulhas': {'bounds': [0, 60, -65, -30]}
                  }
        
    ########################################
    ### Setup / Loading datsets ############
    ########################################

    # Returns a path to the simulations excluding the data and the .nc. These need to be added afterwards
    def get_data_path(self,ending):
        if ending == 'cice':
            path = self.base + self.case_name + '/ice/hist/'
            data_path = path + self.case_name + '.cice.h'
        else:
            path = self.base + self.case_name + '/ocn/hist/'
            data_path  = path + self.case_name + '.mom6.' + ending
        return data_path
    
    def get_grid_path(self):   
        path = self.base + self.case_name + '/ocn/hist/'
        grid_path = path + self.case_name + '.mom6.' + self.grid_end +'.nc'
        return grid_path
    
    def set_tokens(self,token_date=None,token_path=None):
        if token_date is not None:
            self.token_date = token_date
        if token_path is not None:
            self.token_path = token_path
    
    def set_base(self,loc=None,base=None):
        if loc == 'archive':
            self.base = '/glade/derecho/scratch/'+self.user+'/archive/'
        if loc == 'campaign':
            self.base = '/glade/campaign/cesm/development/omwg/projects/MOM6/'
        if base is not None:
            self.base = base
    
    def set_paths(self,native_end=None,surf_end=None,
                  z_end=None,grid_end=None,ice_end=None,frc_end=None):
        if native_end is not None:
            self.native_end = native_end
        if surf_end is not None:
            self.surf_end = surf_end
        if z_end is not None:
            self.z_end = z_end
        if ice_end is not None:
            self.ice_end = ice_end
        if grid_end is not None:
            self.grid_end = grid_end
        if frc_end is not None:
            self.frc_end = frc_end

        
        self.native_path = self.get_data_path(self.native_end)
        self.surf_path = self.get_data_path(self.surf_end)
        self.z_path = self.get_data_path(self.z_end)
        self.frc_path = self.get_data_path(self.frc_end)
        self.ice_path = self.get_data_path(self.ice_end)
        self.grid_path = self.get_grid_path()
    
    def set_date(self,year=None,month=None,date=None):
        if year is not None and month is not None:
            self.date = year + self.token_date + month
        if date is not None:
            self.date = date

    def set_ds_type(self,data_path):
        if data_path == self.native_path:
            self.ds_type = self.native_mode
        elif data_path == self.surf_path:
            self.ds_type = self.surf_mode
        elif data_path == self.z_path:
            self.ds_type = self.z_mode
        elif data_path == self.ice_path:
            self.ds_type = self.ice_mode
        elif data_path == self.frc_path:
            self.ds_type = self.frc_mode
        
    def open_single_file(self,data_path,date=None):
        if date is not None:
            self.date = date
        
        if self.date is not None:
            path = data_path + self.token_path + self.date + '.nc'
            self.ds = xr.open_dataset(path)
            self.set_ds_type(data_path)
            print("Opened file.")
        else:
            print("Not opening file! Set date first.")
    
    def open_all_files(self,data_path,preprocess=None,chunks=None):
        path = data_path + "*.nc"
        self.ds = xr.open_mfdataset(path, \
                                    parallel=True, data_vars='all', \
                                    coords='all', compat='override',
                                    preprocess=preprocess,chunks=chunks)
        self.set_ds_type(data_path)
        print('Opened all files')

        
    def open_files_by_year(self,
                           data_path,
                           match_str,
                           year_start,
                           year_end,
                           preprocess=None,
                           chunks=None):
        """
        Open only those .nc files in data_path whose filename
        (1) encodes a year between year_start and year_end inclusive,
        and (2) contains the substring match_str (if provided).
    
        Parameters
        ----------
        data_path : str or Path
            Directory containing your .nc files (will glob "*.nc").
        year_start : int
            Minimum four‑digit year (e.g. 31 for 0031).
        year_end : int
            Maximum four‑digit year (e.g. 61 for 0061).
        match_str : str, optional
            Substring that must be present in the filename.
        preprocess : callable, optional
            Function to be applied to each dataset before merging.
        chunks : dict or int, optional
            Chunking for dask.
    
        Returns
        -------
        xr.Dataset
            Combined dataset only across the requested years and matching match_str.
        """
        p = Path(data_path)
        files = sorted(p.glob("*.nc"))
    
        # regex to pull out "0031" from "...0031-09.nc"
        year_re = re.compile(r"\.(\d{4})-")
    
        def is_selected(fpath):
            name = fpath.name
            # filter by substring if requested
            if match_str and match_str not in name:
                return False
            # extract year
            m = year_re.search(name)
            if not m:
                return False
            yr = int(m.group(1))
            return year_start <= yr <= year_end
    
        selected_files = [str(f) for f in files if is_selected(f)]
    
        if not selected_files:
            # build descriptive message
            msg = f"No .nc files in {data_path} "
            msg += f"between years {year_start:04d}-{year_end:04d}"
            if match_str:
                msg += f" containing '{match_str}'"
            raise FileNotFoundError(msg)
    
        # open only the selected subset
        self.ds = xr.open_mfdataset(
            selected_files,
            parallel=True,
            data_vars="all",
            coords="all",
            compat="override",
            preprocess=preprocess,
            chunks=chunks
        )
        self.set_ds_type(data_path)
        print(f"Opened {len(selected_files)} files "
              f"({year_start:04d}-{year_end:04d}"
              + (f", match='{match_str}'" if match_str else "")
              + ")")
        self.set_ds_type(data_path)

    
    def open_grid(self):
        self.grid = xr.open_dataset(self.grid_path)[['geolon', 'geolat']]
        if self.ds is not None:
            self.ds = self.ds.assign_coords({'geolon': self.grid['geolon'],
                                            'geolat': self.grid['geolat']})
            print("Opened grid and assigned coordinates.")
        else:
            print("Not assigning coordinates! Open dataset first to assign coordinates.")

    def open_ncfile(self,data_path):
        self.ds = xr.open_dataset(data_path)
        print("Opened file.")
    
    ########################################
    ### Analysis ###########################
    ########################################

    ### Native Coordinates #################

    def compute_ssh_std(self,start_date,end_date):
        if self.ds_type != self.native_mode:
            print("Can only compute ssh_std with native coordinates.")
            return
        ds = self.ds
        ds = ds.sel(time=slice(start_date, end_date))

        ssh2 = ds.SSH**2
        ssh2 = ssh2.mean('time')
        ssh = ds.SSH.mean('time')
        ssh_std = np.sqrt(ssh2-ssh**2)  
        return ssh_std

    def compute_biharmonic(self,depth):
        if self.ds_type != self.native_mode:
            print("Can only compute biharmonic viscosity with native coordinates.")
            return
        
        ds = self.ds
        index = (np.abs(ds.zl - depth)).argmin().item()
        self.depth = ds.zl[index].item()

        biharmonic = ds['difmxybo'].isel(zl=index)
        return biharmonic

    def compute_laplacian(self,depth):
        if self.ds_type != self.native_mode:
            print("Can only compute laplacian viscosity with native coordinates.")
            return
        ds = self.ds
        index = (np.abs(ds.zl - depth)).argmin().item()
        self.depth = ds.zl[index].item()

        laplacian = ds['difmxylo'].isel(zl=index)
        return laplacian

    def compute_2D_average(self,field):
        if self.ds_type != self.native_mode:
            print("Can only compute 2D average with native coordinates.")
            return
        try:
            wet = self.ds.wet[-1]
        except AttributeError:
            print("The wet variable does not exist. Infering from nans.")
            wet= ~np.isnan(self.ds.speed.isel(time=0))
        
        area_t = self.ds.area_t
        num = (field * area_t * wet).sum(dim=['xh', 'yh'])
        denom = (area_t * wet).sum(dim=['xh', 'yh'])
        field_ave = (num/denom)[0]
        return field_ave
    
    def compute_laplacian_2D_average(self):
        if self.ds_type != self.native_mode:
            print("Can only compute 2D average with native coordinates.")
            return
        laplacian = self.ds['difmxylo']
        laplacian_ave = self.compute_2D_average(laplacian)
        laplacian_ave.name = 'laplacian_ave'
        return laplacian_ave

    def compute_biharmonic_2D_average(self):
        if self.ds_type != self.native_mode:
            print("Can only compute 2D average with native coordinates.")
            return
        biharmonic = self.ds['difmxybo']
        biharmonic_ave = self.compute_2D_average(biharmonic)
        biharmonic_ave.name = 'biharmonic_ave'
        return biharmonic_ave

    ### z Coordinates #################

    # Date of the form '0060-01-01'
    def compute_kebt_frac(self,start_date,end_date,vertical_coordinate=None):
        if self.ds_type != self.z_mode:
            print("Can only compute kebt_frac with z coordinates.")
            return
        if vertical_coordinate is None:
            vertical_coordinate = 'z_l'

        ds = self.ds.sel(time=slice(start_date, end_date))
        uo = ds.uo
        vo = ds.vo

        uo_interp = uo.interp(xq=ds.h.coords['xh'], method='linear')
        uo_interp = uo_interp.drop('xq')
        vo_interp = vo.interp(yq=ds.h.coords['yh'], method='linear')
        vo_interp = vo_interp.drop('yq')    
        u_bt= uo_interp.mean(dim=vertical_coordinate)
        v_bt = vo_interp.mean(dim=vertical_coordinate)
        kebt = 0.5*(u_bt**2+v_bt**2)
        kebt = kebt.mean('time')

        ke = ds.KE.mean(vertical_coordinate).mean('time')
        kebt_frac = kebt/ke
        kebt_frac = kebt_frac.assign_coords({'geolon': self.grid['geolon'],
                           'geolat': self.grid['geolat']})
        kebt_frac.name = 'kebt_frac'
        return kebt_frac
    
    def compute_kebt_frac_zonal(self,start_date,end_date,kebt_frac=None):
        if self.ds_type != self.z_mode:
            print("Can only compute kebt_frac with z coordinates.")
            return
        if kebt_frac is  None:
            kebt_frac=self.compute_kebt_frac(start_date,end_date)
        
        try:
            wet = self.ds.wet[-1]
        except AttributeError:
            print("The wet variable does not exist. Infering from nans.")
            wet= ~np.isnan(self.ds.speed.isel(time=0))

        kebt_frac_zonal = (kebt_frac*wet).sum(dim='xh')/wet.sum(dim='xh')
        kebt_frac_zonal.name = 'kebt_frac_zonal'

        return kebt_frac_zonal
    
    def compute_speed(self,depth):
        if self.ds_type != self.z_mode:
            print("Can only compute speed with z coordinates.")
            return
        
        ds = self.ds
        index = (np.abs(ds.z_l - depth)).argmin().item()
        self.depth = ds.z_l[index].item()

        uo = ds.uo.isel(z_l=index)
        vo = ds.vo.isel(z_l=index)

        uo_interp = uo.interp(xq=ds.h.coords['xh'], method='linear')
        uo_interp = uo_interp.drop('xq')

        vo_interp = vo.interp(yq=ds.h.coords['yh'], method='linear')
        vo_interp = vo_interp.drop('yq')

        speed = np.sqrt(uo_interp**2 + vo_interp**2)
        speed = speed.assign_coords({'geolon': self.grid['geolon'],
                       'geolat': self.grid['geolat']})
        return speed
    
    def compute_logspeed(self,depth=None,speed=None):
        if (depth is None) and (speed is None):
            print("One of depth or speed arguments must be given.")
            return
        if speed is None:
            speed = self.compute_speed(depth)
        logspeed = np.log10(np.abs(speed))
        logspeed = logspeed.assign_coords({'geolon': self.grid['geolon'],
                       'geolat': self.grid['geolat']})
        return logspeed

    ########################################
    ### Plotting ###########################
    ########################################

    def plot_global(self, field, vmin, vmax, R=None, R_threshold=None, 
                    contour_color='k', contour_linewidth=1.5, contour_linestyle='-',
                    x='geolon', y='geolat', add_colorbar = True,
                    cmap=cm.RdYlBu_r, norm=None, cbar_label='', title='', cbar_size=0.7,
                    figsize=(12,8), save_path=None, dpi=300, hor_cbar=False):
    
        plt.figure(figsize=figsize)
        orientation = 'horizontal' if hor_cbar else 'vertical'
    
        # 1) plot the main field and keep the mappable
        if add_colorbar:
            mappable = field.plot(
                vmin=vmin, vmax=vmax, norm=norm,
                x=x, y=y,
                cmap=cmap,
                subplot_kws={
                    'projection': ccrs.PlateCarree(central_longitude=250),
                    'facecolor': 'grey'
                },
                transform=ccrs.PlateCarree(),
                add_labels=False,
                add_colorbar=True,
                cbar_kwargs={
                    'shrink': cbar_size,
                    'label': cbar_label,
                    'orientation': orientation
                }
            )
        else:
            mappable = field.plot(
                vmin=vmin, vmax=vmax, norm=norm,
                x=x, y=y,
                cmap=cmap,
                subplot_kws={
                    'projection': ccrs.PlateCarree(central_longitude=250),
                    'facecolor': 'grey'
                },
                transform=ccrs.PlateCarree(),
                add_labels=False,
                add_colorbar=False
            )
            
    
        # 2) overlay contour if requested
        if R is not None and R_threshold is not None:
            R.plot.contour(
                levels=[R_threshold],
                x=x, y=y,
                colors=contour_color,
                linewidths=contour_linewidth,
                linestyles=contour_linestyle,
                transform=ccrs.PlateCarree(),
                add_labels=False
            )
    
        plt.title(title)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
        # 3) **return** the mappable
        return mappable



    def plot_global_discrete(self,
                             field,
                             levels,
                             R=None,
                             R_threshold=None,
                             contour_color='k',
                             contour_linewidth=1.5,
                             contour_linestyle='-',
                             x='geolon',
                             y='geolat',
                             cmap=cm.RdYlBu,
                             cbar_label='',
                             title='',
                             cbar_size=0.7,
                             figsize=(12,8),
                             save_path=None,
                             dpi=300):
        """
        Plot a global field with discrete color bands at the specified levels.
    
        Parameters
        ----------
        field : xarray.DataArray
            2D field to plot, with coords field[x], field[y].
        levels : array‐like
            Sequence of bin edges, e.g. np.linspace(0,2,9) for 8 bands.
        R : xarray.DataArray, optional
            2D field to contour (e.g. resolution function).
        R_threshold : float, optional
            Single contour level for R.
        contour_* : styling for the R‐contour.
        x, y : str
            Names of the longitude/latitude coords in `field`.
        cmap : Colormap
        cbar_label : str
        title : str
        cbar_size : float
        figsize : tuple
        save_path : str, optional
        dpi : int
        """
        # build a BoundaryNorm to map each interval to one color
        norm = mcolors.BoundaryNorm(boundaries=levels,
                                    ncolors=cmap.N,
                                    clip=True)
    
        # set up figure + map projection
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={
                                   'projection': ccrs.PlateCarree(
                                       central_longitude=250),
                                   'facecolor': 'grey'
                               })

        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=0)

    
        # filled contours at discrete levels
        mappable = ax.contourf(
            field[x], field[y], field,
            levels=levels,
            norm=norm,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            extend='both'
        )
    
        # optional overlay contour
        if R is not None and R_threshold is not None:
            ax.contour(
                R[x], R[y], R,
                levels=[R_threshold],
                colors=contour_color,
                linewidths=contour_linewidth,
                linestyles=contour_linestyle,
                transform=ccrs.PlateCarree()
            )
    
        # colorbar with one block per interval
        cb = fig.colorbar(
            mappable,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='uniform',
            extend='both',
            shrink=cbar_size,
            label=cbar_label
        )
    
        # gridlines & tick formatting
        gl = ax.gridlines(draw_labels=False,
                          linestyle='--',
                          color='gray',
                          alpha=0.5)
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
    
        ax.set_title(title)
    
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
        return mappable

    

    def plot_antarctica(self, field, vmin, vmax, R=None, R_threshold=None, 
                        contour_color='k', contour_linewidth=1.5, contour_linestyle='-',
                        x='geolon', y='geolat',
                        left=-180, right=180, bottom=-90, top=-60,
                        cmap=cm.RdYlBu_r, cbar_label='',
                        title='', cbar_size=0.7, figsize=(12,8),
                        save_path=None, dpi=300):
    
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        
        ax1.set_extent([left, right, bottom, top], ccrs.PlateCarree())
        
        # Plot the main field
        field.plot(vmin=vmin, vmax=vmax,
                   x=x, y=y, cmap=cmap, ax=ax1,
                   transform=ccrs.PlateCarree(),
                   add_labels=False, add_colorbar=True,
                   cbar_kwargs={'shrink': cbar_size, 'label': cbar_label})
        
        # Superimpose the contour line where R exceeds the threshold
        if R is not None and R_threshold is not None:
            R.plot.contour(levels=[R_threshold],
                           x=x, y=y,
                           colors=contour_color,  # Color of the contour line
                           linewidths=contour_linewidth,  # Width of the contour line
                           linestyles=contour_linestyle,  # Style of the contour line
                           transform=ccrs.PlateCarree(),
                           ax=ax1,
                           add_labels=False)
    
        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        
        ax1.set_boundary(circle, transform=ax1.transAxes)
        ax1.set_facecolor('grey')  # Set the face color to gray
    
        plt.title(title)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)


    def plot_zonal_average(self,field,c='k',lw=1.5,ymin=-60,ymax=60,xmin=0,xmax=1,
                           ylabel='Latitude',xlabel='',title='',grid=False,
                           figsize=(4,6),save_path=None,dpi=300):
        plt.figure(figsize=figsize)

        ax = plt.subplot(111)
        ax.plot(field,self.grid.yh,c=c,lw=lw)

        ax.set_ylim([ymin,ymax])
        ax.set_xlim([xmin,xmax])
        if grid:
            ax.grid()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)    

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight',dpi=dpi) 
    
    
    def plot_2D_average(self,field,c='k',lw=1.5,ymin=None,ymax=None,
                           ylabel=r'Depth $(m)$',xlabel='',title='',grid=False,
                           figsize=(4,6),save_path=None,dpi=300):
        
        plt.figure(figsize=figsize)

        if ymin is None:
            ymin=-self.ds.zl[-1]
        if ymax is None:
            ymax=-self.ds.zl[0]

        ax = plt.subplot(111)
        ax.plot(field,-self.ds.zl,c=c,lw=lw)

        ax.set_ylim([ymin,ymax])
        if grid:
            ax.grid()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)    

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight',dpi=dpi) 
    
    def plot_4regions(self,field,figsize=(12,10),vmin=None,vmax=None,cmap=cm.RdYlBu_r,
                        add_colorbar=True,cbar_size=0.7,cbar_label='', title='',
                        save_path=None,dpi=300):

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})

        # Iterate over regions
        for (region, info), ax in zip(self.regions.items(), axes.flat):
            field.plot(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                    x='geolon', y='geolat',
                    transform=ccrs.PlateCarree(),
                    add_labels=False,
                    add_colorbar=add_colorbar ,
                    cbar_kwargs={'shrink': cbar_size, 'label': cbar_label})
            ax.set_extent(info['bounds'])
            ax.set_title(region,fontsize=12)
            ax.set_facecolor('grey')
            

        fig.suptitle(title, fontsize=14)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight',dpi=dpi)
    
    def plot_single_region(self, field, region_bounds, figsize=(8, 6), vmin=None, vmax=None, cmap=cm.RdYlBu_r,
                          add_colorbar=True, cbar_size=0.7, cbar_label='', title='', save_path=None, dpi=300):
    
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    
        if len(region_bounds) != 4:
            raise ValueError("Region bounds should be a list/tuple of length 4.")
    
        plot = field.plot(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                   x='geolon', y='geolat',
                   transform=ccrs.PlateCarree(),
                   add_colorbar=False)  # We disable colorbar here
    
        ax.set_extent(region_bounds)
        ax.set_title(title, fontsize=12)
        ax.set_facecolor('grey')    
        if add_colorbar:
            plt.colorbar(plot, ax=ax, shrink=cbar_size, label=cbar_label)  # Pass the plot object here
    
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.show()
        

#### Other Functions ####
def plot_4global(fields,snames=['','','',''],figsize=(12,6),hspace=0.15,wspace=0.1,vmin=None,vmax=None,
                cmap=cm.RdYlBu_r,cbar=True,cbar_label=''):

    fig, axs = plt.subplots(2, 2, figsize=figsize, 
                            subplot_kw={'projection': ccrs.Robinson(central_longitude=250), 'facecolor':'grey'},
                            gridspec_kw={'hspace': hspace,'wspace': wspace},)

    # Flatten the 2D array of subplots to simplify indexing
    axs_flat = axs.flatten()

    # Loop through logspeeds and plot each one
    for i, field in enumerate(fields):
        field.plot(
            vmin=vmin, vmax=vmax,
            x='geolon', y='geolat',
            cmap=cmap,
            ax=axs_flat[i],
            transform=ccrs.PlateCarree(),
            add_labels=False,
            add_colorbar=False,  
        )

    # Add titles
    for i, sname in enumerate(snames):
        axs_flat[i].set_title(sname, fontsize=12)

    # Add a single horizontal color bar at the bottom
    if cbar:
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # Adjust the position and size as needed
        cbar = plt.colorbar(axs_flat[-1].collections[0], cax=cbar_ax, orientation='horizontal')
        cbar.set_label(cbar_label)

def max_min_mean(field):
    x = field.max()
    n = field.min()
    mean = field.mean()
    print("max=",x.values)
    print("min=",n.values)
    print("mean=",mean.values)
    return x,n,mean

def plot_section(field,y,z,e=None,figsize=(10,5),cmap=cm.RdYlBu,alpha=0.5,lw=1.5,cbar=True,
                title='',xlim=None,ylim=None):
    
    fig, ax = plt.subplots(figsize=figsize)
    

    
    im = ax.pcolormesh(y, z, field, cmap=cmap)
    
    if e is not None:
        for i in range(e.shape[0]):
            ax.plot(y,e[i],color='k',lw=lw,alpha=alpha)
    
    
    if cbar:
        cbar = plt.colorbar(im, label='Rho Values')
    
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([y[0],y[-1]])
    
    if ylim:
        ax.set_ylim(ylim)
