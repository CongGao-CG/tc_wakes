import os
import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def main():
    parser = argparse.ArgumentParser(
        description="Plot combined storm track and cross-track grid with proper layering"
    )
    parser.add_argument("ncfile", help="Path to REMSS TC wakes .nc file")
    args = parser.parse_args()

    fn = args.ncfile
    ds = xr.open_dataset(fn)

    # derive base name and track length
    base    = os.path.splitext(os.path.basename(fn))[0]
    n_track = ds.sizes['nii']

    # pick the middle along-track index
    i0 = n_track // 2

    # extract the track and the grid line at i0
    track_lons = ds.lon1.values
    track_lats = ds.lat1.values
    grid_lons  = ds.lon2.isel(nii=i0).values
    grid_lats  = ds.lat2.isel(nii=i0).values

    # compute map extent with a buffer
    buf    = 2.0
    minlon = float(ds.lon2.min()) - buf
    maxlon = float(ds.lon2.max()) + buf
    minlat = float(ds.lat2.min()) - buf
    maxlat = float(ds.lat2.max()) + buf

    # create figure & axis
    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())

    # draw land/ocean/coastlines at bottom layer
    ax.add_feature(cfeature.LAND,  facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="azure",     zorder=0)
    ax.coastlines(resolution="50m",                  zorder=0)

    # 1) plot storm track in black at zorder=1
    ax.plot(
        track_lons, track_lats,
        color='black', linewidth=1.5,
        marker='o', markersize=4,
        transform=ccrs.PlateCarree(),
        label="Storm Track",
        zorder=1
    )

    # 2) plot cross-track grid in blue at zorder=2
    ax.plot(
        grid_lons, grid_lats,
        color='blue', linewidth=1.5,
        marker='o', markersize=4,
        transform=ccrs.PlateCarree(),
        label=f"Cross-Track @ i0={i0}",
        zorder=2
    )

    # 3) highlight selected point in red at zorder=3
    ax.scatter(
        [track_lons[i0]], [track_lats[i0]],
        color='red', s=100, marker='*',
        transform=ccrs.PlateCarree(),
        label=f"Selected Point i0={i0}",
        zorder=3
    )

    # set title and labels
    ax.set_title(f"{base}  —  {n_track} track points", fontsize=14)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude  (°N)")

    # legend & extent
    ax.legend(loc="upper left")
    ax.set_extent([minlon, maxlon, minlat, maxlat],
                  crs=ccrs.PlateCarree())

    # save outputs
    fig.savefig(f"{base}_track.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}_track.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
