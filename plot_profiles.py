import sys
import os
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def fractional_year_to_datetime(year_frac):
    """
    Convert a decimal‐year (e.g. 2025.5) into a datetime,
    mapping the fraction onto the day‐of‐year (handles leap years).
    """
    year = int(year_frac)
    frac = year_frac - year
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days = 366 if is_leap else 365
    return datetime.datetime(year, 1, 1) + datetime.timedelta(days=frac * days)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <input.nc>")
        sys.exit(1)

    infile = sys.argv[1]
    base   = os.path.splitext(os.path.basename(infile))[0]
    # use the full base name (e.g. "2013180N11256.bin_julian_time")
    prefix = base
    outbase = f"{prefix}_profiles"

    ds = xr.open_dataset(infile, decode_times=False)

    # 1) time as decimal years → list of datetimes or None
    tvar  = ds["time"]
    tvals = tvar.values
    tfill = tvar.attrs.get("_Fillvalue")
    times = [
        fractional_year_to_datetime(tv) if (tfill is None or tv != tfill) else None
        for tv in tvals
    ]

    # 2) storm‐grid coords + fill values
    si     = ds["storm_grid_coord_i"].values   # (ni, nk)
    sj     = ds["storm_grid_coord_j"].values
    fill_i = ds["storm_grid_coord_i"].attrs.get("_Fillvalue")
    fill_j = ds["storm_grid_coord_j"].attrs.get("_Fillvalue")

    # 3) pick time index with most valid collocations
    valid_mask = (si != fill_i) & (sj != fill_j)
    counts     = valid_mask.sum(axis=0)
    k_max      = int(counts.argmax())
    slots      = np.where(valid_mask[:, k_max])[0]

    # 4) load SST & pressure DataArrays and lon/lat grids
    sst_da   = ds["sea_surface_temperature"]            # (ni, nj, nk)
    press_da = ds["observation_level_in_pressure"]      # (ni, nj, nk)
    lon2     = ds["lon2"].values                        # (nj, ni)
    lat2     = ds["lat2"].values

    # 5) plot
    fig, ax = plt.subplots(figsize=(6, 8))
    for slot in slots:
        temp_profile = sst_da.isel(ni=slot, nk=k_max).values
        plev_profile = press_da.isel(ni=slot, nk=k_max).values

        i_idx = int(si[slot, k_max])
        j_idx = int(sj[slot, k_max])
        lon, lat = lon2[j_idx, i_idx], lat2[j_idx, i_idx]

        ax.plot(temp_profile, plev_profile, label=f"{lon:.2f},{lat:.2f}")

    # set pressure range 0–200 dbar and invert
    ax.set_ylim(200, 0)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (dbar)")

    date_str = times[k_max].strftime("%Y-%m-%d") if times[k_max] else "unknown_date"
    ax.set_title(f"{prefix}: {len(slots)} profiles on {date_str}")

    ax.legend(fontsize="small", loc="best")
    plt.tight_layout()

    # 6) save
    for ext in ("pdf", "png"):
        fig.savefig(f"{prefix}_profiles.{ext}")
    print(f"Saved {prefix}_profiles.pdf and {prefix}_profiles.png")

if __name__ == "__main__":
    main()