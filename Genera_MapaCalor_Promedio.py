# Este mapa realiza el cálculo del deslizamiento promedio para los escenarios seleccionados, agrupados por tamaño de srom.
# Es decir, para una magnitud (ej: 8.8), agarra todos los putnos de norte a sur donde se aplicó SROM (deslizamientos) y
# grafica el deslizamiento promedio a lo largo del dominio con el fin de ver si existe sobrerepresentación en algunos puntos
# en particular. Procurar que la selección de puntos sea homogénea para no inducir sesgo.

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

# CONFIGURACIÓN
path_base = r'C:\Users\crist\Desktop\Trabajo'

# Magnitud
mw_ref = 92
srom_size = [25, 50, 100, 200]
slip_list = [586, 1389, 1897, 2364, 2978, 3621, 3872, 4532, 4897, 5253, 5545, 5713, 5975, 6315, 6514]

ciudades = {
    'Pisco': -13.7,
    'Mejillones': -23.1,
    'Tongoy': -30.4,
    'Arauco': -37.2
}

# Colormap personalizado
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_slip", ["#ffffff", "#ffff00", "#ff9900", "#ff0000", "#4b0000"], N=256
)

# FUNCIÓN PRINCIPAL 
def promedio_slips_por_tamaño(path_base, mw_ref, srom_size, slip_list):
    """Calcula el promedio de slips por tamaño de SROM para una magnitud fija."""
    slips_por_tamaño = {}
    lons = None
    lats = None

    mw_int = int(mw_ref)
    archivo_dataset = os.path.join(path_base, f'Database/TodosQuakes_mw_{mw_int}_filtrados.nc')

    if not os.path.exists(archivo_dataset):
        raise FileNotFoundError(f"No se encontró el dataset para Mw {mw_ref}")

    ds = xr.open_dataset(archivo_dataset)
    ds['slip'] = ds['slip'].fillna(0.0)

    lats = ds['lat'].values
    lons = ds['lon'].values
    lons = np.where(lons > 180, lons - 360, lons)
    scenario_ids = ds['ScenarioId'].values.astype(int)

    # Iterar sobre cada tamaño de SROM
    for tamaño in srom_size:
        slip_sum = np.zeros_like(ds['slip'].values[:, 0])
        count = 0

        for grilla in slip_list:
            archivo_npz = os.path.join(path_base, f'SROM_resultados_{mw_ref}_{tamaño}_{grilla}.npz')
            if not os.path.exists(archivo_npz):
                continue

            data = np.load(archivo_npz)
            escenarios = data['scenarios'].astype(int)

            # Acumular slip para los escenarios seleccionados
            for esc_id in escenarios:
                if esc_id in scenario_ids:
                    idx = np.where(scenario_ids == esc_id)[0][0]
                    slip = ds['slip'][:, idx].values / 100  # convertir cm → m
                    slip_sum += slip
                    count += 1

        # Promedio solo si hay escenarios válidos
        if count > 0:
            slip_promedio = slip_sum / count
            slips_por_tamaño[tamaño] = slip_promedio

    ds.close()
    return slips_por_tamaño, lons, lats


# CARGAR Y CALCULAR PROMEDIOS
slips_por_tamaño, lons, lats = promedio_slips_por_tamaño(path_base, mw_ref, srom_size, slip_list)

# UNIFICAR ESCALA DE COLORES
todos_valores = np.concatenate([slip for slip in slips_por_tamaño.values() if slip is not None])
vmin, vmax = np.nanmin(todos_valores), np.nanmax(todos_valores)

print(f"Escala de colores unificada: vmin={vmin:.3f}, vmax={vmax:.3f}")

# FIGURA CON GRIDSPEC
osm_tiles = OSM()
fig = plt.figure(figsize=(28, 7))
gs = GridSpec(1, len(srom_size), figure=fig, wspace=0.0)
axes = [fig.add_subplot(gs[i], projection=osm_tiles.crs) for i in range(len(srom_size))]

# PLOTEO
for i, tamaño in enumerate(srom_size):
    ax = axes[i]
    ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1], crs=ccrs.PlateCarree())
    ax.add_image(osm_tiles, 8)

    if tamaño not in slips_por_tamaño:
        print(f"No hay datos promediados para SROM m={tamaño}")
        continue

    slip_data = slips_por_tamaño[tamaño]
    sc = ax.scatter(lons, lats, c=slip_data, cmap=cmap, s=8, vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree())

    # # Líneas de referencia de ciudades
    # for nombre, lat_ciudad in ciudades.items():
    #     ax.hlines(y=lat_ciudad, xmin=min(lons)-1, xmax=max(lons)+1,
    #               colors='k', linestyles='--', linewidth=1.0, transform=ccrs.PlateCarree())
    #     ax.text(max(lons)+0.3, lat_ciudad, nombre, fontsize=9, va='center', transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Formato de ejes
    if i != 0:
        ax.set_yticklabels([])
    ax.set_xticks(np.linspace(min(lons), max(lons), 2), crs=ccrs.PlateCarree()) #3
    ax.set_yticks(np.arange(int(min(lats))-1, int(max(lats))+2, 2), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_title(fr"SROM m = {tamaño}", fontsize=14)

    # Colorbar individual
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.locator = MaxNLocator(integer=False, nbins=5)
    cbar.update_ticks()
    if i == len(srom_size) - 1:
        cbar.set_label("Deslizamiento promedio [m]")

plt.suptitle(fr"Mw {mw_ref/10:.1f}", fontsize=16, y=0.98)
plt.show()
