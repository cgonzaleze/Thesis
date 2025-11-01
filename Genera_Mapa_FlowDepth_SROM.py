# Este código genera el mapa con todos los puntos del dataset de inundación
# y resalta los puntos de interés (grillas seleccionadas) con íconos y etiquetas.

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.img_tiles import OSM

# Config
mw = 90
path_base = r'C:\Users\crist\Desktop\Trabajo'
archivo_dataset_inundacion = f'{path_base}/Database/AllTimeSeriesEta_mw_{mw}_filtrados.nc'

# IDs de grillas destacadas
grillas = [112, 46, 13, 124, 22, 127, 386, 26, 444, 27, 30, 69, 33]

# Ciudades
ciudades = {
    "Arica": (-70.3126, -18.4783),
    "Iquique": (-70.1494, -20.2307),
    "Antofagasta": (-70.4047, -23.6500),
    "La Serena": (-71.2540, -29.9027),
    "Viña del Mar": (-71.5518, -33.0245),
    "San Antonio": (-71.6145, -33.5882),
    "Talcahuano": (-73.1168, -36.7167),
    "Valdivia": (-73.2459, -39.8196),
    "Puerto Montt": (-72.9411, -41.4717),
    "Castro": (-73.7706, -42.4721),
    "Pisco": (-76.2167, -13.7103),
    "Ilo": (-71.3394, -17.6333),
}

# Carga el dataset
ds_inundacion = xr.open_dataset(archivo_dataset_inundacion)
lats = ds_inundacion['latitude'].values
lons = ds_inundacion['longitude'].values
lons = np.where(lons > 180, lons - 360, lons)

# Extrae coordenadas de las grillas destacadas (asegurando que existan)
grillas_validas = [g for g in grillas if g < len(lats)]
lats_sel = lats[grillas_validas]
lons_sel = lons[grillas_validas]

# Figura
fig = plt.figure(figsize=(16, 10))
osm_tiles = OSM()
ax = plt.axes(projection=osm_tiles.crs)
ax.set_extent([min(lons)-1, max(lons)+4, min(lats)-1, max(lats)+1], crs=ccrs.PlateCarree())

# Agrega fondo OSM y límites
ax.add_image(osm_tiles, 8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Puntos de inundación
ax.scatter(lons, lats, color='red', s=1, transform=ccrs.PlateCarree(),
           alpha=0.6, label='Puntos del dominio')

# Grillas destacadas
ax.scatter(lons_sel, lats_sel, marker='^', color='gold', edgecolor='black',
           s=60, transform=ccrs.PlateCarree(), zorder=5, label='Puntos SROM')

# Etiquetas con ID de grilla
for lon, lat, gid in zip(lons_sel, lats_sel, grillas_validas):
    ax.text(lon + 0.1, lat + 0.1, f"ID: {gid}", fontsize=8,
            fontweight='bold', color='black', transform=ccrs.PlateCarree(),
            bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))

# # Ciudades
# for name, (lon, lat) in ciudades.items():
#     ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree())
#     ax.text(lon + 0.15, lat + 0.15, name, fontsize=8, transform=ccrs.PlateCarree())
    
# Ciudades
for name, (lon, lat) in ciudades.items():
    ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree())
    ax.text(lon + 0.15, lat - 0.8, name, fontsize=8, transform=ccrs.PlateCarree())

# Estilo del mapa
lon_min, lon_max = int(min(lons)-1), int(max(lons)+4)
lat_min, lat_max = int(min(lats)-1), int(max(lats)+1)
ax.set_xticks(np.arange(lon_min, lon_max, 4), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(lat_min, lat_max, 2), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xlabel("Longitud", fontsize=11)
ax.set_ylabel("Latitud", fontsize=11)

# Leyenda
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.show()
