# Este código genera el mapa con el área de estudio y los puntos seleccionados (Puntos SROM).

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import alphashape
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.img_tiles import OSM

# Config 
mw = 90
path_base = r'C:\Users\crist\Desktop\Trabajo'
archivo_dataset_slip = f'{path_base}/Database/TodosQuakes_mw_{mw}_filtrados.nc'

# IDs de puntos seleccionados del dominio (SROM)
lista_point_index = [586, 1389, 1897, 2364, 2978, 3621, 3872, 4532, 4897, 5253, 5545, 5713, 5975, 6315, 6514]

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
    "Tacna": (-70.2536, -18.0066)
}

# Carga el dataset
ds_slip = xr.open_dataset(archivo_dataset_slip)
lats = ds_slip['lat'].values
lons = ds_slip['lon'].values
lons = np.where(lons > 180, lons - 360, lons)

# Alpha shape del dominio
all_points = list(zip(lons, lats))
alpha = 10
polygon = alphashape.alphashape(all_points, alpha)

if not isinstance(polygon, Polygon):
    polygon = unary_union(polygon)

lon_contour, lat_contour = polygon.exterior.coords.xy

# Puntos de SROM
lista_validos = [i for i in lista_point_index if i < len(lats)]
lons_sel = lons[lista_validos]
lats_sel = lats[lista_validos]

# Figura
fig = plt.figure(figsize=(16, 10))
osm_tiles = OSM()
ax = plt.axes(projection=osm_tiles.crs)
ax.set_extent([min(lons)-1, max(lons)+4, min(lats)-1, max(lats)+1], crs=ccrs.PlateCarree())

# Fondo OSM y límites
ax.add_image(osm_tiles, 8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Contorno del dominio
ax.plot(lon_contour, lat_contour, color='red', linewidth=1, transform=ccrs.PlateCarree(), label='Límite del dominio')
ax.fill(lon_contour, lat_contour, color='red', alpha=0.2, transform=ccrs.PlateCarree())

# Puntos SROM destacados
ax.scatter(lons_sel, lats_sel, marker='^', color='gold', edgecolor='black',
           s=60, transform=ccrs.PlateCarree(), zorder=5, label='Puntos SROM')

# Etiquetas con el ID del punto
for lon, lat, gid in zip(lons_sel, lats_sel, lista_validos):
    ax.text(lon + 0.1, lat + 0.1, f"ID: {gid}", fontsize=8,
            fontweight='bold', color='black', transform=ccrs.PlateCarree(),
            bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))

# Ciudades
for name, (lon, lat) in ciudades.items():
    ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree())
    ax.text(lon + 0.15, lat + 0.15, name, fontsize=8, transform=ccrs.PlateCarree())

# Estilo del mapa
lon_min, lon_max = int(min(lons)-1), int(max(lons)+4)
lat_min, lat_max = int(min(lats)-1), int(max(lats)+1)
ax.set_xticks(np.arange(lon_min, lon_max, 4), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(lat_min, lat_max, 2), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xlabel("Longitud", fontsize=11)
ax.set_ylabel("Latitud", fontsize=11)

# ---------- LEYENDA ----------
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.show()
