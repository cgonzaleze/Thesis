# Este código se encarga de aplicar las relaciones de escala (Blaser et al., 2010) para filtrar los escenarios generados mediante expansión
# Karhunen-Loeve, con el fin de obtener únicamente la geometría de ruptura relevante para el análisis, una vez realizado el filtrado,
# se guardan únicamente los escenarios que cumplen los criterios en un nuevo dataset, el cual será el utilizado para la aplicación de SROM.

import netCDF4
import h5netcdf
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import os

# Desviaciones estándar consultadas (Blaser et al., 2010)
s_a_L, s_b_L = 0.18*0.25, 0.18*0.25 # Reverse slip type: Orthogonal
s_a_W, s_b_W = 0.17*0.25, 0.17*0.25 # Reverse slip type: Orthogonal

# Se define el rango de magnitudes de los dataset
magnitudes = range(75, 96)  # Desde Mw 7.5 hasta Mw 9.5
datasets = {}
escenarios_filtrados = {}

# Carpeta base, modificarla según corresponda al directorio donde aplique
directorio = r'C:\Users\crist\Desktop\Trabajo\Database'

# Se cargan los dataset en el loop, filtrando dentro
for mw in magnitudes:
    # Ruta del archivo original
    archivo = os.path.join(directorio, f'TodosQuakes_mw_{mw}.nc')
    dataset = xr.open_dataset(archivo)
    datasets[mw] = dataset
    L = dataset['L']
    W = dataset['W']
    Mw = dataset['Mw']

    # Se aplica el filtro de escenarios (Aránguiz et al., 2024) para el largo y ancho de ruptura
    escenarios_filtrados[mw] = dataset.where(
        (L >= 10**(-2.37 - s_a_L + (0.57 - s_b_L)*Mw)) &
        (L <= 10**(-2.37 + s_a_L + (0.57 + s_b_L)*Mw)) &
        (W >= 10**(-1.86 - s_a_W + (0.46 - s_b_W)*Mw)) &
        (W <= 10**(-1.86 + s_a_W + (0.46 + s_b_W)*Mw)),
        drop=True
    )

    # Se guarda el dataset filtrado como archivo NetCDF
    archivo_filtrado = os.path.join(directorio, f'TodosQuakes_mw_{mw}_filtrados.nc')
    escenarios_filtrados[mw].to_netcdf(archivo_filtrado)
    print(f"Archivo guardado: {archivo_filtrado}")

# únicamente para corroborar el procedimiento
for mw, dataset_filtrado in escenarios_filtrados.items():
    print(f"Escenarios filtrados para Mw {mw / 10:.1f}:")
    print(dataset_filtrado)
    print("\n" + "-" * 40 + "\n")
