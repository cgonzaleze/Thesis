# Este código se encarga de homologar el filtrado realizado para las geometrías de ruptura (W y L) mediante las ecuaciones de Blaser et al. (2010).
# Se guardan los dataset filtrados en un nuevo archivo para la aplicación del SROM desde los niveles de inundación.

import xarray as xr
import numpy as np
import os

# Carpeta base, modificarla según corresponda al directorio donde aplique
directorio = r'C:\Users\crist\Desktop\Trabajo\Database'
magnitudes = range(75, 96)  # Desde Mw 7.5 hasta Mw 9.5

for mw in magnitudes:
    # Carga TodosQuakes filtrado
    archivo_tq = os.path.join(directorio, f'TodosQuakes_mw_{mw}_filtrados.nc')
    try:
        todosquakes_dataset = xr.open_dataset(archivo_tq)
    except FileNotFoundError:
        print(f"No existe archivo filtrado para Mw {mw/10:.1f}, se omite.")
        continue

    escenarios_tq = todosquakes_dataset['ScenarioId'].values

    # Carga PTHA (AllTimeSeriesEta)
    archivo_ptha = os.path.join(directorio, f'AllTimeSeriesEta_mw_{mw}.nc')
    try:
        ptha_dataset = xr.open_dataset(archivo_ptha)
    except FileNotFoundError:
        print(f"No existe archivo AllTimeSeriesEta para Mw {mw/10:.1f}, se omite.")
        continue

    escenarios_ptha = ptha_dataset['ScenarioId'].values

    # Crea máscara booleana: True si el ScenarioId está en la lista de filtrados
    mascara = np.isin(escenarios_ptha, escenarios_tq)

    # Aplica máscara a lo largo de la dimensión Scenario
    ptha_filtrado = ptha_dataset.isel(Scenario=mascara)

    # Guarda el dataset filtrado
    archivo_filtrado = os.path.join(directorio, f'AllTimeSeriesEta_mw_{mw}_filtrados.nc')
    ptha_filtrado.to_netcdf(archivo_filtrado)
    print(f"Archivo PTHA filtrado guardado: {archivo_filtrado}")

    # Verificación: Verifica el len entre ambos datasets, si no coinciden, el proceso tuvo un error. Corregir
    num_escenarios_tq = len(escenarios_tq)
    num_escenarios_ptha = np.sum(mascara)
    if num_escenarios_tq == num_escenarios_ptha:
        print(f"Verificación completa para Mw {mw / 10:.1f}: Escenarios coinciden ({num_escenarios_tq}).")
    else:
        print(f"Mw {mw / 10:.1f}: Escenarios no coinciden — TodosQuakes: {num_escenarios_tq}, PTHA filtrado: {num_escenarios_ptha}.")
