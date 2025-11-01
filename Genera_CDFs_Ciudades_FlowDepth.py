# Este código genera las CDF para las 13 ciudades costeras de estudio, tomando los slips definidos
# y graficando la CDF de FlowDepth para Mw 8.6, 8.8, 9.0 y 9.2 en formato mosaico (2x2),
# con ejes uniformes y ticks enteros/consistentes. Modificar según magnitudes y ciudades de interés

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, MaxNLocator

# --- CONFIGURACIÓN ---
mw_list = [86, 88, 90, 92]  # Magnitudes a graficar
grillas = [112, 46, 13, 124, 22, 127, 386, 26, 444, 27, 30, 69, 33]  # puntos de PTHA (ESTOS DEBEN COINCIDIR CON LOS ANALIZADOS EN SROM_FlowDepth_Automatico.py)
# De lo contrario, tirará error porque los archivos de resultados para esos puntos no existen.

# Aquí se define un diccionario donde la ID del punto de grilla de cada ciudad funciona como llave para el nombre de la ciudad de análisis.
ciudades_keys_grilla = {
    112: 'Arica',
    46: 'Iquique',
    13: 'Antofagasta',
    124: 'La Serena',
    22: 'Valparaíso',
    127: 'Viña del Mar',
    386: 'San Antonio',
    26: 'Talcahuano',
    444: 'Concepción',
    27: 'Coronel',
    30: 'Valdivia',
    69: 'Puerto Montt',
    33: 'Ancud'
}

# Slips específicos por ciudad. Aquí se eligen eventos (deslizaimentos) de campo cercano (los 3 primeros) y lejano (los últimos 3) para análisis a partir de distancias.
slips_por_ciudad = {
    'Arica': [1897, 5545, 6514, 2978, 3872, 2364],
    'Iquique': [1897, 5545, 6514, 2978, 3872, 2364],
    'Antofagasta': [5253, 5975, 6315, 2978, 3872, 2364],
    'La Serena': [3621, 4532, 5713, 2364, 1389, 586],
    'Valparaíso': [2978, 3872, 2364, 1897, 5545, 6514],
    'Viña del Mar': [2978, 3872, 2364, 1897, 5545, 6514],
    'San Antonio': [2978, 3872, 2364, 1897, 5545, 6514],
    'Talcahuano': [2978, 3872, 2364, 1897, 5545, 6514],
    'Concepción': [2978, 3872, 2364, 1897, 5545, 6514],
    'Coronel': [2978, 3872, 2364, 1897, 5545, 6514],
    'Valdivia': [2364, 1389, 586, 1897, 5545, 6514],
    'Puerto Montt': [2364, 1389, 586, 1897, 5545, 6514],
    'Ancud': [2364, 1389, 586, 1897, 5545, 6514]
}

# Fija los escenarios para abrir el archivo de resultados
num_escenarios = [25, 50, 100, 200]
colores_mc = 'blue'
colores_por_slip = ['orange', 'purple', 'grey', 'brown', 'green', 'black']
estilos = ['-', '--', '-.', ':']

# FUNCIONES
def extraer_flowdepth(scenarios, probs, ids_dataset, ds, grilla):
    """Extrae los valores de FlowDepth y probabilidades correspondientes a los escenarios válidos."""
    flowdepth = []
    probs_validas = []
    for i, esc in enumerate(scenarios):
        if esc in ids_dataset:
            idx = np.where(ids_dataset == esc)[0][0]
            valor = ds['FlowDepth'].values[idx, grilla]
            flowdepth.append(valor)
            probs_validas.append(probs[i])
    return np.array(flowdepth), np.array(probs_validas)


# LOOP PRINCIPAL
for grilla in grillas:
    ciudad = ciudades_keys_grilla[grilla]

    if ciudad not in slips_por_ciudad:
        print(f"No hay lista de slips definida para {ciudad}.")
        continue

    slip_points = slips_por_ciudad[ciudad]
    pdf_filename = f"CDFs_{ciudad}.pdf"

    with PdfPages(pdf_filename) as pdf:
        # Crea figura en formato mosaico 2x2 con ejes independientes
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
        axes = axes.flatten()

        # Determina límites de ejes X comunes (máximo de todos los Mw)
        x_max_global = 0
        for mw in mw_list:
            path = fr'C:\Users\crist\Desktop\Trabajo\Database\AllTimeSeriesEta_mw_{mw}_filtrados.nc'
            if os.path.exists(path):
                ds = xr.open_dataset(path)
                eta_vals = ds['FlowDepth'].sel(grid_npoints=grilla).values
                eta_vals = eta_vals[~np.isnan(eta_vals)]
                if len(eta_vals) > 0:
                    x_max_global = max(x_max_global, np.nanmax(eta_vals))
                ds.close()
        x_max_global = np.ceil(x_max_global)

        # Grafica 
        for ax, mw in zip(axes, mw_list):
            ptha_dataset_path = fr'C:\Users\crist\Desktop\Trabajo\Database\AllTimeSeriesEta_mw_{mw}_filtrados.nc'
            if not os.path.exists(ptha_dataset_path):
                print(f"No existe dataset para Mw {mw}, se omite.")
                continue

            ds = xr.open_dataset(ptha_dataset_path)
            ids_dataset = ds['ScenarioId'].values
            eta_total = ds['FlowDepth'].sel(grid_npoints=grilla).values
            eta_total = eta_total[~np.isnan(eta_total)]
            if len(eta_total) == 0:
                print(f"No hay datos válidos en Mw {mw} para {ciudad}.")
                continue

            # CDF Monte Carlo
            eta_total_sorted = np.sort(eta_total) # Ordena los valores
            cdf_total = np.linspace(0, 1, len(eta_total_sorted))
            ax.plot(eta_total_sorted, cdf_total, label='Monte Carlo', color=colores_mc, linewidth=2)

            # CDF SROMs
            for j, slip_point in enumerate(slip_points):
                for i, n in enumerate(num_escenarios):
                    archivo = f'SROM_resultados_{mw}_{n}_{slip_point}.npz'
                    if os.path.exists(archivo):
                        data = np.load(archivo)
                        escenarios = data['scenarios']
                        probs = data['probabilities']
                        eta, prob = extraer_flowdepth(escenarios, probs, ids_dataset, ds, grilla)
                        if len(eta) > 0:
                            orden = np.argsort(eta) # Ordena los valores a partir de eta
                            ax.plot(np.sort(eta), np.cumsum(prob[orden]),  # calcula la probabilidad acumulada
                                    label=f'ID: {slip_point}, m={n}',
                                    color=colores_por_slip[j % len(colores_por_slip)],
                                    linestyle=estilos[i % len(estilos)],
                                    linewidth=1.2)

            # Configuración de ejes 
            ax.set_xlim(0, x_max_global)
            ax.set_ylim(-0.02, 1.05)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_title(f"Mw {mw / 10}", fontsize=11)
            ax.set_xlabel("Profundidad de flujo [m]", fontsize=10)
            ax.set_ylabel("Probabilidad acumulada", fontsize=10)
            ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

        plt.suptitle(f"{ciudad}", fontsize=14, y=0.94)
        plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=0.9, h_pad=0.9)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"PDF generado: {pdf_filename}")