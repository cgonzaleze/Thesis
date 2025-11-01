# Código análogo al de FlowDepth, genera CDFs de deslizamientos a partir de la selección de escenarios realizada en PTHA

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, MaxNLocator

# CONFIGURACIÓN
mw_list = [86, 88, 90, 92]  # Magnitudes a graficar
grillas_slip = [5545, 5975, 4532, 2978, 2364, 1389]  # puntos de slip relevantes por ciudad

# Ahora los diccionarios llevan como llave el deslizamiento, debido a que ese es el valor de interés para graficar la CDF por ciudad
ciudades_keys_grilla = {
    5545: 'Iquique',
    5975: 'Antofagasta',
    4532: 'La Serena',
    2978: 'Valparaíso',
    2364: 'Talcahuano',
    1389: 'Valdivia'
}

# Slips por ciudad
slips_por_ciudad = {
    'Iquique': [112, 46, 13, 386, 26, 444],
    'Antofagasta': [112, 46, 13, 386, 26, 444],
    'La Serena': [112, 46, 13, 386, 26, 444],
    'Valparaíso': [22, 127, 386, 27, 30, 69],
    'Talcahuano': [26, 444, 27, 112, 46, 13],
    'Valdivia': [30, 27, 26, 112, 46, 13]
}

num_escenarios = [25, 50, 100, 200]

# Colores y estilos
color_mc = 'blue'
colores_por_grilla = ['orange', 'purple', 'grey', 'brown', 'green', 'black']  # un color por grilla
estilos_por_srom = ['-', '--', '-.', ':']  # un estilo por tamaño SROM

# FUNCIONES
def extraer_slips_desde_ids(scenarios, probs, ids_slip, slip_vector):
    """Extrae los valores de slip correspondientes a los escenarios válidos."""
    slips = []
    probs_validas = []
    for i, esc in enumerate(scenarios):
        if esc in ids_slip:
            idx = np.where(ids_slip == esc)[0][0]
            slips.append(slip_vector[idx])
            probs_validas.append(probs[i])
    return np.array(slips), np.array(probs_validas)


# LOOP PRINCIPAL
for grilla_slip, ciudad in ciudades_keys_grilla.items():

    if ciudad not in slips_por_ciudad:
        print(f"No hay lista de slips definida para {ciudad}.")
        continue

    slips_ciudad = slips_por_ciudad[ciudad]
    pdf_filename = f"CDFs_Slip_{ciudad}.pdf"

    with PdfPages(pdf_filename) as pdf:
        # Crea figura 2x2 con ejes independientes
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
        axes = axes.flatten()

        # Determina límites X comunes
        x_max_global = 0
        for mw in mw_list:
            path = fr'C:\Users\crist\Desktop\Trabajo\Database\TodosQuakes_mw_{mw}_filtrados.nc'
            if os.path.exists(path):
                ds_temp = xr.open_dataset(path)
                slips_temp = ds_temp['slip'].fillna(0.0).values / 100
                x_max_global = max(x_max_global, np.nanmax(slips_temp))
                ds_temp.close()
        x_max_global = np.ceil(x_max_global)

        # Grafica por magnitud
        for ax, mw in zip(axes, mw_list):

            slip_dataset_path = fr'C:\Users\crist\Desktop\Trabajo\Database\TodosQuakes_mw_{mw}_filtrados.nc'
            if not os.path.exists(slip_dataset_path):
                print(f"No existe dataset de SLIP para Mw {mw}, se omite.")
                continue

            ds_slip = xr.open_dataset(slip_dataset_path)
            ds_slip['slip'] = ds_slip['slip'].fillna(0.0)
            ids_slip = ds_slip['ScenarioId'].values.astype(int)

            # Monte Carlo
            slip_vector = ds_slip['slip'].values[grilla_slip, :] / 100 # para pasarlo a metros
            slip_vector = slip_vector[~np.isnan(slip_vector)]
            slip_total_sorted = np.sort(slip_vector)
            cdf_total = np.linspace(0, 1, len(slip_total_sorted))
            ax.plot(slip_total_sorted, cdf_total, label='Monte Carlo', color=color_mc, linewidth=2)

            # SROM aplicado a FlowDepth
            for j, ptha_grilla in enumerate(slips_ciudad):
                color = colores_por_grilla[j % len(colores_por_grilla)]  # color por grilla
                for i, n in enumerate(num_escenarios):
                    archivo = f'SROM_PTHA_resultados_{mw}_{n}_{ptha_grilla}.npz'
                    if os.path.exists(archivo):
                        data = np.load(archivo)
                        escenarios = data['scenarios'].astype(int)
                        probs = data['probabilities']
                        slips, prob = extraer_slips_desde_ids(escenarios, probs, ids_slip, slip_vector)
                        if len(slips) > 0:
                            orden = np.argsort(slips)
                            ax.plot(
                                np.sort(slips), np.cumsum(prob[orden]),
                                label=f'ID: {ptha_grilla}, m = {n}',
                                color=color,
                                linestyle=estilos_por_srom[i % len(estilos_por_srom)],
                                linewidth=1.2
                            )

            # --- Configuración de ejes ---
            ax.set_xlim(0, x_max_global)
            ax.set_ylim(-0.02, 1.05)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_title(f"Mw {mw / 10}", fontsize=11)
            ax.set_xlabel("Deslizamiento [m]", fontsize=10)
            ax.set_ylabel("Probabilidad acumulada", fontsize=10)

            # 🔹 Leyenda siempre al borde derecho fuera del gráfico
            ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

        plt.suptitle(f"{ciudad}", fontsize=14, y=0.94)
        plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=0.9, h_pad=0.9)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"PDF generado: {pdf_filename}")
