# Este código se encarga de aplicar el SROM (Grigoriu et al., 2009) a los escenarios filtrados, mediante SROMPy (Warner, 2018) y es el código
# principal para mi memoria de título. A partir de los valores de deslizamiento en un punto latitud-longitud de la grlla, se obtiene la curva
# de distribución de probabilidad acumulada marginal para el deslizamiento, con el respectivo error de la curva de CDF, Momentos de orden y 
# Correlación. Se compara la curva de SROM con la Monte-Carlo, y se obtiene el vector de samples y los escenarios correspondientes. 

# El error utilizado es el SSE (Suma de Errores Cuadrados) entre ambas curvas, y se optimizan los pesos en cada iteración para 
# obtener la mejor aproximación posible.

# Asegurarse de instalar todas las librerías para ejecutar el código, con las librerías está ready to go (recordar ajustar los directorios)

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

from SROMPy.srom import SROM
from SROMPy.target import SampleRandomVector
from SROMPy.optimize import ObjectiveFunction
from SROMPy.postprocess import Postprocessor
from scipy.stats import norm

# Tamaños de SROM del estudio
lista_srom_sizes = [25, 50, 100, 200]
# Lista de puntos a analizar: Arica, Iquique, Antofagasta, La Serena, Valpo, Viña, San Antonio, Talcahuano, Concepción, Coronel, Valdivia, Pto Montt, Ancud
lista_point_index = [112, 46, 13, 124, 22, 127, 386, 26, 444, 27, 30, 69, 33] # Puntos de métrica a analizar (puntos de costa de cada una de las ciudades comentadas)
# Se recorre la lista de puntos
for point_index in lista_point_index:
    # Se recorre la lista de tamaños de muestra a seleccionar
    for sromsize in lista_srom_sizes:
        # Se recorren las magnitudes de todo el catálogo de terremotos
        for mw in range(75,96):
            # Dataset con los parámetros estadísticos del escenario Mw = 9.2 (ejemplo)
            dataset = xr.open_dataset(fr'C:\Users\crist\Desktop\Trabajo\Database\AllTimeSeriesEta_mw_{mw}_filtrados.nc')
            dataset['FlowDepth'] = dataset['FlowDepth'].fillna(0.0) # Se rellenan posibles valores NaN para evitar errores en la aplicación de SROMPy
            # Valores NaN solo indican la ausencia de deslizamiento en ese punto (o posibles deslizamientos negativos por expansión K-L, dependiendo de la metodología), 
            # por lo que se utiliza 0 para aportar a la estadística. Se verifica que el sromsize no sea mayor a la capacidad del dataset, 
            # así no se detiene el código en un punto no deseado.
            if sromsize > len(dataset['Scenario']):
                print("Tamaño del vector de muestras es mayor a lo que admite el dataset, pasando a la siguiente magnitud.")
                continue
            else:
                # Vector de deslizamientos en el punto point_index
                metric_vector = dataset['FlowDepth'].values[:, point_index]
                # Parámetros del SROM
                sromdim = 1  # Dimensión del SROM (dimensión del vector de deslizamientos)
                srom = SROM(sromsize, sromdim)
                # Inicialización de probabilidades uniformes (Primera iteración)
                probabilities = np.ones(sromsize) / sromsize
                srom.set_params(metric_vector, probabilities)
                # Máximo momento de orden
                max_order = 4 # Máximo orden de momentos a calcular (criterio estadístico)
                # Número de grid points para la gráfica de la CDF
                if mw > 90:
                    num_grid_points = 10000 # Ojo, esta modificación la hice sólo porque el bucle lanzaba error por el número de grid_points
                else: # Mayor num_grid_points, mas puntos en el grafico (mayor coste computacional, ojo)
                    num_grid_points = 100000 # Sólo porque no da error
                # Crear la instancia de SampleRandomVector con el vector de deslizamientos
                samples = metric_vector.reshape(-1, 1)  # Convertir a un arreglo (n_samples, 1)
                random_vector = SampleRandomVector(samples, max_order)
                # Parámetros iniciales para los factores de ponderación
                weights = [1, 1, 1]  # [CDF, Momentos, Correlación]
                tolerance = 0.1 # Diferencia aceptable entre los errores (Igualar órdenes de magnitud entre error de CDF y momento de orden q)
                max_iterations = 10 # Máximo número de iteraciones: 1 PARA PROBAR, 3 GOD, 10 PARA TESIS
                Num_test_samples = 10 # 10000 pero en 10 funciona igual, ya que el current optimal objective NO VARIA, con pocos es igual que con muchos num test samples!! (PROBAR PARA VERIFICAR)
                # Se crea una lista que guarde cada uno de los weights de cada iteracion
                Lista_weights = [] # Se crea una lista que guarda los weights de cada iteración, esto en caso de que la última iteración no sea la mejor
                weight_iteracion_final = [] # Se crea una lista que guarda los weights óptimos del modelo.
                Registro_errores = [] # Se crea una lista que guarda los errores de cada iteración, al final se verificará que el promedio de la última iteración
                # sea el menor, sino lo es, se busca la iteración donde este sea el menor, y se busca el vector weights de dicha iteración, realizando
                # finalmente el SROM con ese vector de weights, optimizando así el proceso de selección de escenarios.
                # Iteración para ajustar los factores de ponderación
                for iteration in range(max_iterations):
                    print(f"Iteración {iteration + 1} con pesos {weights}")
                    # Se agrega el primer vector weights a la lista, para tener el conteo completo de las iteraciones
                    Lista_weights.append(weights)
                    # Optimización con los pesos actuales
                    srom.optimize(
                        target_random_variable=random_vector,
                        weights=Lista_weights[iteration],
                        num_test_samples=Num_test_samples, # 10000 num_test_samples está bueno pero sube considerablemente el tiempo de ejecución
                        error='SSE',
                        max_moment=max_order,
                        cdf_grid_pts=num_grid_points,
                        tolerance=None,
                        options=None,
                        method=None,
                        joint_opt=None
                    )
                    # Se calcula el espacio para graficar las CDF
                    x_grid = np.linspace(0, metric_vector.max(), num_grid_points)  # 100 puntos en el rango de 0 al máximo valor 
                    # de deslizamiento del vector de deslizamientos
                    # Se calculan la CDF para el vector de muestras reducidas
                    srom.compute_cdf(x_grid)
                    srom.compute_corr_mat()
                    srom.compute_moments(max_order)
                    # Obtener las muestras y probabilidades ajustadas
                    srom_samples, probabilities = srom.get_params()
                    # Se calculan la CDF para el vector objetivo
                    random_vector.compute_cdf(x_grid)
                    random_vector.compute_correlation_matrix()
                    random_vector.compute_moments(max_order)
                    random_vector.draw_random_sample(sromsize)
                    random_vector.generate_cdfs()
                    random_vector.generate_correlation()
                    random_vector.generate_moments(max_order)
                    random_vector.generate_statistics(max_order)
                    random_vector.get_plot_cdfs()
                    Lista_errores = srom.errores_entre_curvas # Nomenclatura: ERROR DE MOMENTO, ERROR DE CDF, ERROR DE CORRELACIÓN, Y EL PROMEDIO DE LOS 3.
                    Registro_errores.append(Lista_errores) # Se guarda el error de la iteración en la lista de errores, para luego verificar cuál es el menor.
                    print(srom.errores_entre_curvas)
                    # Llamamos entonces sabiendo que es una lista, a los errores correspondientes en cada uno de los índices
                    cdf_error = Lista_errores[1]
                    moment_error = Lista_errores[0]
                    corr_error = Lista_errores[2]
                    mean_error = Lista_errores[3]
                    # Obtenemos la suma de los errores para tener el total
                    total_errors = cdf_error + moment_error + corr_error
                    # Porcentaje de peso para cada uno
                    cdf_percent = cdf_error / total_errors
                    moments_percent = moment_error / total_errors
                    corr_percent = corr_error / total_errors
                    # Se redefine el vector de weights a partir de los errores de la primera iteración
                    weights = [cdf_percent, moments_percent, corr_percent]
                    # Verificar convergencia
                    if abs(np.log10(cdf_error) - np.log10(moment_error)) <= tolerance:
                        print("Errores ajustados al mismo orden de magnitud.")
                        weight_iteracion_final = Lista_weights[iteration]
                        break 
                    elif iteration+1 == max_iterations:
                        print("Se alcanzó el número máximo de iteraciones.")
                        print("Buscando la mejor iteración..")
                        # Mi idea acá es tratar de verificar que ya, se cumplieron todas las iteraciones, no se cumplió el criterio de que los errores sean
                        # del mismo orden de magnitud, pero tengo los errores y los pesos de cada iteración, puedo ver cuál fue la iteración con el menor error
                        # promedio, buscarlo en los weights y asignar ese vector de weights a la variable weights, y luego hacer el SROM con ese vector. Finalmente,
                        # el ciclo terminará en este break, y se procederá a graficar la CDF y los errores de la iteración con el menor error promedio.
                        lista_mean_errors = [] # Se crea una lista que guardará los errores promedio de cada iteración.
                        for i in range(len(Registro_errores)):
                            lista_mean_errors.append(Registro_errores[i][3]) # Se guarda el error promedio de cada iteración en una lista.
                        lista_mean_errors.pop(0) # Borra la primera iteración ya que los weights de esta no son válidos
                        Lista_weights.pop(0) # Idem
                        indice_mejor_iteracion = lista_mean_errors.index(min(lista_mean_errors))  # Índice de la mejor iteración
                        weight_iteracion_final = Lista_weights[indice_mejor_iteracion]
                        print(f"Iteración {indice_mejor_iteracion + 2} con pesos {weight_iteracion_final}") # +2 para agregarle también la iteración borrada
                        # Optimización con los pesos finales
                        srom.optimize(
                        target_random_variable=random_vector,
                        weights=weight_iteracion_final,
                        num_test_samples=Num_test_samples, # 10000 num_test_samples
                        error='SSE',
                        max_moment=max_order,
                        cdf_grid_pts=num_grid_points,
                        tolerance=None,
                        options=None,
                        method=None,
                        joint_opt=None
                        )
                        # Se calcula el espacio para graficar las CDF
                        x_grid = np.linspace(0, metric_vector.max(), num_grid_points)
                        # Se calculan la CDF para el vector de muestras reducidas
                        srom.compute_cdf(x_grid)
                        srom.compute_corr_mat()
                        srom.compute_moments(max_order)
                        # Obtener las muestras y probabilidades ajustadas
                        srom_samples, probabilities = srom.get_params()
                        # Se calculan la CDF para el vector objetivo
                        random_vector.compute_cdf(x_grid)
                        random_vector.compute_correlation_matrix()
                        random_vector.compute_moments(max_order)
                        random_vector.draw_random_sample(sromsize)
                        random_vector.generate_cdfs()
                        random_vector.generate_correlation()
                        random_vector.generate_moments(max_order)
                        random_vector.generate_statistics(max_order)
                        random_vector.get_plot_cdfs()
                        Lista_errores = srom.errores_entre_curvas
                        Registro_errores.append(Lista_errores)
                        print(srom.errores_entre_curvas)
                        break
                # Se realiza la comparación entre ambas CDF
                comp = Postprocessor(srom, random_vector)
                comp.compare_cdfs(variable='x', plot_dir='.', plot_suffix=fr'CDFcompare_PTHA_{mw}_{sromsize}_{point_index}', show_figure=False, save_figure=True, variable_names=None, x_limits=None)
                comp.compute_moment_error(max_moment=max_order)
                comp.generate_cdf_grids(cdf_grid_pts=num_grid_points)
                # Se crea la lista que guardará los scenarios correspondientes a los samples.
                lista_scenarios = []
                for metric in srom_samples:
                    for id_scenario in range(dataset['ScenarioId'].shape[0]): # Se recorren los escenarios para tener el registro de la ID correspondiente al slip.
                        metric_dataset = dataset['FlowDepth'].values[id_scenario, point_index]
                        if metric == metric_dataset and dataset['ScenarioId'].values[id_scenario] not in lista_scenarios: # Si el slip de samples
                            # es igual al slip del dataset  y la id de su escenario no está ya dentro de la lista_scenarios, se agrega. Así evitamos repetir ID's.
                            lista_scenarios.append(dataset['ScenarioId'].values[id_scenario]) # Agrega la ID del escenario a la lista de scenarios.
                            break
                if len(lista_scenarios) == len(srom_samples): # Si la longitud de la lista de scenarios es igual a la de los samples, se puede proceder a graficar.
                    print("Se han encontrado todos los escenarios correspondientes a los samples.")
                else:
                    print("Existe un error de dimensiones entre samples y escenarios")
                # Ya teniendo los samples seleccionados por SROM y las ID de los respectivos escenarios, podemos guardar estos datos relevantes para poder
                # replicar el procedimiento desde los dataset de PTHA, pudiendo contrastar así la construcción de la CDF mediante los terremotos y mediante
                # las métricas de tsunami.
                # Con esto está listo el grueso de la aplicación del SROMPy y del código en sí, se sigue con el procedimiento para guardar los resultados de
                # los samples y sus escenarios correspondientes, además de la probabilidad de ocurrencia de cada uno de ellos y de la gráfica de la CDF.
                # Se asegura que todos los arrays sean 1D.
                srom_samples = np.array(srom_samples).flatten()
                probabilities = np.array(probabilities).flatten()
                lista_scenarios = np.array(lista_scenarios).flatten()
                lista_errores = np.array(Lista_errores).flatten() # Momentos, CDF, correlacion y promedio. (EN ESE ORDEN)
                lista_weights = np.array(weight_iteracion_final).flatten()
                # Se guarda en archivo comprimido de NumPy.
                np.savez(fr'SROM_PTHA_resultados_{mw}_{sromsize}_{point_index}.npz',
                        slip_samples=srom_samples,
                        probabilities=probabilities,
                        scenarios=lista_scenarios,
                        errors=lista_errores,
                        weights=lista_weights)
                # Se imprime el directorio actual donde se ejecuta el script (por seguridad), así se sabe donde se guardan los archivos creados si es que el código se ejecuta no considerando la carpeta definida.
                print("Directorio actual:", os.getcwd())
                # Se verifica que los datos estén bien formados antes de guardar y tengan el mismo len.
                print("srom_samples:", srom_samples.shape)
                print("probabilities:", probabilities.shape)
                print("lista_scenarios:", lista_scenarios.shape)
                print("lista_errores:", lista_errores.shape)
                print("lista_weights:", lista_weights.shape)
                print(lista_errores)
                print(lista_weights)
                # Con esto está terminado el código, se grafica la CDF y se guardan los resultados para ser llamados en PTHA.