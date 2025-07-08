## Importing Libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import cosine
import scipy.io.wavfile as wav
from scipy.io import wavfile

## T student test function for peoples
def t_student_test(file_csv):
    ## estructure of the csv file: archivo,genero,edad
    df = pd.read_csv(file_csv)
    ## Get the unique values of the 'genero' column
    generos = df['genero'].unique()
    results = {}
    for genero in generos:
        df_genero = df[df['genero'] == genero]
        t_stat, p_value = stats.ttest_1samp(df_genero['edad'], df['edad'].mean())
        results[genero] = (t_stat, p_value)
    ## Print the results
    for genero, (t_stat, p_value) in results.items():
        print(f"Genero: {genero}, T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    return results

## Fourier Transform function from dataframe
def fourier_transform(audio_data, sample_rate=44100):
    n = len(audio_data)
    fft_vals = np.fft.fft(audio_data)
    fft_vals = np.abs(fft_vals[:n // 2 + 1]) ** 2  # energía y mitad positiva
    freq = np.fft.fftfreq(n, d=1/sample_rate)[:n // 2 + 1]
    return freq, fft_vals

def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)

def mel_to_hz(m):
    return 700 * (10**(m / 2595) - 1)
## create the mel filter bank
def mel_filter_bank(sample_rate=44100, n_fft=400, n_filters=26):
    f_min = 0
    f_max = sample_rate / 2
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filter_bank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        # Lado izquierdo del triángulo
        for j in range(left, center):
            filter_bank[i - 1, j] = (j - left) / (center - left)

        # Lado derecho del triángulo
        for j in range(center, right):
            filter_bank[i - 1, j] = (right - j) / (right - center)
    return filter_bank

def fft_window(audio_data, sample_rate=44100, solape_percente=50, window_ms=25, filter='hamming', filtro_mel=None):
    """
    Plots the FFT of the audio data with windowing and overlap.
    Parameters:
        audio_data: np.array, audio signal
        sample_rate: int, sampling rate in Hz
        solape_percente: int, percentage of overlap between windows (0-100)
        window_ms: int, window length in milliseconds
        filter: str, window function name (e.g., 'hamming', 'hann')
    """
    if filtro_mel is None:
        n_fft = int(sample_rate * window_ms / 1000)
        filtro_mel = mel_filter_bank(sample_rate=sample_rate, n_fft=n_fft, n_filters=26)
    window_length = int(sample_rate * window_ms / 1000)
    step = int(window_length * (1 - solape_percente / 100))
    window_func = getattr(np, filter) if hasattr(np, filter) else np.hamming
    n_windows = (len(audio_data) - window_length) // step + 1
    energy_fft_values = []
    for i in range(n_windows):
        start = i * step
        end = start + window_length
        segment = audio_data[start:end]
        if len(segment) < window_length:
            break
        windowed = segment * window_func(window_length)
        freq, fft_vals = fourier_transform(windowed, sample_rate)
        ## calculate the mean values of the FFT
        n_fft = int(sample_rate * window_ms / 1000)  
        energy_by_band = filtro_mel @ fft_vals
        energy_fft_values.append(energy_by_band)        
    energy_fft_values = np.array(energy_fft_values)
    ## add name to return
    
    vector_medio = np.mean(energy_fft_values, axis=0)   
    return vector_medio

def analize_data(directory, cut=30):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            audio_files.append(os.path.join(directory, filename))
    ##Open Audio Files
    names_audios = []
    vector_medio_fft_values = []
    for audio_file in audio_files:
        sample_rate, data = wav.read(audio_file)
        ## Normalize the audio data
        data = data / np.max(np.abs(data))
        ## Convertir audios a mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        ## Cut all audios to cuts exactly
        if len(data) > sample_rate * cut:
            data = data[:sample_rate * cut]
        elif len(data) < sample_rate * cut:
            data = np.pad(data, (0, sample_rate * cut - len(data)), 'constant')
        vector_medio=fft_window(data, sample_rate=sample_rate, solape_percente=50, window_ms=25, filter='hamming')
        vector_medio_fft_values.append(vector_medio)
        names_audios.append(os.path.basename(audio_file))
    return vector_medio_fft_values, names_audios


def distancia_coseno(audio_claro, vector_medio_fft_values_propio, vector_medio_fft_values_otros):
    ## Comparacion entre mi audio claro y los audios test
    distancias_test_vs_ref = [cosine(v, audio_claro) for v in vector_medio_fft_values_propio]
    distancias_test_vs_impostores = [
    cosine(v_test, v_imp)
    for v_test in vector_medio_fft_values_propio
    for v_imp in vector_medio_fft_values_otros
    ]
    plt.figure(figsize=(10, 5))
    
    plt.hist(distancias_test_vs_impostores, bins=10, alpha=0.7, label='Test vs. Impostores')
    plt.title("Distancias Coseno entre vectores de audio")
    plt.xlabel("Distancia coseno")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 5))
    
    plt.hist(distancias_test_vs_ref, bins=10, alpha=0.7, label='Test vs. Referencia')
    plt.title("Distancias Coseno entre vectores de audio")
    plt.xlabel("Distancia coseno")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True)
    plt.show()
    return distancias_test_vs_ref, distancias_test_vs_impostores

def Prueba_estadistica(distancias_test_vs_ref, distancias_test_vs_impostores):
    stat, p = mannwhitneyu(distancias_test_vs_ref, distancias_test_vs_impostores)
    print(f"U = {stat:.4f}, p-value = {p:.4f}")

    if p < 0.05:
        print("Las distancias son significativamente diferentes.")
    else:
        print("No hay diferencia estadística clara.")

def umbral(distancias_test_vs_ref, distancias_test_vs_impostores):
    #promediar las distancias
    max_test = np.mean(distancias_test_vs_ref)
    min_impostores = np.mean(distancias_test_vs_impostores)
    umbral = (max_test + min_impostores) / 2
    return umbral



# Funcion de verificacion final

def verificar_audio(ruta_audio, vector_referencia, umbral=0.2):
    vector, nombre = analize_data(ruta_audio)
    distancias_test_vs_ref = [cosine(v, vector[0]) for v in vector_referencia]
    max_test = np.mean(distancias_test_vs_ref)
    test = (max_test) 
    if (umbral>=test):
        print("El audio es de la persona "+ str(test))
    else:
        print("El audio no es de la persona "+ str(test))



t_student_test('metadata_impostores.csv')
"""
Genero: F, T-statistic: 0.4759, P-value: 0.6455
Genero: M, T-statistic: -0.5903, P-value: 0.5695

## Interpretación de los resultados
Los resultados indican que no hay diferencias 
significativas en la edad entre los géneros F y M,
ya que los valores p son mayores que 0.05, 
por lo que estan pareados en edad.
"""
vector_medio_fft_values_otros, names_audios_otros= analize_data('Audios')
vector_medio_fft_values_propio, names_audios_propio= analize_data('Audios_propios', cut=20)
# Buscar posicion del elemento llamado Audio_claro.wav
indice_audio_claro = names_audios_propio.index('Audio_claro.wav')

# separar grabacion de referecia con nombre Audio_claro dentro de Audios_propios
audio_claro=vector_medio_fft_values_propio[indice_audio_claro]
nombre_claros=names_audios_propio[indice_audio_claro]

# Eliminar el Audio_claro de la otra lista
del vector_medio_fft_values_propio[indice_audio_claro]
del names_audios_propio[indice_audio_claro]

# Realizar pruebas estadisticas
distancias_test_vs_ref, distancias_test_vs_impostores = distancia_coseno(audio_claro, vector_medio_fft_values_propio, vector_medio_fft_values_otros)
Prueba_estadistica(distancias_test_vs_ref, distancias_test_vs_impostores)
"""
Se realizó una prueba estadística de Mann–Whitney
U entre las distancias coseno obtenidas del conjunto
de prueba frente al audio de referencia, y las distancias
frente al conjunto de impostores. El valor U = 85 y un p-value < 0.0001
indican una diferencia altamente significativa entre ambas distribuciones, 
confirmando que el sistema discrimina adecuadamente entre la identidad real y los impostores.
"""

# Concluir umbral de descicion
descision=umbral(distancias_test_vs_ref, distancias_test_vs_impostores)
print(descision)
"""
Se establece el umbral de desicion para distancias inferiores a 0.19
al realizar la prueba coseno
"""

# Verificacion despues de definir umbral
verificar_audio("Pruebas", vector_medio_fft_values_propio, umbral=descision+0.02)
