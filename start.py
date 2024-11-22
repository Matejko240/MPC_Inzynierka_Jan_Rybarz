import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from modelODE import model_ode
from mpc import optimize_parameters
import random
import csv
import os

def generate_unique_filename(base_name, extension):
    filename = f"{base_name}{extension}"
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_name}({counter}){extension}"
        counter += 1
    return filename

def save_to_csv(filename, solver_time, optimized_parameters, optimize_keys,state):
    """Funkcja zapisująca dane do pliku CSV.
    
    Args:
        filename (str): Ścieżka do pliku CSV.
        solver_time (float): Czas symulacji.
        optimized_parameters (dict): Zoptymalizowane parametry.
        optimize_keys (list): Klucze parametrów do optymalizacji.
        reference_trajectory (list): Trajektoria referencyjna.
        optimized_trajectory (list): Trajektoria zoptymalizowanych parametrów.
    """
    with open(filename, "a", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        flat_state = np.array(state).flatten().tolist()
        # Zapisujemy czas i zoptymalizowane parametry
        row = [solver_time] + [optimized_parameters[key] for key in optimize_keys] + flat_state
        csvwriter.writerow(row)
        """
        # Zapisujemy 10 wierszy dla reference_trajectory
        csvwriter.writerow(["Trajektoria referencyjna:"])
        for ref in reference_trajectory:
            csvwriter.writerow(ref)
        
        # Zapisujemy 10 wierszy dla optimized_trajectory
        csvwriter.writerow(["Trajektoria zoptymalizowanych parametrow:"])
        for opt in optimized_trajectory:
            csvwriter.writerow(opt)
        """
            
def main():
    """_summary_

    Returns:
        _type_: _description_
    """
    
        # Tworzenie folderu "dane", jeśli nie istnieje
    if not os.path.exists("dane"):
        os.makedirs("dane")
        
    tEnd = 10
    sample_time = 0.5
    lambda_e = 1.0 # współczynnik błędu qr_d1
    lambda_u = 1.0 # współczynnik błędu qr_d2
    optimize_keys = ['l1','l2','l3']  # Lista parametrów do optymalizacji
    N_pred = 10
    np.random.seed(123456789)

    Pi = np.pi

    m1 = 60.0 # kg, masa pierwszego segmentu
    m2 = 50.0 # kg, masa drugiego segmentu
    m3 = 40.0 # kg, masa trzeciego segmentu

    mc = m1 + m2 + m3

    l1 = 2.0 # m, długość pierwszego ramienia
    l2 = 4.0 # m, długość drugiego ramienia
    l3 = 3.0  # m, długość trzeciego ramienia

    a = 0
    b = 0
    c = 0

    g = -9.81

    Tv = 0.01
    Ts = 0.01
    Tk = 5

    Kp = 2000 * np.eye(3)  # Wzmocnienie proporcjonalne
    Kd = 200 * np.eye(3) # Wzmocnienie różniczkowe

    init_qr_d1 = np.array([0, 0, 0]) # Początkowa prędkość w stawach
    init_qr = np.array([0, 0, 0]) # Początkowa pozycja w stawach

    qr = init_qr


    parameters = {
        'm1': m1,
        'm2': m2,
        'm3': m3,
        'mc': mc,
        'l1': l1,
        'l2': l2,
        'l3': l3,
        'a': a,
        'b': b,
        'c': c,
        'g': g,
        'Tv': Tv,
        'Ts': Ts,
        'Tk': Tk,
        'Kp': Kp,
        'Kd': Kd
    }

    ic = np.zeros(6)
    ic[0:3] = init_qr_d1
    ic[3:6] = init_qr

    solver = ode(model_ode)
    solver.set_integrator('vode', rtol=1e-2, atol=1e-3,nsteps=50000)
    solver.set_f_params(parameters)
    solver.set_initial_value(ic, 0)

    t = [0]
    youtput = [ic]


    
    init_parameters=parameters.copy()
    
    # Przypisanie losowych wartości dla parametrów z optimize_keys
    for key in optimize_keys:
        if key in parameters:
            # Przypisanie losowej wartości w określonym zakresie
            parameters[key] = random.uniform(0.1, 1000.0) 
    print("Current parameters (parametry obecne):", parameters)
    
    # Nazwa bazowa pliku
    base_name = f"dane/mpc_log_{'_'.join(optimize_keys)}_tEnd{tEnd}_dt{sample_time}_N_pred{N_pred}_lambdaU{lambda_u}_lambdaE{lambda_e}"
    
    # Generowanie unikalnej nazwy pliku CSV
    filename = generate_unique_filename(base_name, ".csv")
    print(f"Nazwa pliku: {filename}")
    
    optimized_values_series = {key: [] for key in optimize_keys}  # Inicjalizacja przechowywania wartości

    # Tworzenie pliku CSV i zapis nagłówka
    with open(filename, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        parameter_names = ["czas"] + [f"parametr_{key}" for key in optimize_keys] + ["qr_d2[0]","qr_d2[1]","qr_d2[2]","qr_d1[0]","qr_d1[1]","qr_d1[2]"]
        # trejectory = ["qr_d2[0]","qr_d2[1]","qr_d2[2]","qr_d1[0]","qr_d1[1]","qr_d1[2]"]
        real_values =[","] + [f"wartosc_rzeczywista_{key} = {init_parameters[key]}" for key in optimize_keys]
        csvwriter.writerow(parameter_names )
        # csvwriter.writerow(trejectory)
        csvwriter.writerow(real_values)
        csvwriter.writerow(["Czas symulacji oraz wartosci optymalizowanych parametrow w kazdej iteracji"])
     
     
     
        # Optymalizacja parametrów dla t = 0.0
    optimized_parameters = optimize_parameters(
        initial_parameters=init_parameters,
        current_parameters=parameters,
        current_state=solver.y,
        t=solver.t,
        N_pred=N_pred,
        dt=sample_time,
        lambda_u=lambda_u,
        lambda_e=lambda_e,
        optimize_keys=optimize_keys
    )

    # Aktualizacja parametrów po optymalizacji początkowej i przekazanie ich do solvera
    parameters.update(optimized_parameters)
    solver.set_f_params(init_parameters)

    # Zapisanie wyników optymalizacji dla t = 0.0 do serii oraz pliku CSV
    for key in optimize_keys:
        optimized_values_series[key].append(optimized_parameters[key])

    save_to_csv(filename, solver.t, optimized_parameters, optimize_keys, solver.y)

                    
    while solver.successful() and solver.t < tEnd:
        # 1. Krok integracji, aby uzyskać nowy stan systemu
        solver.integrate(solver.t + sample_time)
        t.append(solver.t)
        youtput.append(solver.y)

        # 2. Optymalizacja parametrów na podstawie nowego stanuW
        optimized_parameters= optimize_parameters(
            initial_parameters=init_parameters,
            current_parameters=parameters,
            current_state=solver.y,
            t=solver.t,
            N_pred=N_pred,
            dt=sample_time,
            lambda_u=lambda_u,
            lambda_e=lambda_e,
            optimize_keys=optimize_keys
        )

        # 3. Aktualizacja parametrów po optymalizacji i przekazanie ich do solvera
        parameters.update(optimized_parameters)
        #solver.set_f_params(parameters)

        # 4. Zbieranie wartości zoptymalizowanych parametrów dla wykresu
        for key in optimize_keys:
            optimized_values_series[key].append(optimized_parameters[key])

        # 5. Zapis danych do pliku CSV
        save_to_csv(filename, solver.t, optimized_parameters, optimize_keys, solver.y)

        
    t = np.array(t)
    youtput = np.array(youtput)

    tLength = len(t)
    additional = {
        'qchd': np.zeros((3, tLength)),
        'qchd_d1': np.zeros((3, tLength)),
        'qchd_d2': np.zeros((3, tLength)),
        'dets': np.zeros((2, tLength)),
        'k': np.zeros((3, tLength))
    }
    error_norm = np.zeros(tLength)
    error_d1_norm = np.zeros(tLength)
    for i in range(tLength):
        _, additionalNew,parameters = model_ode(t[i], youtput[i, :], parameters)
        additional['qchd'][:, i] = additionalNew['qchd']
        additional['qchd_d1'][:, i] = additionalNew['qchd_d1']
        additional['qchd_d2'][:, i] = additionalNew['qchd_d2']
        additional['dets'][:, i] = additionalNew['dets']
        additional['k'][:, i] = additionalNew['k']
        
        error_norm[i] = np.linalg.norm(additionalNew['error'])
        error_d1_norm[i] = np.linalg.norm(additionalNew['error_d1'])
    sim_data = {
        'time': t,
        'out_qr_d1': youtput[:, 0:3], # Prędkości w stawach
        'out_qr_d0': youtput[:, 3:6]  # Pozycje w stawach
    }
    print("blad pozycji",np.linalg.norm(additionalNew['error']))
    print("blad prędkości",np.linalg.norm(additionalNew['error_d1']))
    print("m1",parameters['m1'] )
    print("m2",parameters['m2'] )
    print("m3",parameters['m3'] )
    print("l1",parameters['l1'] )
    print("l2",parameters['l2'] )
    print("l3",parameters['l3'] )
    
    time_series = np.array(t)  # Czas symulacji

# Dopasowanie rozmiarów time_series i optimized_values_series
    min_length = min(len(time_series), len(list(optimized_values_series.values())[0]))
    time_series = time_series[:min_length]
    for key in optimized_values_series:
        optimized_values_series[key] = optimized_values_series[key][:min_length]

    # Rysowanie jednego wykresu dla wszystkich optymalizowanych parametrów
    plt.figure(figsize=(10, 6))  # Większy rozmiar wykresu dla lepszej widoczności

    # Dodanie wartości dla każdego parametru na jednym wykresie
    for key in optimize_keys:
        plt.plot(time_series, optimized_values_series[key], label=f"Optymalizowana wartość {key}", linewidth=2)
        plt.axhline(y=init_parameters[key], color='grey', linestyle='--', label=f"Rzeczywista wartość {key} = {init_parameters[key]}")

    # Ustawienia wykresu
    plt.yscale('log')  # Skala logarytmiczna na osi Y
    plt.xlabel('Czas [s]', fontsize=12)
    plt.ylabel('Wartości parametrów (skala logarytmiczna)', fontsize=12)
    plt.title('Porównanie wartości rzeczywistych i optymalizowanych parametrów', fontsize=14)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)  # Dodanie siatki dla lepszej czytelności
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)  # Przezroczyste tło legendy
    plt.tight_layout()  # Automatyczne dopasowanie elementów wykresu

    # Zapisanie i wyświetlenie wykresu
    plot_filename = generate_unique_filename(base_name, ".png")
    plt.savefig(plot_filename)
    plt.show()

    
if __name__ == "__main__":
    main()
