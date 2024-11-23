import numpy as np
from scipy.optimize import minimize
from scipy.integrate import ode
from modelODE import model_ode 
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution

def cost_function( predicted_trajectory, reference_trajectory, lambda_u,lambda_e, N_pred):
    """
    Funkcja kosztu penalizująca błąd trajektorii oraz sterowanie.
    :param parameters: Parametry modelu (długości i masy)
    :param predicted_trajectory: Przewidywana trajektoria manipulatora
    :param reference_trajectory: Trajektoria referencyjna
    :param control_effort: Nakłady na sterowanie
    :param lambda_u: Waga penalizująca sterowanie
    :return: Wartość funkcji kosztu
    """
    if predicted_trajectory.shape != reference_trajectory.shape:
        # Wypisanie ostrzeżenia o różnych rozmiarach
        print("Ostrzeżenie: Przewidywana i referencyjna trajektoria mają różne rozmiary!")
        
        # Dopasowanie rozmiarów wektorów
        min_length = min(predicted_trajectory.shape[0], reference_trajectory.shape[0])
        predicted_trajectory = predicted_trajectory[:min_length]
        reference_trajectory = reference_trajectory[:min_length]
        
    error_trajectory = predicted_trajectory[:, 3:6] - reference_trajectory[:, 3:6]
    control_effort = predicted_trajectory[:, 0:3] - reference_trajectory[:, 0:3]
    norm_error = np.sum(np.linalg.norm(error_trajectory, axis=1)) #suma norm euklidesowych
    control_effort_error = np.sum(np.linalg.norm(control_effort, axis=1)) #suma norm euklidesowych
    
    cost = (
    lambda_e * norm_error +
    lambda_u * control_effort_error
    ) / N_pred
    return cost


def calculate_trajectory(current_state, parameters, N_pred, dt,t):
    
 
    predicted_trajectory = np.zeros((N_pred, 3))  # Predicted x, y, z trajectory
    
    # Start with the current state
    solver = ode(model_ode)
    #  inne solvery: dopri5, vode, lsoda
    solver.set_integrator('lsoda', rtol=1e-5, atol=1e-6, nsteps=50000)
    solver.set_f_params(parameters)
    solver.set_initial_value(current_state, t)
    
    tEnd = t + N_pred * dt
    #print("tEnd",tEnd,"t",t,"N_pred",N_pred,"dt",dt)
    ic = np.zeros(6)
    ic[0:6] = current_state
    youtput = [ic]
    t = [t]
    while solver.successful() and solver.t < tEnd:
      
        solver.integrate(solver.t + dt)
        t.append(solver.t)
        youtput.append(solver.y)
        
    t = np.array(t)
    youtput = np.array(youtput)  

    return youtput


# Funkcja pomocnicza: aktualizuje słownik parametrów na podstawie zoptymalizowanych wartości
def update_parameters(parameters, keys, values):
    for key, value in zip(keys, values):
        parameters[key] = value
    return parameters
# Funkcja celu: zwraca koszt na podstawie zoptymalizowanych parametrów
def objective_function(optimized_values, current_parameters, current_state, t, N_pred, dt, lambda_u, lambda_e, optimize_keys, reference_trajectory):
    # Tymczasowy słownik parametrów z zoptymalizowanymi wartościami
    temp_parameters = update_parameters(current_parameters.copy(), optimize_keys, optimized_values)
    
    # Predykcja trajektorii na podstawie obecnych parametrów
    predicted_trajectory = calculate_trajectory(current_state, temp_parameters, N_pred, dt, t)
    
    # Obliczenie kosztu na podstawie funkcji kosztu
    cost = cost_function(predicted_trajectory, reference_trajectory, lambda_u, lambda_e, N_pred)
    
    # Wyświetlanie wartości dla celów debugowania
    print("Zoptymalizowane wartości:", optimized_values)
    print("Koszt:", cost)
    
    return cost

def optimize_parameters(initial_parameters, current_parameters, current_state, t, N_pred, dt, lambda_u, lambda_e, optimize_keys):
    
    # Tworzymy trajektorię referencyjną dla optymalizacji
    reference_trajectory = calculate_trajectory(current_state, initial_parameters, N_pred, dt, t)
    
    # Ustawienie ograniczeń dla każdego parametru w optimize_keys
    #bounds = [(1, 10)] * len(optimize_keys)
    bounds = []
    for key in optimize_keys:
        if key.startswith('m'):
            bounds.append((10, 100))  # Granice dla kluczy zaczynających się od 'm'
        elif key.startswith('l'):
            bounds.append((1, 10))  # Granice dla kluczy zaczynających się od 'l'
        else:
            bounds.append((1, 100))  # Domyślne granice dla innych kluczy (opcjonalnie)
  
    print(f"Klucz: {optimize_keys}, Granice: {bounds}")        
    # Wywołanie optymalizacji przy użyciu metody differential_evolution, używając objective_function bezpośrednio (wczesniej L-BFGS-B)
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(current_parameters, current_state, t, N_pred, dt, lambda_u, lambda_e, optimize_keys, reference_trajectory),
        strategy='best1bin', 
        maxiter=100, 
        tol=0.01, 
        disp=True, 
        seed=42, 
        workers=-1,          # dla wielowątkowości
        updating='deferred'  # jawne ustawienie strategii aktualizacji
    )
    
    # Zaktualizowanie parametrów na podstawie wyniku optymalizacji
    optimized_parameters = update_parameters(current_parameters, optimize_keys, result.x)
    
    # Obliczenie trajektorii dla zoptymalizowanych parametrów
    # optimized_trajectory = calculate_trajectory(current_state, optimized_parameters, N_pred, dt, t)
    
    return optimized_parameters