#from modelODE import model_ode
import numpy as np
from scipy.optimize import minimize
from effectorTrajectoryGenerator3D import effectorTrajectoryGenerator3D
from scipy.integrate import ode
from modelODE import model_ode 
from scipy.optimize import Bounds
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



def optimize_parameters(initial_parameters, current_parameters, current_state, t, N_pred, dt, lambda_u, lambda_e, optimize_keys):

    # Funkcja pomocnicza: wyciąga wartości parametrów, które chcemy optymalizować
    def extract_values(parameters, keys):
        return np.array([parameters[key] for key in keys])
    
    # Funkcja pomocnicza: aktualizuje słownik parametrów na podstawie zoptymalizowanych wartości
    def update_parameters(parameters, keys, values):
        for key, value in zip(keys, values):
            parameters[key] = value
        return parameters
    
    # Tworzymy trajektorię referencyjną dla optymalizacji
    reference_trajectory = calculate_trajectory(current_state, initial_parameters, N_pred, dt, t)
    
    # Funkcja celu: zwraca koszt na podstawie zoptymalizowanych parametrów
    def objective_function(optimized_values):
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
    
    # Wyciągamy początkowe wartości parametrów, które będą optymalizowane
    initial_values = extract_values(current_parameters, optimize_keys)
    
    # Ustawienie ograniczeń dla każdego parametru w optimize_keys
    bounds = Bounds([0.1] * len(optimize_keys), [1000] * len(optimize_keys))
    
    # Wywołanie optymalizacji przy użyciu metody SLSQP
    result = minimize(objective_function, initial_values, method='L-BFGS-B', bounds=bounds, options={'disp': True, 'maxiter': 100, 'gtol': 1e-3, 'ftol': 1e-4})
    
    # Zaktualizowanie parametrów na podstawie wyniku optymalizacji
    optimized_parameters = update_parameters(current_parameters, optimize_keys, result.x)
    
    return optimized_parameters
