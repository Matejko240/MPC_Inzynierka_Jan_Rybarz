#from modelODE import model_ode
import numpy as np
from scipy.optimize import minimize
from effectorTrajectoryGenerator3D import effectorTrajectoryGenerator3D
from scipy.integrate import ode

def cost_function(parameters, predicted_trajectory, reference_trajectory, control_effort, lambda_u):
    """
    Funkcja kosztu penalizująca błąd trajektorii oraz sterowanie.
    :param parameters: Parametry modelu (długości i masy)
    :param predicted_trajectory: Przewidywana trajektoria manipulatora
    :param reference_trajectory: Trajektoria referencyjna
    :param control_effort: Nakłady na sterowanie
    :param lambda_u: Waga penalizująca sterowanie
    :return: Wartość funkcji kosztu
    """
    error = predicted_trajectory[:, :3] - reference_trajectory
    cost = np.sum(np.linalg.norm(error, axis=1)) + lambda_u * np.sum(np.linalg.norm(control_effort, axis=1))
    return cost


def predict_future_states(current_state, parameters, N_pred, dt,t):
    from modelODE import model_ode  # Opóźniony import
 
    predicted_trajectory = np.zeros((N_pred, 3))  # Predicted x, y, z trajectory
    
    # Start with the current state
    solver = ode(model_ode)
    solver.set_integrator('dopri5', rtol=1e-3, atol=1e-5)

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
    for i in range(min(N_pred, len(youtput))):
        # Get the predicted state from the modelODE function
        _, additional_info, _ = model_ode(t[i], youtput[i, :], parameters)
        
        # Extract the end-effector position from the additional info if necessary
        # Or use the predicted state to build the trajectory
        end_effector_position = additional_info['end_effector_position']
        #print("end=",end_effector_position)
        # Store the predicted position for this step
        predicted_trajectory[i, :] = end_effector_position
    return predicted_trajectory

def calculate_reference_trajectory(t,dt, N_pred, parameters):
    extended_ref = np.zeros((N_pred, 3))  # Initialize array to hold reference trajectory (N_pred x 3)

    # Loop over each time step to calculate the reference trajectory
    for i in range(N_pred):
        current_time = t + i * dt  # Time for each prediction step
        # Use effectorTrajectoryGenerator3D to get the full trajectory
        full_trajectory = effectorTrajectoryGenerator3D(current_time, parameters)[0]
        # Extract the x, y, z positions (first three elements)
        extended_ref[i, :] = full_trajectory
    return extended_ref
def optimize_parameters(initial_parameters, current_state,t, N_pred, dt, lambda_u, optimize_keys):
    
    # Wyciągamy tylko te wartości, które chcemy optymalizować
    def extract_values(parameters, keys):
        return np.array([parameters[key] for key in keys])
    
    # Aktualizujemy tylko zoptymalizowane wartości
    def update_parameters(parameters, keys, values):
        for key, value in zip(keys, values):
            parameters[key] = value
        return parameters
    
    # Funkcja celu - optymalizacja tylko wybranych parametrów
    def objective_function(optimized_values):
        # Tymczasowy słownik parametrów z zoptymalizowanymi wartościami
        temp_parameters = update_parameters(initial_parameters.copy(), optimize_keys, optimized_values)
        control_input = np.zeros((N_pred, 3))  # Zastępcze sterowanie dla predykcji
        reference_trajectory = calculate_reference_trajectory(t, dt, N_pred, temp_parameters)
        predicted_trajectory = predict_future_states(current_state, temp_parameters, N_pred, dt,t)
        cost = cost_function(temp_parameters, predicted_trajectory, reference_trajectory, control_input, lambda_u)
        return cost
    
    # Wyciągamy początkowe wartości tylko dla wybranych kluczy
    initial_values = extract_values(initial_parameters, optimize_keys)
    
    # Optymalizacja przy użyciu BFGS
    result = minimize(objective_function, initial_values, method='BFGS', options={'disp': True, 'maxiter': 100, 'gtol': 1e-4})
    
    # Zaktualizowanie parametrów na podstawie wyniku optymalizacji
    optimized_parameters = update_parameters(initial_parameters, optimize_keys, result.x)
    
    return optimized_parameters
