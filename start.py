import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from modelODE import model_ode
from mpc import optimize_parameters
import random
def main():
    """_summary_

    Returns:
        _type_: _description_
    """
    tEnd = 10
    np.random.seed(123456789)

    Pi = np.pi

    m1 = 50.0 # kg, masa pierwszego segmentu
    m2 = 50.0 # kg, masa drugiego segmentu
    m3 = 50.0 # kg, masa trzeciego segmentu

    mc = m1 + m2 + m3

    l1 = 1.0 # m, długość pierwszego ramienia
    l2 = 1.0 # m, długość drugiego ramienia
    l3 = 1.0  # m, długość trzeciego ramienia

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

    sample_time = 0.5

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
    optimize_keys = ['l1']  # Lista parametrów do optymalizacji
    # Przypisanie losowych wartości dla parametrów z optimize_keys
    for key in optimize_keys:
        if key in parameters:
            # Przypisanie losowej wartości w określonym zakresie
            parameters[key] = random.uniform(0.1, 10.0)  # Zakres możesz dostosować do swoich potrzeb
    print("Current parameters (parametry obecne):", parameters)      
    while solver.successful() and solver.t < tEnd:
        solver.integrate(solver.t + sample_time)
        t.append(solver.t)
        youtput.append(solver.y)
        # Wywołanie optymalizacji parametrów
        optimized_parameters = optimize_parameters(
            initial_parameters=init_parameters, #parametry rzeczywiste których szukamy
            current_parameters=parameters, #parametry obecne
            current_state=solver.y, # obecny stan
            t=solver.t, # obecny czas
            N_pred=5, # ilość kroków predykcji
            dt=sample_time, # próbka czasu
            lambda_u=0.00, # współczynnik control effort
            lambda_e = 1.0,
            
            optimize_keys=optimize_keys  # Optymalizujemy tylko wybrane parametry
        )
        print("Current parameters (parametry obecne):", parameters)
        print("czas=",solver.t)
        # Zaktualizowanie parametrów po optymalizacji
        parameters.update(optimized_parameters)
        solver.set_f_params(parameters)

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

    # Drawing the final chart

    plt.figure()

    # Trajektoria referencyjna
    plt.plot(sim_data['time'], additional['qchd'][0, :], label='qchd_x (ref)', color='red', linestyle='--')
    plt.plot(sim_data['time'], additional['qchd'][1, :], label='qchd_y (ref)', color='green', linestyle='--')
    plt.plot(sim_data['time'], additional['qchd'][2, :], label='qchd_z (ref)', color='blue', linestyle='--')

    # Aktualna trajektoria efektora
    plt.plot(sim_data['time'], additional['k'][0, :], label='x (actual)', color='red')
    plt.plot(sim_data['time'], additional['k'][1, :], label='y (actual)', color='green')
    plt.plot(sim_data['time'], additional['k'][2, :], label='z (actual)', color='blue')
    



    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja efektora')
    plt.title('Trajektoria efektora (rzeczywista vs referencyjna)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    # Wykres dla błędów pozycji i prędkości
    plt.figure()

    # Norma błędu pozycji
    plt.plot(sim_data['time'], error_norm, label='Norma błędu pozycji (e)', color='black', linestyle=':')
    
    # Norma błędu prędkości
    plt.plot(sim_data['time'], error_d1_norm, label='Norma błędu prędkości (e_d1)', color='purple', linestyle='-.')

    plt.xlabel('Czas [s]')
    plt.ylabel('Norma błędu')
    plt.title('Norma błędu pozycji i prędkości')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
