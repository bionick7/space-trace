# Load standard modules
import numpy as np

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

import tudatviz


def propagation():
    # Load spice kernels
    spice.load_standard_kernels()

    # Set simulation start and end epochs
    simulation_start_epoch = DateTime(2000, 1, 1).epoch()
    simulation_end_epoch   = DateTime(2000, 1, 2).epoch()

    # Create default body settings for "Earth"
    bodies_to_create = ["Earth", "Moon"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)


    # Add empty settings to body settings
    body_settings.add_empty_settings("Delfi-C3")

    # Create system of bodies (in this case only Earth)
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Define bodies that are propagated
    bodies_to_propagate = ["Delfi-C3"]

    # Define central bodies of propagation
    central_bodies = ["Earth"]

    # Define accelerations acting on Delfi-C3
    acceleration_settings_delfi_c3 = dict(
        Earth=[propagation_setup.acceleration.point_mass_gravity()]
    )

    acceleration_settings = {"Delfi-C3": acceleration_settings_delfi_c3}

    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    # Set initial conditions for the satellite that will be
    # propagated in this simulation. The initial conditions are given in
    # Keplerian elements and later on converted to Cartesian elements
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=earth_gravitational_parameter,
        semi_major_axis=6.99276221e+06,
        eccentricity=4.03294322e-03,
        inclination=1.71065169e+00,
        argument_of_periapsis=1.31226971e+00,
        longitude_of_ascending_node=3.82958313e-01,
        true_anomaly=3.07018490e+00,
    )

    # Create termination settings
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

    # Create numerical integrator settings
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        time_step = 60.0,
        coefficient_set = propagation_setup.integrator.rk_4 )
    
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position("Moon", "Earth"),
    ]

    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings,
        output_variables = dependent_variables_to_save
    )
    # Create simulation object and propagate the dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    # Extract the resulting state history and convert it to an ndarray
    states = dynamics_simulator.propagation_results.state_history
    dependent_variables = dynamics_simulator.propagation_results.dependent_variable_history

    return states, dependent_variables


def visualize(states, dependent_variables):
    states_arr = result2array(states)
    var_arr = result2array(dependent_variables)
    scene = tudatviz.Scene()
    scene.add_trajectory(states_arr[:,0], states_arr[:,1:4])
    scene.add_static_body(0, 0, 0, radius=6.7e6, name='Earth', color=(0, 0.5, 1))
    scene.add_moving_body(states_arr[:,0], var_arr[:,1:4], radius=1.7e6, name='Moon')
    scene.add_trajectory(states_arr[:,0], var_arr[:,1:4])

    # Detailed visualization control
    with tudatviz.show_interactable(scene) as app:
        app.set_focus('Moon')

        while app.is_running():
            app.step()


if __name__ == '__main__':
    states, dependent_variables = propagation()
    visualize(states, dependent_variables)