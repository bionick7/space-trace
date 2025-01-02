import unittest
import numpy as np
from spacetrace.scene import Scene, Trajectory, Body, ReferenceFrame

class TestScene(unittest.TestCase):

    def setUp(self):
        self.scene = Scene(scale_factor=2.1)

    def test_trajectory_patches(self):
        epochs = np.linspace(0, 1, 2**14+1)
        states = np.random.uniform(-1, 1, (2**14+1, 6))
        self.scene.add_trajectory(epochs, states, name="Test Trajectory", color='blue')

        self.assertRaises(ValueError, self.scene.add_trajectory, epochs, states[:-1])
        self.assertEqual(len(self.scene.trajectories), 1)
        self.assertEqual(len(self.scene.trajectory_patches), 2)
        self.assertEqual(self.scene.trajectory_patches[0][1], (2**14 - 1) * 6)
        self.assertEqual(self.scene.trajectory_patches[0][2], 0)
        self.assertEqual(self.scene.trajectory_patches[1][1], 6)
        self.assertEqual(self.scene.trajectory_patches[1][2], 0)


    def test_position_fetching(self):
        epochs = np.linspace(0, 1, 2**14+1)
        states = np.random.uniform(-1, 1, (2**14+1, 6))
        states[:,1] = np.linspace(0, 1, 2**14+1)
        self.scene.add_trajectory(epochs, states, name="Body")

        trajectory = self.scene.trajectories[0]
        self.assertEqual(trajectory.name, "Body")
        for rand_val in np.random.uniform(0, 1, 10):
            x, y, z = trajectory.get_position(rand_val)
            self.assertAlmostEqual(-z, rand_val * self.scene.scale_factor)

    def test_reference_frame(self):
        self.assertIsInstance(self.scene.reference_frame, ReferenceFrame)
        self.assertEqual(self.scene.reference_frame.name, "Default Frame")

    def test_time_bounds(self):
        epochs = np.linspace(0, 1, 1000)
        states = np.zeros((1000, 3))
        self.scene.add_trajectory(epochs * 10, states)
        self.scene.add_moving_body(epochs * 10 + 20, states)

        min_t, max_t = np.inf, -np.inf
        for entity in self.scene.entities:
            min_t = min(min_t, np.min(entity.epochs))
            max_t = max(max_t, np.max(entity.epochs))
        self.assertEqual(self.scene.time_bounds[0], min_t)
        self.assertEqual(self.scene.time_bounds[1], max_t)