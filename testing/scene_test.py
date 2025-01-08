import unittest
import numpy as np
from spacetrace.scene import Scene, Trajectory, Body, TransformShape, VectorShape

'''
    Far from rigorous, as mosts testing is done visually.
    This justs checks if all the data is received correctly in setup
'''

class TestScene(unittest.TestCase):

    def setUp(self):
        self.scene = Scene(scale_factor=2.1)

    def test_trajectory_patches(self):
        epochs = np.linspace(0, 1, 2**14+1)
        states = np.random.uniform(-1, 1, (2**14+1, 6))
        self.scene.add(Trajectory(epochs, states, name="Test Trajectory", color='blue'))

        self.assertRaises(ValueError, Trajectory, epochs, states[:-1])
        self.assertEqual(len(self.scene.trajectories), 1)
        self.assertEqual(len(self.scene.trajectory_patches), 2)
        self.assertEqual(self.scene.trajectory_patches[0][1], (2**14 - 1) * 6)
        self.assertEqual(self.scene.trajectory_patches[0][2], 0)
        self.assertEqual(self.scene.trajectory_patches[1][1], 6)
        self.assertEqual(self.scene.trajectory_patches[1][2], 0)

    def test_vector_setting(self):
        epochs = np.linspace(0, 1, 1000)
        vector_objects = [
            VectorShape(epochs, np.array([0, 0, 0]), np.array([0, 1, 0])),
            VectorShape(epochs, np.zeros((len(epochs, 3))), np.array([0, 1, 0])),
            VectorShape(epochs, np.array([0, 0, 0]), np.ones((len(epochs, 3)))),
            VectorShape(epochs, np.zeros((len(epochs, 3))), np.ones((len(epochs, 3))))
        ]
        for i in range(3):
            self.assertAlmostEqual(sum(vector_objects.positions[i]), sum(vector_objects.positions[i+1]))
            self.assertAlmostEqual(sum(vector_objects.vectors[i]), sum(vector_objects.vectors[i+1]))

    def test_vector_setting(self):
        epochs = np.linspace(0, 1, 1000)
        matrix_objects = [
            TransformShape(epochs, np.array([0, 0, 0]), np.eye(3)),
            TransformShape(epochs, np.zeros((len(epochs, 3))), np.eye(3)),
            TransformShape(epochs, np.array([0, 0, 0]), np.repeat(np.eye(3)[np.newaxis,:,:], 1000, axis=0)),
            TransformShape(epochs, np.zeros((len(epochs, 3))), np.repeat(np.eye(3)[np.newaxis,:,:], 1000, axis=0))
        ]
        for i in range(3):
            self.assertAlmostEqual(sum(matrix_objects.positions[i]), sum(matrix_objects.positions[i+1]))
            self.assertAlmostEqual(sum(matrix_objects.vectors[i]), sum(matrix_objects.vectors[i+1]))
        
        self.assertIsInstance(self.scene.transforms[0], TransformShape)
        self.assertEqual(self.scene.transforms[0].name, "Default Frame")

    def test_vector_setting(self):
        epochs = np.linspace(0, 1, 2**14+1)
        vector_objects = [
            VectorShape(epochs, np.array([0, 0, 0]),        np.array([1, 1, 1])),
            VectorShape(epochs, np.zeros((len(epochs), 3)), np.array([1, 1, 1])),
            VectorShape(epochs, np.array([0, 0, 0]),        np.ones((len(epochs), 3))),
            VectorShape(epochs, np.zeros((len(epochs), 3)), np.ones((len(epochs), 3))),
        ]

        for i in range(3):
            self.assertAlmostEqual(np.sum(vector_objects[i].positions), np.sum(vector_objects[i+1].positions))
            self.assertAlmostEqual(np.sum(vector_objects[i].vectors), np.sum(vector_objects[i+1].vectors))

    def test_position_fetching(self):
        epochs = np.linspace(0, 1, 2**14+1)
        states = np.random.uniform(-1, 1, (2**14+1, 6))
        states[:,1] = np.linspace(0, 1, 2**14+1)
        self.scene.add(Trajectory(epochs, states, name="Body"))

        trajectory = self.scene.trajectories[0]
        self.assertEqual(trajectory.name, "Body")
        for rand_val in np.random.uniform(0, 1, 10):
            x, y, z = trajectory.get_position(rand_val)
            self.assertAlmostEqual(-z, rand_val)

    def test_time_bounds(self):
        epochs = np.linspace(0, 1, 1000)
        states = np.zeros((1000, 3))
        self.scene.add(Trajectory(epochs * 10, states))
        self.scene.add(Body(epochs * 10 + 20, states, 1))

        min_t, max_t = np.inf, -np.inf
        for entity in self.scene.entities:
            min_t = min(min_t, np.min(entity.epochs))
            max_t = max(max_t, np.max(entity.epochs))
        self.assertEqual(self.scene.time_bounds[0], min_t)
        self.assertEqual(self.scene.time_bounds[1], max_t)
