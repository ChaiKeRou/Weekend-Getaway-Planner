import unittest
from planner import allocate

class TestingPlanner(unittest.TestCase):

    def test_01(self):

        # initialising test
        preferences = [[0], [1], [0,1], [0, 1], [1, 0], [1], [1, 0], [0, 1], [1]]
        licences = [1, 4, 0, 5, 8]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 4, 2, 3 ,6], [1, 5, 8, 7]],
                    [[0, 4, 2, 3, 7], [1, 5, 8, 6]],
                    [[0, 4, 2, 6, 7], [1, 5, 8, 3]],
                    [[0, 4, 3, 6, 7], [1, 5, 8, 2]],
                    [[0, 4, 2, 3], [1, 5, 8, 6, 7]],
                    [[0, 4, 2, 6], [1, 5, 8, 3, 7]],
                    [[0, 4, 2, 7], [1, 5, 8, 3, 6]],
                    [[0, 4, 3, 6], [1, 5, 8, 2, 7]],
                    [[0, 4, 3, 7], [1, 5, 8, 2, 6]],
                    [[0, 4, 6, 7], [1, 5, 8, 2, 3]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_02(self):

        # initialising test
        preferences = [[], [], [], [], [], [], [], []]
        licences = [4, 2, 0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_03(self):

        # initialising test
        preferences = [[1, 0], [0, 1], [1, 0], [], [1], [0], [1]]
        licences = [6, 5, 2]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_04(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0]]
        licences = [0, 4, 3, 2, 1]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3, 4]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_05(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0]]
        licences = [0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_06(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0]]
        licences = []
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_07(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0], [1]]
        licences = [5, 4, 3, 2, 1, 0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_08(self):

        # initialising test
        preferences = [[0]]
        licences = [0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)
    
    def test_09(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0], [0]]
        licences = [1, 4, 3]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_10(self):

        # initialising test
        preferences = [[0], [0, 1], [0], [1, 0], [1], [1]]
        licences = [4, 2, 0, 5]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 1, 2, 3], [4, 5]],
                    [[0, 1, 2], [3, 4, 5]],
                    [[0, 2, 3], [1, 4, 5]],
                    [[0, 2], [1, 3, 4, 5]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])
        
    def test_11(self):

        # initialising test
        preferences = [[0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]
        licences = [10, 1, 7, 3, 5, 0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_12(self):

        # initialising test
        preferences = [[0, 1, 2], [0, 2], [0], [0, 1], [2, 1], [0, 1], [0], [2], [1, 2], [0], [0, 2], [1]]
        licences = [4, 2, 0, 1]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)
    
    def test_13(self):

        # initialising test
        preferences = [[1], [0], [0, 1], [1, 0], [0, 1], [1, 0], [1], [1], [0], [0, 1]]
        licences = [6, 3, 4, 9, 1]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[1, 8, 4, 5, 9], [0, 6, 7, 2, 3]],
                    [[1, 8, 3, 5, 9], [0, 6, 7, 2, 4]],
                    [[1, 8, 3, 4, 5], [0, 6, 7, 2, 9]],
                    [[1, 8, 2, 5, 9], [0, 6, 7, 3, 4]],
                    [[1, 8, 2, 4, 9], [0, 6, 7, 3, 5]],
                    [[1, 8, 2, 4, 5], [0, 6, 7, 3, 9]],
                    [[1, 8, 2, 3, 9], [0, 6, 7, 4, 5]],
                    [[1, 8, 2, 3, 5], [0, 6, 7, 4, 9]],
                    [[1, 8, 2, 3, 4], [0, 6, 7, 5, 9]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_14(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [2], [2]]
        licences =  [0, 1, 5, 6, 10, 11]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_15(self):

        # initialising test
        preferences = [[0], [1], [0,1], [0], [1, 0], [1], [1], [0, 1], [1]]
        licences = [1, 4, 0, 5, 8]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 4, 2, 3], [1, 5, 8, 6, 7]],
                    [[0, 4, 3, 7], [1, 5, 8, 6, 2]],
                    [[0, 4, 2, 3, 7], [1, 5, 8, 6]],
                    [[0, 4, 3], [1, 5, 8, 6, 2, 7]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_16(self):

        # initialising test
        preferences = [[1], [0, 1], [0], [1, 0], [0], [0]]
        licences = [0, 1, 2, 4]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[2, 3, 4, 5], [0, 1]],
                    [[2, 4, 5], [0, 1, 3]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])
    
    def test_17(self):

        # initialising test
        preferences = [[0], [0]]
        licences = [1, 0]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_18(self):

        # initialising test
        preferences = [[0], [0], []]
        licences = [2]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_19(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0]]
        licences = [2, 1, 4]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3, 4]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_20(self):

        # initialising test
        preferences = [[0], [], [0], [0]]
        licences = [0, 2, 1]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)
    
    def test_21(self):

        # initialising test
        preferences = [[0], [1], [1, 0], [1], [0], [1], [0]]
        licences = [3, 4, 1, 2]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 4, 2, 6], [1, 3, 5]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_22(self):

        # initialising test
        preferences = [[]]
        licences = []
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_23(self):

        # initialising test
        preferences = [[0], [0], [0], [0]]
        licences = [0, 2]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_24(self):

        # initialising test
        preferences = [[0], [0], [0]]
        licences = [2, 0, 1]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])
    
    def test_25(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0, 1], [1]]
        licences = [0, 1, 4, 5]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3], [4, 5]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_26(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0, 1], [1]]
        licences = [0, 1, 5]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_27(self):

        # initialising test
        preferences = [[0, 2], [1], [1, 0], [0], [0], [1], [0], [2], [2], [2], [2, 1], [1]]
        licences = [0, 3, 2, 5, 10, 9]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 3, 4, 6], [2, 5, 1, 11], [10, 9, 8, 7]], 
	                [[2, 3, 4, 6], [10, 5, 1, 11], [0, 9, 8, 7]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_28(self):

        # initialising test
        preferences = [[3, 0, 1], [3, 1, 2, 0], [0, 3, 2, 1], [2, 1, 3, 0], [3, 2, 1, 0], [3, 1, 0, 2], [3, 2, 0, 1], [2, 3, 0], [0, 3, 1, 2], [2, 0, 3, 1], [1, 2, 0, 3], [2, 3, 1, 0], [1, 2, 3, 0], [3, 2, 0], [3, 2], [3, 2, 0], [0, 2, 1, 3], [2, 0, 3, 1]]
        licences = [0, 5, 10, 15, 7, 11, 2]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)
    
    def test_29(self):

        # initialising test
        preferences = [[0], [1], [0], [0, 1], [1], [0], [1, 0]]
        licences = [1, 3, 2, 6]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 2, 5, 3], [1, 4, 6]],
	                [[0, 2, 5, 6], [1, 4, 3]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_30(self):

        # initialising test
        preferences = [[0, 1], [1, 0], [0, 1], [0, 1], [0], [1, 0]]
        licences = [5, 3]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_31(self):

        # initialising test
        preferences = [[1, 0], [0, 1], [0], [0], [0], [1], [1], [0, 1], [1], [1, 2], [2], [2]]
        licences = [0, 1, 5, 6, 9, 10]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_32(self):

        # initialising test
        preferences = [[1, 0, 2], [0], [0], [0], [0], [2, 1, 0], [1], [1], [1], [0, 1, 2], [2], [2]]
        licences = [0, 1, 5, 6, 9, 10]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]],
                    [[5, 1, 2, 3, 4], [9, 6, 7, 8], [0, 10, 11]],
                    [[9, 1, 2, 3, 4], [0, 6, 7, 8], [5, 10, 11]],
                    [[0, 1, 2, 3, 4], [9, 6, 7, 8], [5, 10, 11]],
                    [[5, 1, 2, 3, 4], [0, 6, 7, 8], [9, 10, 11]],
                    [[9, 1, 2, 3, 4], [5, 6, 7, 8], [0, 10, 11]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])


# Helper Functions
def find_chosen_output(cars, expected):
    for i in range(len(expected)):
        is_output = True
        for j in range(len(expected[i])):
            if set(cars[j]) != set(expected[i][j]):
                is_output = False
        if is_output:
            return i
    return 0


if __name__ == '__main__':
    unittest.main()
    