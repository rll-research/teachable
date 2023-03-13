import unittest
from utils import agent_saver


class PickleUtilsTestCases(unittest.TestCase):
    people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},
                  2: {'name': 'Marie', 'age': '22', 'sex': 'Female'},
                  3: {'name': 'Luna', 'age': '24', 'sex': 'Female', 'married': 'No'},
                  4: {'name': 'Peter', 'age': '29', 'sex': 'Male', 'married': 'Yes'}}

    def test_serialization_works(self):

        pickle_utils.save_object(self.people, "people_test")
        from_disk = pickle_utils.load_object("people_test")

        self.assertDictEqual(self.people, from_disk)

    def test_serialization_works_with_prefix(self):

        pickle_utils.save_object(self.people, "people_test", prefix='sub')
        from_disk = pickle_utils.load_object("people_test", prefix='sub')

        self.assertDictEqual(self.people, from_disk)





if __name__ == '__main__':
    unittest.main()
