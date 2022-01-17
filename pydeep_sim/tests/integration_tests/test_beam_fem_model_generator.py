import os
import pytest
import sys

from pydeep_sim.main import main


def test_beam_fem_model_generator(inputdir,tmpdir):
    print(inputdir)
    arguments = [
        '--input=' + os.path.join(inputdir, 'input_beam_fem_model_generator.json'),
        '--output=' + str(tmpdir),
    ]
#    breakpoint()
    main(arguments)
    pass

#test_beam_fem_model_generator('pydeep_sim/tests/integration_tests/input_files/', '/tmp')
@pytest.fixture
def inputdir():
    """ Return the path to the json input-files of the function test. """
    dirpath = os.path.dirname(__file__)
    input_files_path = os.path.join(dirpath, 'input_files')
    return input_files_path