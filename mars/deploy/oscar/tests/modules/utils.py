import os
import shutil
import pytest


@pytest.fixture
def cleanup_third_party_modules_output():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    shutil.rmtree(output_dir, ignore_errors=True)
    yield
    shutil.rmtree(output_dir, ignore_errors=True)


def get_output_filenames():
    return os.listdir(os.path.join(os.path.dirname(__file__), 'output'))
