import setuptools
import os
import sys

package_root = 'engine'


def _search_dir(dp, name):
    py_name = list()

    if os.path.isfile(os.path.join(dp, name)):
        if name.endswith('.py') and name != '__init__.py':
            py_name.append('{}.{}'.format('.'.join(dp.split('/')), name[:name.find('.py')]))
        return py_name

    sdp = os.path.join(dp, name)
    for sd in os.listdir(sdp):
        if sd == '__init__.py' or sd.find('egg-info') != -1:
            continue
        if not os.path.isfile(os.path.join(dp, name, sd)):
            py_name.extend(_search_dir(sdp, sd))
        else:
            if sd.endswith('.py'):
                py_name.append('{}.{}'.format('.'.join(sdp.split('/')), sd[:sd.find('.py')]))

    return py_name


def compose_py_model():
    py_model = list()
    for d in os.listdir(package_root):
        if d == '__init__.py' or d.find('egg-info') != -1:
            continue
        py_model.extend(_search_dir(package_root, d))
    print(py_model)
    return py_model


def get_version() -> str:
    return '3.0.2'


setuptools.setup(
    name='{}_frame'.format(package_root),
    version=get_version(),
    author='shengdewu',
    author_email='786222104@qq.com',
    description='train engine for pytorch',
    classifiers=[
        "programing language :: python ::3",
        "pytorch version :: 1.7+"
        ],
    py_modules=compose_py_model()
)
