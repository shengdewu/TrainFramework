import setuptools
import os

package_root = 'engine'


def compose_py_model():
    py_model = list()
    for d in os.listdir(package_root):
        if d == '__init__.py' or d.find('egg-info') != -1:
            continue
        if os.path.isfile(os.path.join(package_root, d)):
            if d.endswith('.py'):
                py_model.append('{}.{}'.format(package_root, d[:d.find('.py')]))
            continue
        for sd in os.listdir(os.path.join(package_root, d)):
            if sd == '__init__.py':
                continue
            if not os.path.isfile(os.path.join(package_root, d, sd)):
                raise FileNotFoundError(os.path.join(package_root, d, sd))
            if not sd.endswith('.py'):
                continue
            py_model.append('{}.{}.{}'.format(package_root, d, sd[:sd.find('.py')]))
    print(py_model)
    return py_model


setuptools.setup(
    name=package_root,
    version='1.0.0',
    author='shengdewu',
    author_email='786222104@qq.com',
    description='train engine for pytorch',
    install_requires=[
        "fvcore == 0.1.5.post20210825",
        "Pillow == 8.1.2",
        "numpy == 1.16.0"
    ],
    classifiers=[
        "programing language :: python ::3",
        "pytorch version :: 1.7+"
        ],
    py_modules=compose_py_model()
)
