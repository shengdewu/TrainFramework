import setuptools

setuptools.setup(
    name='tframework',
    version='1.0.0',
    author='shengdewu',
    author_email='786222104@qq.com',
    description='train framework for pytorch',
    classifiers=[
        "programing language :: python ::3",
        "pytorch version :: 1.7+"
        ],
    py_modules=[
        "tframework.log.Log",
        "tframework.model.BaseModel",
        "tframework.schedule.BaseSchedule"
        ]
)
