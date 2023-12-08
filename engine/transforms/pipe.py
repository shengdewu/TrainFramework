from .build import build_transformer
from typing import Dict
import logging


class TransformCompose:
    def __init__(self, transformers: list):
        self.transformers = list()

        for transformer in transformers:
            if callable(transformer):
                self.transformers.append(transformer)
            elif isinstance(transformer, dict):
                kwargs = dict()
                arch_name = ''
                for k, v in transformer.items():
                    if k.lower() == 'name':
                        arch_name = v
                        continue
                    kwargs[k.lower()] = v
                self.transformers.append(build_transformer(arch_name, **kwargs))
            else:
                raise TypeError('transform must be callable or a dict')
        return

    def _check_field(self, kwargs: Dict):
        fields = list()

        img_fields = kwargs.get('img_fields', None)
        if img_fields is not None:
            fields.extend(img_fields)

        color_fields = kwargs.get('color_fields', None)
        if color_fields is not None:
            fields.extend(color_fields)

        shapes = [kwargs[field].shape[0:2] for field in set(fields)]
        base_shape = shapes[0]
        shapes = [shape[0] - base_shape[0] + shape[1] - base_shape[1] for shape in shapes]
        assert sum(shapes) == 0, 'the shape of the field of img_fields or color_fields not equal'

        if kwargs.get('img_shape', (0, 0)) != base_shape:
            kwargs['img_shape'] = base_shape
            logging.warning('the img_shape not equal the shape of the field of img_fields or color_fields')
        if kwargs.get('ori_shape', (0, 0)) != base_shape:
            kwargs['ori_shape'] = base_shape
            logging.warning('the ori_shape not equal the shape of the field of img_fields or color_fields')

        return

    def __call__(self, kwargs: Dict):
        if 'img_fields' not in kwargs.keys() and 'color_fields' not in kwargs.keys():
            return kwargs

        self._check_field(kwargs)

        for transformer in self.transformers:
            kwargs = transformer(kwargs)
        return kwargs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transformers:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
