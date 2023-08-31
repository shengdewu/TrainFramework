from .build import build_transformer


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

    def __call__(self, kwargs):
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
