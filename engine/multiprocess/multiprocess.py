import multiprocessing
import abc


class MultiProcess(abc.ABC):
    def __init__(self):
        return

    def schedule(self, nthread, data_source_list, args=()):
        """
        :param nthread: 使用的进程数量
        :param data_source_list: 数据源列表
        :param args: 其他参数
        :return:
        """
        data_source_total = len(data_source_list)

        if data_source_total < nthread:
            raise RuntimeError('data source total num {} less than nthread {}'.format(data_source_total, nthread))

        nper = data_source_total // nthread
        nratain = data_source_total % nthread

        data_source = dict()
        for number in range(nthread):
            data_source[number] = data_source_list[number * nper: (number+1) * nper]

        if nratain > 0:
            nratain_source_list = data_source_list[data_source_total-nratain:]
            for i, source in enumerate(nratain_source_list):
                data_source[i].append(source)

        active_thread = list()
        for number in range(nthread):
            param = [data_source[number], number, args]
            #1
            # active_thread.append(multiprocessing.Process(
            #     target=self.__class__.excute,
            #     name='task_schedule'+str(n),
            #     args=(self, param)))
            #2
            active_thread.append(multiprocessing.Process(
                target=self.execute,
                name='task_schedule'+str(number),
                args=(param, )))

        for thread in active_thread:
            thread.start()

        for thread in active_thread:
            thread.join()
        return

    @abc.abstractmethod
    def execute(self, args=()):
        return
