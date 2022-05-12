import logging
import os
import sys


class Log(object):

    @staticmethod
    def init_log(log_name, log_path):
        if not os.path.isdir(log_path):
            os.makedirs(log_path)

        logging.basicConfig(
            filename=log_path + '/' + log_name + '.log',
            format='<%(levelname)s %(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s> %(message)s',
            level=logging.INFO)

        log = logging.getLogger()
        stdout_handler = logging.StreamHandler(sys.stdout)
        log.addHandler(stdout_handler)
        return
