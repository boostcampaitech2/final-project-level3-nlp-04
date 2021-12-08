import logging
import os
import errno
import datetime
import config as c


class LogHelper:
    __instance = None

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            cls.__instance = LogHelper()
        return cls.__instance

    def __init__(self):
        self.log = None

        self.os_path = os.path.dirname(os.path.realpath(__file__)).replace('/core', '').replace('\\core', '')
        self.log_folder_nm = 'Log'
        self.log_folder_path = '{0}/{1}'.format(self.os_path, self.log_folder_nm)
        self.is_log = c.LOG_CONFIG['is_log']
        self.is_debug = c.LOG_CONFIG['is_debug']
        self.is_info = c.LOG_CONFIG['is_info']
        self.is_error = c.LOG_CONFIG['is_error']

        self.today = None

        self._dir_check()

    @staticmethod
    def _mkdir(path):
        if not os.path.isdir(path):
            try:
                os.makedirs(path, exist_ok=True)  # Python>3.2
            except TypeError:
                try:
                    os.makedirs(path)
                except OSError as exc:  # Python >2.5
                    if exc.errno == errno.EEXIST and os.path.isdir(path):
                        pass
                    else:
                        raise

    def _dir_check(self):
        if self.is_log:
            now_date = datetime.datetime.now().strftime('%Y%m%d')

            self._mkdir(self.log_folder_path)
            self._mkdir('{0}/{1}'.format(self.log_folder_path, now_date))

            if self.log is None or self.today != now_date:
                file_handler = logging.FileHandler('{0}/{1}/{2}/log.txt'.format(self.os_path,
                                                                                self.log_folder_nm, now_date), 'a')

                self.log = logging.getLogger('kotra')

                for handler in self.log.handlers[:]:
                    if isinstance(handler, logging.FileHandler):
                        self.log.removeHandler(handler)

                self.log.setLevel(logging.DEBUG)

                # fm = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                # fm = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
                fm = logging.Formatter('%(levelname)s: %(message)s')
                file_handler.setFormatter(fm)

                self.log.addHandler(file_handler)
                self.today = now_date

    def d(self, msg, file_name=None, func_name=None):
        print('DEBUG: {msg}'.format(msg=msg))
        if self.is_log and self.is_debug:
            self._dir_check()
            log_msg = self._log_additional_info(self._log_message(msg), file_name, func_name)
            self.log.debug(log_msg)

    def i(self, msg, file_name=None, func_name=None):
        print('INFO: {msg}'.format(msg=msg))
        if self.is_log and self.is_info:
            self._dir_check()
            log_msg = self._log_additional_info(self._log_message(msg), file_name, func_name)
            self.log.info(log_msg)

    def e(self, msg, file_name=None, func_name=None):
        print('ERROR: {msg}'.format(msg=msg))
        if self.is_log and self.is_error:
            self._dir_check()
            log_msg = self._log_additional_info(self._log_message(msg), file_name, func_name)
            self.log.error(log_msg)

    @staticmethod
    def _log_message(msg):
        log_msg = '{0} [{1}]'.format(msg, datetime.datetime.now().strftime('%H:%M:%S'))
        return log_msg

    @staticmethod
    def _log_additional_info(log_msg, file_name=None, func_name=None):
        if file_name is not None:
            log_msg += ' > {0}'.format(file_name)
        if func_name is not None:
            log_msg += ' > {0}()'.format(func_name)

        return log_msg


