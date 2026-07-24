import datetime
import logging
import os


class myLogger(logging.Logger):
    def __init__(
        self,
        name: str = 'myLogger',
        log_dir: str = './logs',
        log_filename: str = 'default',
        debug: bool = False,
        verbose: bool = False,
        propagate: bool = True,
        arg_dict: None | dict = None,
    ):
        super().__init__(name)

        self.init_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.basedir = log_dir
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        os.makedirs(log_dir, exist_ok=True)

        if log_filename == 'default':
            log_filename = f"{os.path.splitext(os.path.basename(__file__))[0]}.txt"
        logger_file = logging.FileHandler(os.path.join(log_dir, log_filename))
        logger_file.setFormatter(formatter)
        self.addHandler(logger_file)

        if propagate:
            self.propagate = True
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.addHandler(stream_handler)
        else:
            self.propagate = False

        log_level = logging.DEBUG if debug else logging.INFO
        self.setLevel(log_level)
        logger_file.setLevel(log_level)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        if debug:
            verbose = True

        if verbose and arg_dict:
            self.info(arg_dict)

    @staticmethod
    def _join_values(values, separator=' '):
        return separator.join(str(value) for value in values)

    def __call__(self, *values):
        self.info(self._join_values(values))

    def getTime(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def getInitTime(self):
        return self.init_time

    def csvLog(self, *values):
        self.info(f"{self._join_values(values, ',')},")

    def print(self, *values):
        msg = self._join_values(values)
        if self.propagate:
            self.info(msg)
        else:
            print(msg)

    def info_(self, *values):
        self.info(self._join_values(values))

    def error_(self, *values):
        msg = self._join_values(values)
        print(msg)
        return super().error(msg)

    def debug_(self, *values):
        msg = self._join_values(values)
        if self.level == logging.DEBUG and self.propagate:
            self.print(msg)
        return super().debug(msg)

    def registerDir(self, directory: str):
        self.basedir = directory

    def announceDir(self):
        self.print(f"Directory using: {self.basedir}")


def setLogger(
    name: str = 'myLogger',
    log_dir: str = './logs',
    log_filename: str = 'default',
    debug: bool = False,
    verbose: bool = False,
    args=None,
) -> logging.Logger:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    os.makedirs(log_dir, exist_ok=True)
    if log_filename == 'default':
        log_filename = f"{os.path.splitext(os.path.basename(__file__))[0]}.log"
    logger_file = logging.FileHandler(os.path.join(log_dir, log_filename))
    logger_file.setFormatter(formatter)
    logger.addHandler(logger_file)
    if debug:
        logger.setLevel(logging.DEBUG)
        verbose = True
    if verbose:
        logger.info(args)

    return logger


if __name__ == '__main__':
    logger = myLogger(
        name='myLogger',
        log_dir='./logs',
        log_filename='myLogger.log',
        debug=True,
        verbose=True,
        propagate=True,
    )
    logger("hello")
