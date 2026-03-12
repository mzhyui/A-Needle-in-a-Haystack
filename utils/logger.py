import datetime
import logging
import os

class myLogger(logging.Logger):
    handlers = []
    def __init__(self, name: str = 'myLogger', log_dir: str = './logs', log_filename: str = 'default', debug: bool = False, verbose: bool = False, propagate: bool = True, arg_dict:None|dict=None):
        super().__init__(name)

        self.init_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.basedir = log_dir
        
        # Set the base logging level based on the debug flag
        # logging.basicConfig(level=logging.INFO) if not debug else logging.basicConfig(level=logging.DEBUG)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Ensure log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Set the log filename based on the provided arguments
        log_filename = os.path.splitext(os.path.basename(__file__))[0] + '.txt' if log_filename == 'default' else log_filename
        logger_file = logging.FileHandler(os.path.join(log_dir, log_filename))
        logger_file.setFormatter(formatter)
        
        # Add the file handler
        self.addHandler(logger_file)
        
        # Add a stream handler if propagation is enabled
        if propagate:
            self.propagate = True
            streamer = logging.StreamHandler()
            streamer.setFormatter(formatter)
            self.addHandler(streamer)
        else:
            self.propagate = False

        # Adjust the logger level for debug mode
        if debug:
            self.setLevel(logging.DEBUG)
            logger_file.setLevel(logging.DEBUG)
            mpl_logger = logging.getLogger('matplotlib')
            mpl_logger.setLevel(logging.WARNING)
            verbose = True
        else:
            self.setLevel(logging.INFO)
            logger_file.setLevel(logging.INFO)
            mpl_logger = logging.getLogger('matplotlib')
            mpl_logger.setLevel(logging.WARNING)

        
        # Log the initial arguments if verbose is set
        if verbose and arg_dict:
            self.info(arg_dict)

            

    def __call__(self, *values):
        info_str = ""
        for value in values:
            info_str += str(value) + " "
        self.info(info_str)

    def getTime(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def getInitTime(self):
        return self.init_time
    
    def csvLog(self, *values):
        info_str = ""
        for value in values:
            info_str += str(value) + ","
        self.info(info_str)

    def print(self, *values):
        msg = " ".join(str(value) for value in values)
        if self.propagate:
            self.info(msg)
        else:
            print(msg)

    def info_(self, *values):
        msg = " ".join(str(value) for value in values)
        self.info(msg)

    def error_(self, *values):
        msg = " ".join(str(value) for value in values)
        print(msg)
        return super().error(msg)
    
    def debug_(self, *values):
        msg = " ".join(str(value) for value in values)
        if self.level == logging.DEBUG and self.propagate:
            self.print(msg)
        return super().debug(msg)
    
    def registerDir(self, directory: str):
        self.basedir = directory
    
    def announceDir(self):
        self.print(f"Directory using: {self.basedir}")
        

def setLogger(name: str = 'myLogger', log_dir: str = './logs', log_filename: str = 'default', debug: bool = False, verbose: bool = False, args = None) -> logging.Logger:
    if not debug:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger(name)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.splitext(os.path.basename(__file__))[0] + '.log' if log_filename == 'default' else log_filename
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
    logger = myLogger(name='myLogger', log_dir='./logs', log_filename='myLogger.log', debug=True, verbose=True, propagate=True)
    logger("hello")