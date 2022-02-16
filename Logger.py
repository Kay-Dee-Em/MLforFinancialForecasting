import logging

class CustomFormatter(logging.Formatter):

    logging.SUCCESS = 25  
    logging.addLevelName(logging.SUCCESS, 'SUCCESS')

    green = "\u001b[32;1m"
    white = "\u001b[37;1m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.INFO: white + format,
        logging.SUCCESS: green + format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)