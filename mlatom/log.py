import logging


class ColouredFormatter(logging.Formatter):
    RESET_SEQ = "\033[0m"
    Colour_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"
    
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)
    GREY = 90
    
    Colours = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
        "TIME": GREY,
    }
    
    def __init__(self, msg, use_colour=True):
        self.use_colour = use_colour
        datefmt = "%Y-%m-%d %H:%M:%S"
        if use_colour:
            datefmt = (
                self.Colour_SEQ % (self.Colours["TIME"]) + datefmt + self.RESET_SEQ
            )
        
        super().__init__(self.formatter_message(msg), datefmt=datefmt)
    
    def format(self, record):
        if self.use_colour:
            levelname = record.levelname
            if levelname in self.Colours:
                levelname_colour = (
                    self.Colour_SEQ % (self.Colours[levelname])
                    + "%-10s" % f"[{levelname}]"
                    + self.RESET_SEQ
                )
                record.levelname = levelname_colour
                record.msg = (
                    self.Colour_SEQ % (self.Colours[levelname])
                    + record.msg
                    + self.RESET_SEQ
                )
        elif record.levelname in self.Colours:
            record.levelname = ("%-10s" % f"[{record.levelname}]")
            
        return logging.Formatter.format(self, record)
    
    def formatter_message(self, message):
        if self.use_colour:
            message = message.replace("$RESET", self.RESET_SEQ).replace(
                "$BOLD", self.BOLD_SEQ
            )
        else:
            message = message.replace("$RESET", "").replace("$BOLD", "")
        return message


def getColouredLogger(
    name: str = None,
    fmt: str = "$BOLD%(levelname)s$RESET %(asctime)s %(name)s:\t %(message)s (%(filename)s:%(lineno)d)",
    stderr: bool = False,
    fileout: str = None,
    level: int = None,
    use_colour: bool = True,
):
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(level)
    logger.propagate = False
    files_out = [
        handler.baseFilename
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    has_stderr = '<stderr>' in [handler.stream.name for handler in logger.handlers]
    if fileout and fileout not in files_out:
        formatter = ColouredFormatter(fmt, use_colour=use_colour)
        handler = logging.FileHandler(fileout, "a")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if stderr and not has_stderr:
        formatter = ColouredFormatter(fmt, use_colour=use_colour)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
