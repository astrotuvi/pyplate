import os
import sys
import datetime as dt


class ProcessLog:
    """
    Plate process log class

    """

    def __init__(self, file_path):
        if file_path:
            self.path = file_path
            self.enable = True
        else:
            self.path = None
            self.enable = False

        self.handle = None

        self.platedb = None
        self.archive_id = None
        self.plate_id = None
        self.scan_id = None
        self.process_id = None

    def open(self):
        """
        Open log file.

        """

        if self.enable:
            log_dir = os.path.dirname(self.path)

            try:
                os.makedirs(log_dir)
            except OSError:
                if not os.path.isdir(log_dir):
                    print('Could not create directory {}'.format(log_dir))

            try:
                self.handle = open(self.path, 'w', 1)
            except IOError:
                print('Could not open log file {}'.format(self.path))
                self.handle = sys.stdout
        else:
            self.handle = sys.stdout

    def write(self, message, timestamp=True, double_newline=True,
              level=None, event=None, solution_num=None):
        """
        Write a message to the log file and optionally to the database.

        Parameters
        ----------
        message : str
            Message to be written to the log file
        timestamp : bool
            Write timestamp with the message (default True)
        double_newline : bool
            Add two newline characters after the message (default True)
        solution_num : int
            Astrometric solution number

        """

        log_message = '{}'.format(message)

        if solution_num:
            log_message = '{{Solution {:d}}} {}'.format(solution_num,
                                                        log_message)

        if timestamp:
            log_message = '[{}] {}'.format(str(dt.datetime.now()), log_message)

        if double_newline:
            log_message += '\n'

        self.handle.write('{}\n'.format(log_message))

        if level is not None:
            self.to_db(level, message, event=event, solution_num=solution_num)

    def to_db(self, level, message, event=None, solution_num=None):
        """
        Write a log message to the database.

        Parameters
        ----------
        level : int
            Log level (1 = error, 2 = warning, 3 = info, 4 = debug, 5 = trace)
        message : str
            Message to be written to the log file
        event : int
            Event code (default None)
        solution_num : int
            Astrometric solution number

        """

        if self.platedb is not None and self.process_id is not None:
            self.platedb.write_processlog(level, message, event=event,
                                          solution_num=solution_num,
                                          process_id=self.process_id,
                                          scan_id=self.scan_id,
                                          plate_id=self.plate_id,
                                          archive_id=self.archive_id)

    def close(self):
        """
        Close log file.

        """

        if self.enable and self.handle is not sys.stdout:
            self.handle.close()
