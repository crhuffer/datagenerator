import datetime


class Timer:

    def __init__(self, taskname: str = 'Task'):
        import datetime

        self.taskname = taskname
        self.datetime_start = datetime.datetime.now()

    def stop(self):
        self.datetime_end = datetime.datetime.now()
        self.timedelta = self.datetime_end - self.datetime_start
        self.duration_seconds = self.timedelta.total_seconds()

    def stopAndPrint(self):
        self.stop()

        print('{taskname} took {duration} seconds to run'.format(taskname=self.taskname,
                                                                 duration=self.duration_seconds))
