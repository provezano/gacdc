class Job:
    def __init__(self, job_id, processing_time, successors=None, predecessors=None, release_date=None, completion_time=None):
        self._id = job_id # Job id
        self._processing_time = processing_time # Time to process this job
        self._release_date = release_date # Time this job is available for processing
        self._completion_time = completion_time # Time this job will be finished

        # If this is a first stage job, it can have successors. If this is a second stage job, it can have predecessors.
        self._successors = successors
        self._predecessors = predecessors

    @property
    def id(self):
        return self._id

    @property
    def processing_time(self):
        return self._processing_time

    @processing_time.setter
    def processing_time(self, value):
        self._processing_time = value

    @property
    def successors(self):
        return self._successors

    @successors.setter
    def successors(self, sucessors):
        self._successors = sucessors

    @property
    def predecessors(self):
        return self._predecessors

    @predecessors.setter
    def predecessors(self, predecessors):
        self._predecessors = predecessors

    @property
    def release_date(self):
        return self._release_date

    @release_date.setter
    def release_date(self, value):
        self._release_date = value

    @property
    def completion_time(self):
        return self._completion_time

    @completion_time.setter
    def completion_time(self, value):
        self._completion_time = value

    def total_successors(self):
        return len(self.successors)

    def total_predecessors(self):
        return len(self.predecessors)

    def __eq__(self, other):
        if self is not None and other is not None:
            if self.id == other.id:
                return True
        return False

    def __lt__(self, other):
        return self.processing_time < other.processing_time

    def __str__(self):
        return "ID: "+ str(self._id) + " | P: " + str(self._processing_time) + \
                (' | Succ: '+ ', '.join(str(e) for e in self._successors) if self._successors != None
                                         else (' | Pred: '+ ', '.join(str(e) for e in self._predecessors)))