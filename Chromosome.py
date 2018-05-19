import copy
from random import shuffle
from random import random
from random import choice
from Utils import Utils

class Chromosome:
    def __init__(self, id, m1, m2, jobs_list_1, jobs_list_2, makespan_1=None, makespan_2=None):
        self._id = id
        self._m1 = m1
        self._m2 = m2
        self._jobs_list_1 = jobs_list_1
        self._jobs_list_2 = jobs_list_2
        self._makespan_1 = makespan_1
        self._makespan_2 = makespan_2
        self._first_stage = None
        self._second_stage = None
        self._strategy = ''

    def update_stats(self):
        self.compute_makespan_1()
        self.compute_release_date()
        self.compute_makespan_2()

    def compute_release_date(self):
        for job2 in self.jobs_list_2:
            f_stage = self.jobs_list_1
            job2.release_date = 0
            if f_stage != []:
                job2.release_date = max([j.release_date for j in f_stage if j.id in job2.predecessors])

    # swap an element with other
    def mutation(self, mutation_rate=0.2):
        jobs_1 = copy.deepcopy(self._jobs_list_1)

        for i in range(len(jobs_1)):
            if (random() < 0.2):
                other_index = choice(range(len(jobs_1)))
                if i != other_index:
                   jobs_1[other_index], jobs_1[i] = jobs_1[i], jobs_1[other_index]

        return jobs_1

    def compute_makespan_1(self):
        def get_machine_index(machines):
            min_row = min_col = 0
            min_start_time = 0

            j = 0
            while j < len(machines[0]) and machines[0][j] is not None:
                min_start_time += machines[0][j].processing_time
                j += 1
            min_col = j

            for i in range(1, len(machines)):
                j = 0
                total_pt = 0
                while j < len(machines[i]) and machines[i][j] is not None:
                    total_pt += machines[i][j].processing_time
                    j += 1

                if j < len(machines[i]):
                    if total_pt < min_start_time:
                        min_start_time = total_pt
                        min_row = i
                        min_col = j
                else:
                    print("Erro: Máquina cheia!")
            return min_row, min_col, min_start_time

        self._first_stage = [[None] * len(self._jobs_list_1) for _ in range(self._m1)]
        self._makespan_1 = 0

        for job in self._jobs_list_1:
            m_row, m_col, start_time = get_machine_index(self._first_stage)
            job.release_date = start_time + job.processing_time
            self._makespan_1 = max(self._makespan_1, job.release_date)
            self._first_stage[m_row][m_col] = job

    def compute_makespan_2(self):
        def get_machine_index_2(machines):
            min_row = min_col = 0
            min_start_time = 0

            j = 0
            while j < len(machines[0]) and machines[0][j] is not None:
                min_start_time = machines[0][j].completion_time
                j += 1
            min_col = j

            for i in range(1, len(machines)):
                j = 0
                total_pt = 0
                while j < len(machines[i]) and machines[i][j] is not None:
                    total_pt = machines[i][j].completion_time
                    j += 1

                if j < len(machines[i]):
                    if total_pt < min_start_time:
                        # print(total_pt, min_makespan, "Entrou")
                        min_start_time = total_pt
                        min_row = i
                        min_col = j
                else:
                    print("Erro: Máquina cheia!")
            return min_row, min_col, min_start_time

        self._second_stage = [[None] * len(self._jobs_list_2) for _ in range(self._m2)]
        self._makespan_2 = 0

        for job in self._jobs_list_2:

            m_row, m_col, start_time = get_machine_index_2(self._second_stage)
            if m_col > 0:
                job.completion_time = job.processing_time + max(start_time, job.release_date)
            else:
                job.completion_time = job.processing_time + job.release_date

            self._makespan_2 = max(self._makespan_2, job.completion_time)
            self._second_stage[m_row][m_col] = job

    def print_first_stage(self):
        print("Makespan:", self._makespan_1)
        print([j.id for j in self._jobs_list_1])
        for i in range(len(self._first_stage)):
            for j in range(len(self._first_stage[i])):
                if self._first_stage[i][j] is not None:
                    print((self._first_stage[i][j].release_date), end="\t")
                else:
                    print("-", end="\t")
            print()

    def print_second_stage(self):
        print("Makespan:", self._makespan_2)
        print([j.id for j in self._jobs_list_2])
        for i in range(len(self._second_stage)):
            for j in range(len(self._second_stage[i])):
                if self._second_stage[i][j] != None:
                    print(self._second_stage[i][j].completion_time, end="\t")
                else:
                    print("-", end="\t")
            print()

    @property
    def id(self):
        return self._id

    @property
    def strategy(self):
        return self._strategy

    @property
    def makespan_1(self):
        return self._makespan_1

    @makespan_1.setter
    def makespan_1(self, value):
        self._makespan_1 = value

    @property
    def makespan_2(self):
        return self._makespan_2

    @makespan_2.setter
    def makespan_2(self, value):
        self._makespan_2 = value

    @property
    def jobs_list_1(self):
        return self._jobs_list_1

    @jobs_list_1.setter
    def jobs_list_1(self, value):
        self._jobs_list_1 = value

    @property
    def jobs_list_2(self):
        return self._jobs_list_2

    @jobs_list_2.setter
    def jobs_list_2(self, value):
        self._jobs_list_2 = value

    def apply_lpt_rule_1(self):
        self._strategy = 'LPT'
        self._jobs_list_1.sort(reverse=True)

    def apply_spt_rule_1(self):
        self._strategy = 'SPT'
        self._jobs_list_1.sort(reverse=False)

    def apply_lns_rule_1(self):
        self._strategy = 'LNS'
        self._jobs_list_1.sort(key=lambda job: job.total_successors())

    def apply_sns_rule_1(self):
        self._strategy = 'SNS'
        self._jobs_list_1.sort(key=lambda job: job.total_successors(), reverse=True)

    def apply_random_rule_1(self):
        self._strategy = 'RND'
        shuffle(self._jobs_list_1)

    def apply_lpt_rule_2(self):
        self._jobs_list_2.sort(key=lambda job: job.processing_time + job.release_date, reverse=True)

    def apply_spt_rule_2(self):
        self._jobs_list_2.sort(key=lambda job: job.processing_time + job.release_date, reverse=False)

    def apply_lns_rule_2(self):
        self._jobs_list_2.sort(key=lambda job: job.total_predecessors())

    def apply_sns_rule_2(self):
        self._jobs_list_2.sort(key=lambda job: job.total_predecessors(), reverse=True)

    def apply_available_1_rule_2(self):
        self._jobs_list_2.sort(key=lambda job: job.release_date, reverse=False)

    def apply_available_2_rule_2(self):
        self._jobs_list_2.sort(key=lambda job: job.release_date, reverse=True)

    def apply_random_rule_2(self):
        shuffle(self._jobs_list_2)

    def __str__(self):
        return ("ID: "+str(self._id)+" | JList: ["+
                ", ".join([str(i._id) for i in self._jobs_list_1])+ "], [" +
                ", ".join([str(i._id) for i in self._jobs_list_2]) +
                "] | MSPAN(1): "+str(self._makespan_1) + "| MSPAN(2): "+str(self._makespan_2))