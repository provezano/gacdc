from Job import Job


class Utils:
    @staticmethod
    def read_from_file(path):
        """
            Read an instance from file.
            
            Keyword arguments:
                path: full path name of a instance file.
            
            Returns a 6-tuple:
                j1: number of inbound jobs
                j2: number of outbound jobs
                m1: number of machines in first stage
                m2: number of machines in second stage
                jobs_1: list of first stage jobs
                jobs_2: list of second stage jobs
        """
        with open(path) as f: lines = [line.rstrip('\n') for line in f]

        j1 = int(lines[0]) # number of inbound jobs
        j2 = int(lines[1]) # number of outbound jobs
        m1 = int(lines[2]) # number of first stage machines
        m2 = int(lines[3]) # number of second stage machines

        p1 = [] # processing time j1
        for p in lines[5:j1 + 5]:
            p1.append(int(p))

        p2 = [] # processing time j2
        for p in lines[j1+6:j2+j1+6]:
            p2.append(int(p))

        # Precedent/Sucessor matrix
        precedents_matrix = [l.split() for l in lines[j2+j1+7:]]

        predecessors = [[] for _ in range(j2)]
        for i in range(j2):
            for j in range(j1):
                if precedents_matrix[i][j] == '1':
                    predecessors[i].append(j)

        successors = [[] for _ in range(j1)]
        for j in range(j1):
            for i in range(j2):
                if precedents_matrix[i][j] == '1':
                    successors[j].append(i)

        jobs_1 = []
        for i in range(len(p1)):
            jobs_1.append(Job(i, p1[i], successors=successors[i]))

        jobs_2 = []
        for i in range(len(p2)):
            jobs_2.append(Job(i, p2[i], predecessors=predecessors[i]))

        #print("M1: {} M2: {} P1: {} P2: {} Predecessors: {} Successors: {}".format(j1, j2, m1, m2, predecessors, successors))
        return j1, j2, m1, m2, jobs_1, jobs_2


    @staticmethod
    def get_machine_index(machines):
        min_row = min_col = 0
        min_start_time = 0

        j = 0
        while j < len(machines[0]) and machines[0][j] != None:
            min_start_time += machines[0][j].processing_time
            j += 1
        min_col = j

        for i in range(1, len(machines)):
            j = 0
            total_pt = 0
            while j < len(machines[i]) and machines[i][j] != None:
                total_pt += machines[i][j].processing_time
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


    @staticmethod
    def test_makespan_1(jobs_1, m1):
        first_stage = [[None] * len(jobs_1) for _ in range(m1)]
        makespan_1 = 0

        for job in jobs_1:
            m_row, m_col, start_time = Utils.get_machine_index(first_stage)
            job.release_date = start_time + job.processing_time
            makespan_1 = max(makespan_1, job.release_date)
            first_stage[m_row][m_col] = job

        return makespan_1