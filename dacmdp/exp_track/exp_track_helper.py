import argparse
import pickledb
import time 
import torch 
import time
import pickledb
import os 
import numpy as np
import socket


class Experiment():
    def __init__(self, id, meta="None", expPrefix="", expSuffix=""):
        self.id = id
        self.meta = ' '.join(meta.split())
        self.expPrefix = ' '.join(expPrefix.split())
        self.expSuffix = ' '.join(expSuffix.split())

    @property
    def command(self):
        return " ".join([self.expPrefix,  self.expSuffix, "--exp_id", self.id, "--exp_meta", "'" + self.meta +"'"])
    
    @property
    def options(self):
        return " ".join([self.expSuffix, "--exp_id", self.id, "--exp_meta", "'" + self.meta +"'"])

class ExperimentPool():
    def __init__(self, listOfExperiments = []):
        self.expPool = {}
        for exp in listOfExperiments:
            self.add_experiment(exp)

    def add_experiment(self, exp):
        assert exp.id not in self.expPool, f"{exp.id} alread taken"
        self.expPool[exp.id] = exp

    def get_by_meta(self, meta):
        metaPool = {exp.meta:exp for exp in self.expPool.values()}
        return metaPool[meta]

    def get_by_id(self, id):
        return self.expPool[id]

    @staticmethod
    def joinPools(*pools):
        experiments = []
        for pool in pools:
            for exp in pool.expPool.values():
                experiments.append(exp)
        return ExperimentPool(experiments)

class Tracker:
    def __init__(self, db_name, base_path = "logs/tracker_logs/"):
        self.dbname = base_path+db_name
        self.db = pickledb.load(self.dbname, True) 
        
    def check_id(self, job_id, hard_test = True):
        self.db = pickledb.load(self.dbname, True) 
        if job_id not in self.db.getall():
            print("Experiment Id Missing from Database:", job_id)
            if hard_test:
                assert False
            return False
        else:
            return True


    def change_status_to(self, job_id, status, aux_dict = {}):
        self.check_id(job_id)
        self.db = pickledb.load(self.dbname, True) 
        self.db.set(job_id, { **self.db.get(job_id), 
                              **{"status":status},
                              **aux_dict, 
                            })
        self.db.dump()

        #SanityCheck
        self.db = pickledb.load(self.dbname, True) 
        if self.db.get(job_id)["status"] != status:
            time.sleep(np.random.randint(5))
            self.change_status_to(job_id, status, aux_dict)
        print("Status of job {} successfully switched to {}".format(job_id, status))
            

    def dependencies_met(self, dependencies):
#         self.check_id(job_id)
        self.db = pickledb.load(self.dbname, True) 
        dependencies_met = True
        for dd in dependencies :
            self.check_id(dd)
            if self.db.get(dd)["status"]!="finished":
                dependencies_met = False
        return dependencies_met
    
    def is_queued(self, job_id):
        self.check_id(job_id)
        self.db = pickledb.load(self.dbname, True) 
        return self.db.get(job_id)["status"] == "queued"
    
    def is_running(self, job_id):
        self.check_id(job_id)
        self.db = pickledb.load(self.dbname, True) 
        return self.db.get(job_id)["status"] == "running"
    
    def seed_all_jobs(self, job_ids):
        self.db = pickledb.load(self.dbname, True) 
        for job_id in job_ids:
            self.db.set(job_id, {"status":"queued"})
            print("Seeding Job:", job_id)
        
        self.db.dump()
        
        #SanityCheck
        self.db = pickledb.load(self.dbname, True) 
        for job_id in job_ids:
            assert self.db.get(job_id)["status"] =="queued"

    def reset_error_jobs(self, job_ids = None):
        job_ids = job_ids or self.db.getall()
        self.db = pickledb.load(self.dbname, True)
        reset_job_id_list = []
        for job_id in job_ids:
            if self.db.get(job_id)["status"] =="Error":
                self.db.set(job_id, {"status": "queued"})
                reset_job_id_list.append(job_id)

        self.db.dump()

        # SanityCheck
        self.db = pickledb.load(self.dbname, True)
        for job_id in reset_job_id_list:
            assert self.db.get(job_id)["status"] == "queued"


    def reset_jobs_for_step(self, step_id , job_ids = None, reset_to ="queued" ):
        job_ids = job_ids or self.db.getall()
        self.db = pickledb.load(self.dbname, True)
        reset_job_id_list = []
        for job_id in job_ids:
            if step_id in job_id:
                self.db.set(job_id, {"status": reset_to})
                reset_job_id_list.append(job_id)

        self.db.dump()

        # SanityCheck
        self.db = pickledb.load(self.dbname, True)
        for job_id in reset_job_id_list:
            assert self.db.get(job_id)["status"] == reset_to


    def choose_not_allocated_gpu(self, device_list):
        self.db = pickledb.load(self.dbname, True) 
        host_name = socket.gethostname().split(".")[0]
        allocated_devices =[self.db.get(k)["GPU_ID"] for k in self.db.getall() if self.db.get(k)["status"] in ["running", "allocated"]  and "GPU_ID" in self.db.get(k)]
        for d in device_list:
            if host_name + "GPU:"+ str(d) not in allocated_devices:
                return d 
        return None