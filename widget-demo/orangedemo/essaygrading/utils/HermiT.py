import os
import subprocess

class HermiT:

    def __init__(self):
        self.path = "C:/Users/zigsi/Desktop/OIE/HermiT/"

    def check_unsatisfiable_cases(self, ontology):
        os.chdir(self.path)
        onto_path = "ontologies/ontology_tmp.owl"
        ontology.serialize(onto_path, format='pretty-xml')
        IRI = "file:///" + self.path + onto_path
        print("Hermit call")
        output = subprocess.call(['java', '-jar', self.path + "HermiT.jar", '-U', IRI],
                                 stdout=open('ontologies/logs/logfile.log', 'w'),
                                 stderr=open('ontologies/logs/logfile.err', 'w'))
        print("Finished")
        os.remove(onto_path)
        if output != 0:
            #print("ERROR COUNT +1")
            #errorCount[0] = errorCount[0] + 1
            #extrNumber = extrNumber - 1
            #O.remove((subject, RDF.type, AURI))
            print("qweqwe")
        else:
            '''with open('ontologies/logs/logfile.log', 'r') as f:
                read = f.read()
                f.close()
                if read[38:49] != "owl:Nothing":
                    print("unsatisfiable COUNT +1")
                    errorCount[1] = errorCount[1] + 1
                    extrNumber = extrNumber - 1
                    O.remove((subject, RDF.type, AURI))
                else:
                    print("Ontology OK")'''

