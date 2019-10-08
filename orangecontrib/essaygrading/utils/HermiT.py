import os
import subprocess

class HermiT:

    def __init__(self):
        self.path = "C:/Users/zigsi/Desktop/OIE/HermiT/"

    def check_unsatisfiable_cases(self, ontology, remove=True):
        os.chdir(self.path)
        onto_path = "ontologies/ontology_tmp.owl"
        ontology.serialize(onto_path, format='pretty-xml')
        IRI = "file:///" + self.path + onto_path
        print("Hermit call")
        output = subprocess.call(['java', '-jar', self.path + "HermiT.jar", '-U', IRI],
                                 stdout=open('ontologies/logs/logfile.log', 'w'),
                                 stderr=open('ontologies/logs/logfile.err', 'w'))
        print("Finished")
        if remove:
            os.remove(onto_path)
        else:
            print(onto_path)
        if output != 0:
            #print("ERROR COUNT +1")
            #errorCount[0] = errorCount[0] + 1
            #extrNumber = extrNumber - 1
            #O.remove((subject, RDF.type, AURI))
            print("Ourput != 0:")
            print(output)
            print("LOG")
            with open('ontologies/logs/logfile.log', 'r') as f:
                read = f.read()
                f.close()
                print(read)
            print("ERR")
            with open('ontologies/logs/logfile.err', 'r') as f:
                read = f.read()
                f.close()
                print(read)
        else:
            print("Output == 0")
            with open('ontologies/logs/logfile.log', 'r') as f:
                read = f.read()
                f.close()
                print(read)
                if read[38:49] != "owl:Nothing":
                    print("unsatisfiable COUNT +1")
                    #errorCount[1] = errorCount[1] + 1
                    #extrNumber = extrNumber - 1
                    #ontology.remove((subject, RDF.type, AURI))
                else:
                    print("Ontology OK")
                    return True
        return False
