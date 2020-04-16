import os
import subprocess
import re

class HermiT:

    def __init__(self):
        # self.path = "C:/Users/zigsi/Desktop/OIE/HermiT/"
        self.path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/external/hermit/"

    def debugPrint(self, i='[X]', *args, **kwargs, ):
        print("[" + str(i) + "] ", end="")
        print(*args, **kwargs)

    def check_unsatisfiable_cases(self, ontology, remove=True, explain=False, i=0):
        '''
        :param ontology:
        :param remove:
        :param explain:
        :return: True or False, if explain==True, return True if ontology OK, else returns list of parsed explanations

        explain OK  |   Result
        --------------------
        FALSE	Yes |   TRUE
        FALSE	No  |   FALSE
        TRUE	Yes	|   TRUE
        TRUE	No	|   List<Exp>

        '''
        os.chdir(self.path)
        onto_path = "ontologies/ontology_tmp_test_" + str(i) + ".owl"
        ontology.serialize(onto_path, format='pretty-xml')
        IRI = "file:///" + self.path.replace("\\", "/") + onto_path
        self.debugPrint(i, "Hermit call")
        if explain:
            output = subprocess.call(['java', '-jar', self.path + "HermiT.jar", '-U', IRI, '-X'],
                                     stdout=open('ontologies/logs/logfile_' + str(i) + '.log', 'w'),
                                     stderr=open('ontologies/logs/logfile_' + str(i) + '.err', 'w'))
        else:
            output = subprocess.call(['java', '-jar', self.path + "HermiT.jar", '-U', IRI],
                                     stdout=open('ontologies/logs/logfile_' + str(i) + '.log', 'w'),
                                     stderr=open('ontologies/logs/logfile_' + str(i) + '.err', 'w'))

        self.debugPrint(i, "Finished")
        if remove:
            os.remove(onto_path)
        else:
            self.debugPrint(i, onto_path)
        if output != 0:
            #print("ERROR COUNT +1")
            #errorCount[0] = errorCount[0] + 1
            #extrNumber = extrNumber - 1
            #O.remove((subject, RDF.type, AURI))
            self.debugPrint(i, "Ourput != 0:")
            self.debugPrint(i, output)
            self.debugPrint(i, "LOG")
            with open('ontologies/logs/logfile_' + str(i) + '.log', 'r') as f:
                read = f.read()
                f.close()
                self.debugPrint(i, read)
                self.debugPrint(i, "ERR")
            with open('ontologies/logs/logfile_' + str(i) + '.err', 'r') as f:
                read = f.read()
                f.close()
                self.debugPrint(i, read)
            if explain:
                # Return a list of explanations; list is empty of no explanations are found
                explanations = self.read_explanations(i)
                return explanations

        else:
            self.debugPrint(i, "Output == 0")
            with open('ontologies/logs/logfile_' + str(i) + '.log', 'r') as f:
                read = f.read()
                f.close()
                self.debugPrint(i, read)
                if read[38:49] != "owl:Nothing" or len(read) > 52:
                    self.debugPrint(i, "unsatisfiable COUNT +1")
                    #errorCount[1] = errorCount[1] + 1
                    #extrNumber = extrNumber - 1
                    #ontology.remove((subject, RDF.type, AURI))
                else:
                    self.debugPrint(i, "Ontology OK")
                    return True
        return False

    def read_explanations(self, i):
        read = None
        with open('explanations.txt', 'r') as f:
            read = f.read()
            f.close()
            self.debugPrint(i, "Explanations: ")
            self.debugPrint(i, read)

        lines = read.split("\n")
        self.debugPrint(i, lines)
        explanations = []
        lines = lines[1:-1]  # we don't need '#1' and new line at the end of file
        exp = []
        for line in lines:
            if line.startswith("#"):
                explanations.append(exp)
                exp = []
            else:
                exp.append(line)
        if len(exp) > 0:
            explanations.append(exp)
        self.debugPrint(i, "Explanations structured:")
        self.debugPrint(i, explanations)
        # Parse explanations
        return self.parse_explanations(explanations, i)

    def parse_explanations(self, explanations, i):
        p_explanations = []
        for explanation in explanations:
            p_explanation = []
            for exp in explanation:
                p_exp = self.parse_exp(exp, i)
                p_explanation.append(p_exp)
            p_explanations.append(p_explanation)
        return p_explanations

    def parse_exp(self, explanation, i):
        num_groups = 4
        text = re.search("(.+?)\( *<(.+?)> *<(.+?)> *<(.+?)> *\)", explanation)
        if text is None:
            num_groups = 3
            text = re.search("(.+?)\( *<(.+?)> *<(.+?)> *\)", explanation)
        self.debugPrint(i, "Parsing explanation")
        self.debugPrint(i, explanation)
        self.debugPrint(i, "Printing text")
        #print(text)
        parsed_explanation = None
        if text:
            #for i in range(1, num_groups+1):
            #    print(text.group(i))
            typ = text.group(1)
            exp_text = ""
            if typ == "ObjectPropertyAssertion":
                exp_text = "Relation not consistent: " + self.url_to_readable_string(text.group(3)) + " " + self.url_to_readable_string(text.group(2)) + " " + self.url_to_readable_string(text.group(4)) + "."
                self.debugPrint(i, exp_text)
            elif typ == "DisjointObjectProperties":
                exp_text = "Relations " + self.url_to_readable_string(text.group(2)) + " and " + self.url_to_readable_string(text.group(3)) + " are opposite/disjoint."
                self.debugPrint(i, exp_text)
            elif typ == "DisjointClasses":
                exp_text = "Concepts " + self.url_to_readable_string(text.group(2)) + " and " + self.url_to_readable_string(text.group(3)) + " are opposite/disjoint."
            elif typ == "ClassAssertion":
                exp_text = "'" + self.url_to_readable_string(text.group(3)) + " is " + self.url_to_readable_string(text.group(2)) + "'."
            else:
                self.debugPrint(i, "Unknown relation type: " + str(typ))
            parsed_explanation = exp_text
        return parsed_explanation

    def url_to_readable_string(self, URL):
        return str(URL).split("#")[1]
