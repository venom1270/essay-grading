import os
import subprocess
# https://github.com/AnthonyMRios/pyclausie

class OpenIEExtraction:

    openie = None

    def __init__(self, openie_system="ClausIE"):
        pass

    def extract_triples(self, sentences):
        pass

# https://github.com/AnthonyMRios/pyclausie

# POMEMBNO!!!!

# Tisti, ki je napisal ta wrapper je en velk kekec
# https://github.com/AnthonyMRios/pyclausie/issues/2
# a problem je biu ker datoteke pred brisanjem ni zapru................ kekc
# FIX: v vrstici 68 dodaš: input_file.close()

# VECJI KEKEC OD PRICAKOVANJ: v Triples.py je treba dvakrat dodati:
#if len(line.split('\t')) < 4:
#    continue
# enkrat < 4, drugic < 5


# TRETJI FIX:
# povsod odstrani decode() v Triples.py
# problem je biu ker lolek uporabla PIPE. zazene calusie in majstr caka da se konca, ampak PIPE ma omejen buffer
# kar pomen da je verjetno clausi caku da se buffer sprazne da lahko naprej poslje.... sm naredu da je pol kr prek fajlov use... bol zihr
# SubprocessBackend.py:
'''
    def extract_triples(self, sentences, ids=None, java='java',
                        print_sent_confidence=False):
        """ This method takes a list of sentences and ids (optional)
            then returns a list of triples extracted by ClausIE.

            currently supported options:
                -l - Used if ids is a list and not None.
                -p - Returns the confidence score of the extracted
                     triple.
            
            Note: sentences and ids must be a list even if you only
                  extract triples for one sentence.
        """
        input_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            if ids is not None:
                for identifier, sentence in zip(ids, sentences):
                    input_file.write('{0!r}\t{1!r}\n'.format(identifier, sentence).encode('utf8'))
            else:
                for sentence in sentences:
                    input_file.write('{0!r}\n'.format(sentence).encode('utf8'))
            input_file.flush()

            command = [self.java_command,
                       '-jar', self.jar_filename,
                       '-f', input_file.name,
                       '-o', input_file.name + "_out"]
            if ids is not None:
                command.append('-l')
            if print_sent_confidence:
                command.append('-p')
            sd_process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
            return_code = sd_process.wait()
            stderr = sd_process.stderr.read()
            stdout = sd_process.stdout.read()
            self._raise_on_bad_exitcode(return_code, stderr)
        finally:
            input_file.close()
            os.remove(input_file.name)
        
        out = open(input_file.name + "_out")
		
        triples = Corpus.from_tsv(out.read().splitlines(), print_sent_confidence)
        out.close()
        os.remove(input_file.name + "_out")
        return triples
'''

# ZDEJ KONCN MISLM DA DELA :)
class ClausIE(OpenIEExtraction):

    def __init__(self):
        from pyclausie import ClausIE
        self.openie = ClausIE.get_instance()

    def extract_triples(self, sentences):
        triples = []
        for essay_sentences in sentences:
            print(essay_sentences)
            print("EXTRACTING")
            triples = self.openie.extract_triples(essay_sentences)
            triples.append(triples)
        return triples


class OpenIE5(OpenIEExtraction):

    def __init__(self):
        self.path = "C:\\Users\\zigsi\\Desktop\\OIE\\OpenIE-standalone-master"
        os.chdir(self.path)

    def extract_triples(self, sentences):

        # write them to file and run openIE5; also remember which sentences match with which essay
        sent_num = []
        with open("input.txt", "w") as input_file:
            for essay_sentences in sentences:
                sent_num.append(len(essay_sentences))
                for sentence in essay_sentences:
                    input_file.write(sentence + "\n")
            input_file.close()

        # java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar
        p = subprocess.Popen(['java', '-Xmx10g', '-XX:+UseConcMarkSweepGC', '-jar', 'openie-assembly-5.0-SNAPSHOT.jar', 'input.txt', 'output.txt'])
        p.wait()

        triples = []
        with open("output.txt", "r") as output_file:
            lines = output_file.readlines()
            fi = 0
            for i in sent_num:
                essay_triples = []
                for j in range(i):
                    sentence_triples = []
                    #fi+0 je stavek
                    #fi+1 je ekstrakcija
                    #fi+x je prazna vrstica, ki pomeni konec
                    while len(lines[fi+1]) > 5:
                        split = lines[fi+1].split(" ", 1)
                        confidence = split[0]
                        t = split[1]
                        fi += 1
                        # TODO: zakaj se vcasih pojavi COntext?? zaenkrat skip
                        if t.find("Context(") == -1:
                            t_split = t.split("; ")
                            # odstrani oklepaj in zaklepaj
                            t_split[0] = t_split[0][1:]
                            t_split[-1] = t_split[-1][0:-2]
                            # dodaj v triple objekt
                            sentence_triples.append((confidence, t_split))
                    fi += 2
                    essay_triples.append(sentence_triples)
                triples.append(essay_triples)

        return triples








if __name__ == "__main__":
    q = ClausIE()
    q.extract_triples([['Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it.', 'How did you learn about other countrys/states outside of yours?', "Well I have by computer/internet, it's a new way to learn about what going on in our time!", "You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows.", 'Believe it or not the computer is much interesting then in class all day reading out of books.', "If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right.", 'You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by.', 'Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place.', 'Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble.', 'Thank you for listening.']])
    #q.extract_triples([["Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place."]])
