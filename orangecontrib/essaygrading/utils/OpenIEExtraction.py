import os
import subprocess
from orangecontrib.essaygrading.utils.lemmatizer import lemmatizeSentences
from orangecontrib.essaygrading.utils.lemmatizer import stemSentences


class OpenIEExtraction:
    openie = None

    def __init__(self, openie_system="ClausIE"):
        pass

    def extract_triples(self, sentences, id=1):
        pass


# https://github.com/AnthonyMRios/pyclausie

# POMEMBNO!!!!

# Tisti, ki je napisal ta wrapper je en velk kekec
# https://github.com/AnthonyMRios/pyclausie/issues/2
# a problem je biu ker datoteke pred brisanjem ni zapru................ kekc
# FIX: v vrstici 68 dodaš: input_file.close()

# VECJI KEKEC OD PRICAKOVANJ: v Triples.py je treba dvakrat dodati:
# if len(line.split('\t')) < 4:
#    continue
# enkrat < 4, drugic < 5


# TRETJI FIX:
# povsod odstrani decode() v Triples.py
# problem je biu ker on uporabla PIPE. zazene calusie in majstr caka da se konca, ampak PIPE ma omejen buffer
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


class ClausIE(OpenIEExtraction):

    def __init__(self):
        from orangecontrib.essaygrading.external.pyclausie.pyclausie import ClausIE
        self.openie = ClausIE.get_instance()

    def extract_triples(self, sentences, id=1):
        triples = []

        # OPTIMIZATION: join all sentences in one array and then group them by essay
        # sentences = sentences[0:15]
        sent_num = []
        all_sentences = []
        for essay_sentences in sentences:
            sent_num.append(len(essay_sentences))
            for sentence in essay_sentences:
                all_sentences.append(sentence)

        # all_sentences = lemmatizeSentences(all_sentences, stem_ing=True)
        # mogoce zanimivo: https://stackoverflow.com/questions/47856247/extract-verb-phrases-using-spacy
        # Try stemming, because ClausIE does not have robust error handling
        # all_sentences = stemSentences(all_sentences)
        print(all_sentences)
        extracted_triples = self.openie.extract_triples(all_sentences)

        print(len(extracted_triples))

        global_i = 0
        triples = []
        sent_count = 1
        for i in range(len(sent_num)):
            essay_triples = []
            for j in range(sent_num[i]):
                sentence_triples = []
                while len(extracted_triples) > global_i and extracted_triples[global_i].index == str(sent_count):
                    sentence_triples.append(extracted_triples[global_i])
                    global_i += 1
                essay_triples.append(sentence_triples)
                sent_count += 1
            triples.append(essay_triples)

        return triples


class OpenIE5(OpenIEExtraction):

    def __init__(self):
        self.path = "C:\\Users\\zigsi\\Desktop\\OIE\\OpenIE-standalone-master"
        os.chdir(self.path)

    def extract_triples(self, sentences, id=1):

        # write them to file and run openIE5; also remember which sentences match with which essay
        sent_num = []

        input_name = str(id) + ".txt"
        output_name = str(id) + "_out.txt"

        with open(input_name, "w", encoding="utf8") as input_file:
            for essay_sentences in sentences:
                sent_num.append(len(essay_sentences))
                # essay_sentences = lemmatizeSentences(essay_sentences)
                for sentence in essay_sentences:
                    if len(sentence) < 5:
                        print(sentence)
                    input_file.write(sentence + "\n")
            input_file.close()

        # java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar
        p = subprocess.Popen(
            ['java', '-Xmx10g', '-XX:+UseConcMarkSweepGC', '-jar', 'openie-assembly-5.0-SNAPSHOT.jar', input_name,
             output_name, '--ignore-errors'])
        p.wait()

        out_file = open(str(id) + "out_parsed.txt", "w", encoding="utf8")

        triples = []
        essay_index = 1
        with open(output_name, "r", encoding="utf8") as output_file:
            lines = output_file.readlines()
            fi = 0
            for i in sent_num:
                essay_triples = []
                for j in range(i):
                    sentence_triples = []
                    # fi+0 je stavek
                    # fi+1 je ekstrakcija
                    # fi+x je prazna vrstica, ki pomeni konec
                    while len(lines) > fi + 1 and len(lines[fi + 1]) > 5:
                        split = lines[fi + 1].split(" ", 1)
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
                            triple = {}
                            if len(t_split) == 3:
                                # triple = {'subject': t_split[0], 'predicate': t_split[1], 'object': t_split[2]}
                                triple = Triple(t_split[0], t_split[1], t_split[2])
                            else:
                                if t_split[2].startswith("T:") or t_split[2].startswith("L:"):
                                    # triple = {'subject': t_split[0], 'predicate': t_split[1], 'object': t_split[3], 'time': t_split[2]}
                                    triple = Triple(t_split[0], t_split[1], t_split[3], t_split[2])
                                elif t_split[3].startswith("T:") or t_split[3].startswith("L:"):
                                    # triple = {'subject': t_split[0], 'predicate': t_split[1], 'object': t_split[2], 'time': t_split[3]}
                                    triple = Triple(t_split[0], t_split[1], t_split[2], t_split[3])
                                else:
                                    print("ERROR")
                                    print(t_split)
                                    # triple = {'subject': t_split[0], 'predicate': t_split[1], 'object': t_split[2]}
                                    triple = Triple(t_split[0], t_split[1], t_split[2])
                                    sentence_triples.append(triple)
                                    # triple = {'subject': t_split[0], 'predicate': t_split[1], 'object': t_split[3]}
                                    triple = Triple(t_split[0], t_split[1], t_split[3])
                            # sentence_triples.append((confidence, t_split))
                            # out_file.write(str(essay_index) + " " + str(j) + " (" + triple["subject"] + "; " + triple["predicate"] + "; " + triple["object"] + ")\n")
                            out_file.write(str(essay_index) + " " + str(j) + " (" + triple.subject + "; " +
                                           triple.predicate + "; " + triple.object + ")\n")
                            sentence_triples.append(triple)
                    fi += 2
                    essay_triples.append(sentence_triples)
                triples.append(essay_triples)
                essay_index += 1

                # We have a list: [Essay1:[Triples sentence 1: [...], Triples sentence 2:[...], Essay 2: [...]...]]
                # Flatten the list
                final_triples = []  # over sentences
                final_all_triples = []  # over essays
                index = 0
                # print(triples)
                for triples_essay in triples:
                    for triples_sentence in triples_essay:
                        for triple in triples_sentence:
                            triple.index = index
                            final_triples.append([triple])
                        index += 1
                    final_all_triples.append(final_triples)

        return final_all_triples


class Triple:
    index = 0
    subject = ""
    predicate = ""
    object = ""
    time = ""

    def __init__(self, sub, pred, obj, time="", index=0):
        self.index = index
        self.subject = sub
        self.predicate = pred
        self.object = obj
        self.time = time


if __name__ == "__main__":
    q = ClausIE()
    q.extract_triples([[
        'Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it.',
        'How did you learn about other countrys/states outside of yours?',
        "Well I have by computer/internet, it's a new way to learn about what going on in our time!",
        "You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows.",
        'Believe it or not the computer is much interesting then in class all day reading out of books.',
        "If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right.",
        'You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by.',
        'Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place.',
        'Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble.',
        'Thank you for listening.']])
    # q.extract_triples([["Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place."]])
