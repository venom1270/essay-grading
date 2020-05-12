from nltk.stem import PorterStemmer
import copy

class URIRefEntity:

    '''
    URIRef struct class to hold structured information about added entities into ontology.
    '''

    URIRef = None  # URI
    text = None  # "original" text (may be preprocessed
    stemmed = None  # stemmed text
    original = None  # OPTIONAL; original text
    type = None  # "SubObj" or "Pred" || EKŠULI: "Predicate" or "SubjectObject"

    def __init__(self, URIRef=None, text=None, stemmed=None, type=None, original=None):
        self.URIRef = URIRef
        self.text = text
        self.stemmed = stemmed
        self.type = type
        self.original = original

        if self.text is not None and self.stemmed is None:
            self.stemmed = PorterStemmer().stem(self.text)


class EntityStore:

    '''
    Class for holding all added URIRefEntities in ontology. Supports fast lookup, adding and removing in addition to
    creating a snapshot for data restoration.
    '''

    entities = []  # array of URIRefEntity
    entities_snapshot = []  # "old" entities

    lookup_pred = dict()  # two dictionaries for quick lookup
    lookup_subobj = dict()
    lookup_p_snapshot = dict()  # lookup snapshots
    lookup_so_snapshot = dict()

    def __init__(self):
        pass

    def add_entity(self, entity: URIRefEntity):
        '''
        Add entity to store.
        :param entity: URIRefEntity to add to store.
        '''
        self.entities.append(entity)
        if entity.type == "Predicate":
            self.lookup_pred[entity.text] = entity
            self.lookup_pred[entity.stemmed] = entity
        elif entity.type == "SubjectObject":
            self.lookup_subobj[entity.text] = entity
            self.lookup_subobj[entity.stemmed] = entity

    def add(self, URIRef, text, stemmed, type, original=None):
        '''
        Convert parameters to URIRefEntity and add to store.
        :param URIRef: rdflig URIRef() of element.
        :param text: element text.
        :param stemmed: stemmed element text.
        :param type: element type ("SubjectObject" or "Predicate").
        :param original: original element text (optional).
        '''
        self.add_entity(URIRefEntity(URIRef=URIRef, text=text, stemmed=stemmed, type=type, original=original))

    # Search by only one parameter + type!!!
    def find(self, URIRef=None, text=None, stemmed=None, type=None) -> URIRefEntity:
        '''
        Search store for URIRefEntity by one of parameters + type.
        :return: URIRefEntity if found or None.
        '''
        if type is None:
            return None
        if URIRef is None and text is None and stemmed is None:
            return None

        # Try finding via lookup
        if type == "Predicate":
            if text is not None and self.lookup_pred.get(text):
                return self.lookup_pred.get(text)
            if stemmed is not None and self.lookup_pred.get(stemmed):
                return self.lookup_pred.get(stemmed)
        elif type == "SubjectObject":
            if text is not None and self.lookup_subobj.get(text):
                return self.lookup_subobj.get(text)
            if stemmed is not None and self.lookup_subobj.get(stemmed):
                return self.lookup_subobj.get(stemmed)

        # Try finding "by bruteforcing"
        search_list = [e for e in self.entities if e.type == type]
        if text is not None:
            results = [e for e in search_list if e.text == text]
        if stemmed is not None:
            results = [e for e in search_list if e.text == text]
        if URIRef is not None:
            results = [e for e in search_list if e.text == text]

        if len(results) > 0:
            return results[0]
        else:
            return None

    def get_list(self, parameter=None, type=None):
        '''
        Get list of all entites with specified parameter.
        e.g. "text" to get a list of "text" of all URIRefEntites in store.
        :param parameter: "text", "stemmed" or "URIRef"
        :param type: "SubjectObject" or "Predicate".
        :return: list matching parameters.
        '''
        search_list = self.entities
        if type is not None:
            search_list = [e for e in search_list if e.type == type]
        if parameter == "text":
            return [e.text for e in search_list]
        if parameter == "stemmed":
            return [e.stemmed for e in search_list]
        if parameter == "URIRef":
            return [e.URIRef for e in search_list]
        return search_list

    def get_by_index(self, index, type=None) -> URIRefEntity:
        '''
        Get entity by index in store.
        :param index: index of entity.
        :param type: "SubjectObject" or "Predicate".
        :return: URIRefEntity at index
        '''
        entity_list = self.entities
        if type is not None:
            entity_list = [e for e in entity_list if e.type == type]
        return entity_list[index]

    def snapshot(self):
        '''
        Create a snapshot of all internal stores.
        '''
        self.entities_snapshot = copy.deepcopy(self.entities)
        self.lookup_p_snapshot = copy.deepcopy(self.lookup_pred)
        self.lookup_so_snapshot = copy.deepcopy(self.lookup_subobj)

    def restore_snapshot(self):
        '''
        Sets current stores to snapshot state.
        '''
        self.entities = copy.deepcopy(self.entities_snapshot)
        self.lookup_pred = copy.deepcopy(self.lookup_p_snapshot)
        self.lookup_subobj = copy.deepcopy(self.lookup_so_snapshot)






