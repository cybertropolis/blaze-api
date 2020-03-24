from nltk.corpus import wordnet as wn


class SynsetGenerator:
    """ """

    def __init__(self):
        pass

    def get_synset_labels(self, words):
        """ """
        return

    def get_synset_offset_and_labels(self, words):
        """ """
        synsets = [wn.synset(word) for word in words]
        for synset in synsets:
            synset_label = '{0}{1}'.format(
                synset.pos(), str(synset.offset().zfill(8)))
            synset_label_to_human = ' '.join(synset.lemma_names())
            yield synset_label, synset_label_to_human
