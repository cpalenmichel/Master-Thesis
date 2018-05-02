from xml.dom import minidom

class ISFile:
    def __init__(self, file):
        self.markables = []
        self.read(file)

    def read(self, filename):
        xmldoc = minidom.parse(filename)
        markables = xmldoc.getElementsByTagName('markable')

        for markable in markables:
            markable_to_add = Markable()
            markable_to_add.id = markable.attributes['id'].value.strip()
            markable_to_add.is_status = markable.attributes['information_status'].value
            span =  markable.attributes['span'].value.replace('word_', '')
            span_list = span.split('..')
            if len(span_list) > 1:
                markable_to_add.span_start = int(span_list[0])
                markable_to_add.span_end = int(span_list[1])
            else:
                markable_to_add.span_start = int(span_list[0])
                markable_to_add.span_end = int(span_list[0])
            if 'mediated_type' in markable.attributes:
                markable_to_add.mediated_type = markable.attributes['mediated_type'].value
                if markable_to_add.mediated_type == 'bridging':
                    markable_to_add.bridge_type = markable.attributes['bridge_type'].value
                    markable_to_add.bridged_from = markable.attributes['bridged_from'].value
                    if 'second_bridged_from' in markable.attributes:
                        markable_to_add.second_bridged_from = markable.attributes['second_bridged_from'].value
                    if 'third_bridged_from' in markable.attributes:
                        markable_to_add.third_bridged_from = markable.attributes['third_bridged_from'].value
            self.markables.append(markable_to_add)

class Markable:
    def __init__(self):
        # id
        self.id = None
        # IS notes spans
        self.span_start = None
        self.span_end = None

        # IS Status
        self.is_status = None
        # Mediated_type
        self.mediated_type = None
        # Bridging only attributes
        self.bridged_from = None
        self.bridge_type = None
        self.second_bridged_from = None
        self.third_bridged_from = None

    def __repr__(self):
        # TODO think of a better representation
        return self.id + ' ' + str(self.span_start) + '--' + str(self.span_end)

if __name__ == "__main__":
    test_file = ISFile('ISAnnotationWithoutTrace/wsj_1004_entity_level.xml')
    for markable in test_file.markables:
        print(markable)
        print(markable.bridged_from)
        print()