

# map BIOES tags
class TagVocab():
    def __init__(self, type_list = None):
        self.PAD_TAG = '<PAD>'

        pos_list = ['B', 'I', 'E', 'S']
        if type_list == None:
            type_list = ['疾病和诊断', '检查', '检验', '手术', '药物', '解剖部位']
        tag_list = []
        for pos in pos_list:
            tag_list = tag_list + [pos + '-' + types for types in type_list]
        tag_list.append('O')
        self.tag_to_ix = {tag_list[i]: i + 1 for i in range(len(tag_list))}
        self.tag_to_ix[self.PAD_TAG] = 0
        self.ix_to_tag = {value: key for key, value in self.tag_to_ix.items()}

    def map_tag(self, tags, dim = 2):
        if dim == 2:
            return [self.tag_to_ix[tag] for tag in tags]
        if dim == 1:
            return self.tag_to_ix[tags]