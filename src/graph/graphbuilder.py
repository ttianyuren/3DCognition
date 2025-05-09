class GraphBuilder:
    def __init__(self, llm=None):
        self.llm = llm

    def get_objects(self, llm_response):
        objects = llm_response.split('[')[1].split(']')[0].split(',')
        objects = objects[:5]
        for i in range(len(objects)):
            objects[i] = objects[i].strip()
        return objects

    def get_relations(self, llm_response, objects):
        relations = []
        for line in llm_response.split('\n'):
            relation = {'source': '', 'target': '', 'type': ''}
            parts = line.strip().split(': ')
            if len(parts) == 2:
                relation_info = parts[1].strip()
                relation_parts = relation_info.split(' is ')
                if len(relation_parts) == 2:
                    source_target = parts[0].strip().split(' and ')
                    if len(source_target) == 2:
                        source = source_target[0].strip()
                        target = source_target[1].strip()
                        relation_type = relation_parts[1].strip()
                        if source in objects and target in objects:
                            relation = {'source': source, 'target': target, 'type': relation_type}
                            relations.append(relation)
        return relations
    
    def get_relations_objects(self, llm_response):
        relations = []
        objects = []
        for line in llm_response.split('\n'):
            relation = {'source': '', 'target': '', 'type': ''}
            parts = line.strip().split(': ')
            if len(parts) == 2:
                relation_info = parts[1].strip()
                relation_parts = relation_info.split(' is ')
                if len(relation_parts) == 2:
                    source_target = parts[0].strip().split(' and ')
                    if len(source_target) == 2:
                        source = source_target[0].strip().lower()
                        target = source_target[1].strip().lower()
                        relation_type = " ".join(relation_parts[1].strip().split()[:-1]).strip().lower()
                        if source not in objects:
                            objects.append(source)
                        if target not in objects:
                            objects.append(target)
                        relation = {'source': source, 'target': target, 'type': relation_type}
                        relations.append(relation)
        return relations, objects

    def parse_text_description(self, description):
        # object_prompt = f"""
        # Please extract all objects mentioned in the following text description and the output format is "[<object 1>, <object 2>,...]":
        # Text Description: {description}
        # """
        # object_response = self.llm(object_prompt)
        # try:
        #     objects = self.get_objects(object_response)
        # except:
        #     objects = []

        relation_prompt = f"""
        Please describe the relationships between the objects in the following text description and provide the information in the specified format.
        The format should be: "<Object A> and <Object B>: <Object A> is <relation type> <Object B>".
        If there are multiple relationships, please list them one per line.

        Text Description: {description}

        Example output format:
        Book and Table: Book is on Table
        """
        relation_response = self.llm(relation_prompt)
        relations, objects = self.get_relations_objects(relation_response)

        return objects, relations

    def build_graph(self, objects, relations):
        graph = {
            'nodes': [{'id': obj} for obj in objects],
            'edges': [{'source': r['source'], 'target': r['target'], 'type': r['type']} for r in relations]
        }
        return graph

    def build_graph_from_text(self, text_goal):
        objects, relations = self.parse_text_description(text_goal)
        graph = self.build_graph(objects, relations)
        print("<><><><><><> New built Goal Graph:", graph)
        return graph
