DATA_VER = '01'
KG_RAW = 'kg_mapped_manual.csv'

EXTRACT_CONFIG = {
    'data_path': './synaptix/example_data/',
    'out_name' : 'kg_SYNAPTIX.csv',
    'map_tmplt' : {
            'relation': '',
            'display_relation': 'relationship_type',
            'x_id': 's.uri', 
            'x_index': 'ID(s)', 
            'x_type': 'head(labels(s))',
            'x_name': 's.prefLabel',
            'x_source': 's.uri', 
            'y_id': 't.uri',
            'y_index': 'ID(t)',
            'y_type': 'head(labels(t))',
            'y_name': 't.prefLabel',
            'y_source': 't.uri', 
            'edge_source': 'r.provenance',
            'relation_synaptix': 'type(r)',
        },
    'filter_node' : {
        'gene': {'species': 'human'}
        }
}
MAP_CHUNCK = 1000
URI_MANUAL_MAP = {
    "'purl.obolibrary.org', 'obo', 'HP_" : "HP",
    "'purl.obolibrary.org', 'obo', 'MONDO_" : "MONDO",
    "'purl.obolibrary.org', 'obo', 'DOID_" : "DOID",
    "www.orpha.net" : "ORPHANET",
    "www.ebi.ac.uk": "EFO",
    "ns.merckgroup.com": "HUMAN PROTEIN ATLAS",
    "identifiers.org": "SNOMED"
}