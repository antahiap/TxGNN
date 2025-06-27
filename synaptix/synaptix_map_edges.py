
map_e = {
    'protein_protein':{
        'status': 'not available',
        'source': ['target'],
        'target': ['target'],
        'edges': [],
        'skip_list': [],
        'comment': '',
    },
    'drug_protein':{ 
        'status': 'ok',
        'source': ['drug'],
        'target': ['target'],
        'edges': ['target'],
        'comment': '',
        'skip_list': [],
    },
    'contraindication':{  
        'status': 'ok',
        'source':  ['drug'],
        'target': ['disease'],
        'edges': ['contraindicated_drug'],
        'comment': '',
        'skip_list': [],
    },
    'indication':{ 
        'status': 'ok',
        'source': ['drug'], 
        'target': ['disease'],
        'edges': ['indicated_drug'],
        'comment': '',
        'skip_list': [], 'map': {}
    },
    'off-label use':{ 
        'status': 'ok-Natalie',
        'source': [''],
        'target': [''],
        'edges': ['off_label_drug'], 
        'comment': '',
        'skip_list': [], 'map': {}
    },
    'drug_drug': { 
        'status': 'ok',
        'source': ['drug'],
        'target': ['drug'], 
        'edges': ['is_alternative_form_of_other_drug'], 
        'comment': '.',
        'skip_list': [], 
    },
    'phenotype_protein': { 
        'status': 'not available',
        'source': ['gene'],
        'target': ['effect/phenotype'],
        'comment': 'no node, effect/phenotype.',
        'edges': [],
        'skip_list': [], 'map': {},
        'query': ' '
    },
    'phenotype_phenotype':{
        'status': 'not available', 
        'source': ['effect/phenotype'],
        'target': ['effect/phenotype'],
        'edges': [],
        'comment': 'no node, effect/phenotype.',
        'skip_list': [], 'map': {}
    },
    'disease_phenotype_negative':{ 
        'status': 'not available-Natalie',  
        'source': [''],
        'target': [''],
        'edges': [],
        'comment': 'only positive is available.',
        'skip_list': [], 'map': {}
    },
    'disease_phenotype_positive':{
        'status': 'ok-Natalie', 
        'source': [''],
        'target': [''],
        'edges': ['phenotype_diagnostic', 'phenotype_frequent', 'phenotype_occasional', 'phenotype_pathognomonic', 'phenotype_rare', 'phenotype_very_frequent'],
        'comment': 'no node, effect/phenotype.',
        'skip_list': [], 'map': {}
    },
    'disease_protein': {
        'status': 'ok',
        'source': ['disease'],  
        'target': ['target'],
        'edges': ['target___all'],
        'skip_list': ['is_a', 'member_of'],
        'comment': '',
    },
    'disease_disease': {  
        'status': 'ok',
        'source': ['disease'],
        'target': ['disease'], 
        'edges': ['related_concept'],
        'skip_list': [],
        'comment': 'no relations, all relations between disese are, is_a.',
    },
    'drug_effect':{
        'status': 'ok-Natalie',
        'source': [''], 
        'target': [''],
        'edges': ['phenotype_always_present'],
        'comment': 'no node, effect',
        'skip_list': [], 'map': {}
    },
    'bioprocess_bioprocess':{
        'status': 'not available',
        'source': [],
        'target': [],
        'edges': [],
        'comment': 'no node, bioprocess.',
        'skip_list': [], 'map': {}
    },
    'molfunc_molfunc': {
        'status': 'not available',
        'source': ['molecular_function'],
        'target': ['molecular_function'],
        'edges': [],
        'skip_list': ['is_a'],
        'comment': 'no relations, all relations between nodes are, is_a.',
    },
    'cellcomp_cellcomp':{
        'status': 'not available',
        'source': [''],
        'target': [''],
        'edges': [],
        'comment': 'no node, cellular_component',
        'skip_list': [], 'map': {}
    },
    'molfunc_protein':{
        'status': 'not available',
        'source': ['molecular_function'],
        'target': ['target'],
        'edges': [],
        'skip_list': [],
        'comment': 'no relation available',
        'map':{
            'relation': 'r.relationship_type',
            'display_relation': 'relationship_type',
            'x_id': 's.did',
            'y_id': 't.did'
        }
    },
    'cellcomp_protein':{
        'status': 'ok-Natalie',
        'source': [''],
        'target': [''],
        'edges': ['HAS_PREDICTED_LOCATION'],
        'comment': 'no node, cellular_component',
        'skip_list': [], 'map': {}
    },
    'bioprocess_protein':{
        'status': 'ok-Natalie',
        'source': [''],
        'target': [''],
        'edges': ['HAS_MECHANISM_OF_ACTION_BROAD_TERM'],
        'comment': 'no node, bioprocess.',
        'skip_list': [], 'map': {}
    },
    'exposure_protein':{
        'status': 'not available',
        'source': [],
        'target': [],
        'edges': [],
        'comment': 'no node, exposure',
        'skip_list': [], 'map': {}
    },
    'exposure_disease':{
        'status': 'not available',
        'source': [],
        'target': [],
        'edges': [],
        'comment': 'no node, exposure',
        'skip_list': [], 'map': {}
    },
    'exposure_exposure':{
        'status': 'not available',
        'source': [],
        'target': [],
        'edges': [],
        'comment': 'no node, exposure',
        'skip_list': [], 'map': {}
    },
    'exposure_bioprocess':{
        'status': 'not available',
        'source': [],
        'target': [],
        'edges': [],
        'comment': 'no node, exposure',
        'skip_list': [], 'map': {}
    },
    'exposure_molfunc':{
        'status': 'not available',
        'source': [],
        'target': [],
        'edges': [],
        'comment': 'no node, exposure',
        'skip_list': [], 'map': {}
    },
    'exposure_cellcomp':{
        'status': 'not available',
        'source': [],
        'target': [],
        'edges': [],
        'comment': 'no node, exposure',
        'skip_list': [], 'map': {}
    },
    'pathway_pathway':{
        'status': 'not available',
        'source': ['pathway'],
        'target': ['pathway'],
        'edges': [],
        'skip_list': ['is_a'],
        'comment': 'no relations between pathways',
        'map':{
            'relation': 'r.relationship_type',
            'display_relation': 'relationship_type',
            'x_id': 's.did',
            'y_id': 't.did'
        }
    },
    'pathway_protein':{
        'status': 'ok-Natalie',
        'source': [''],
        'target': [''],
        'edges': ['drug_effect'],
        'skip_list': ['is_a', ],
        'comment': 'no relations, only relation available is member_of',
    },
    'anatomy_anatomy':{
        'status': 'not available',
        'source': [],
        'target': [],
        'edges': [],
        'comment': 'no node, anatomy',
        'skip_list': [], 'map': {}
    },
    'anatomy_protein_present':{
        'status': 'ok-Natalie',
        'source': [''],
        'target': [''],
        'edges': ['HAS_TISSUE_LABEL', 'HAS_CELL_LABEL'],
        'comment': 'no node, anatomy',
        'skip_list': [], 'map': {}
    },
    'anatomy_protein_absent':{
        'status': 'Not available',
        'source': [''],
        'target': [''],
        'edges': [],
        'comment': 'only present is available.',
        'skip_list': [], 'map': {}
    },
}
