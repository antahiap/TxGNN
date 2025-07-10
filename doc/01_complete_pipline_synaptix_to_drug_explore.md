# Complete Pipline Synaptix Data to DataExplore TxGNN



## Table of Contents
- [Data Extraction](#)
    - [Data Matching - Classes From Synaptix Matching To Prime_Kg](#data-matching---classes-from-synaptix-matching-to-prime_kg)
    - [Data Source Mapping](#data-source-mapping)
    - [Data Varification and output for training](#data-varification-and-output-for-training)
- [TxGNN Training - Pretraining And Finetuning](#)
    - [TxGNN Data Matching](#)
    - [Training Log](#)
    - [Model Outcome Comparison](#)
- [TxGNN Training - Explainability](#)
- [Drug Explore](#)
    - [Database load](#)
    - [Server Data Prepration](#)
- [edge source contribution](#)




## 1. Data Extraction: matching classes From Synaptix Matching To Prime_Kg

Set the configuartions in `submodules/TxGNN/synaptix/constants.py`, `submodules/TxGNN/synaptix/synaptix_map_edges.py` and `.env`. The exmple of test out put is provided in `.images/neo4j/example_data`.

### Senario 1: Python Code Directly
Extraction from neo4j: Extract data from neo4j based on the mapping in `submodules/TxGNN/synaptix/synaptix_map_edges.py`. The focus of data extract is finding the correspondent edge in the new database that is reperentetive of primekg. Later the nodes are extracted. For testing the setup:

```
cd ./submodules/TxGNN/
conda activate env-txgnn
python synaptix/extract_kg.py test_opt
```

for final run, recommended to run on background as it might take time
```
cd ./submodules/TxGNN/
nohup python synaptix/extract_kg.py > synaptix/extract_kg.log 2>&1 &
```

** NOTE **
- You might need to set the neo4j database info in `.env` file.
- The config in `extract_kg.py` defines the path for output as well as how the maping is done. 

Extracting data from neo4j and setting it up for TxGNN training. The focus of data extract is finding the correspondent edge in the new database that is reperentetive of primekg. Later the nodes are extracted. The main output of this pipline is `node.csv`, `edge.csv`, `kg.csv` that are formated as primekg output that is used in txgnn training.

### Scenario 2: Running Python Notebook
use the template `submodules/TxGNN/synaptix/synaptix_map_edges.py` to define edges to be extracted from the 
`notebooks/16_synaptix.ipynb` includes the notebook for exploring this template and compare it with primekg data.

## 2. Data Source Mapping
As the sources are not available extracted in neo4j, the source mapping is done in two step from the nodes uri. First here with the code `submodules/TxGNN/synaptix/prep_data.py`, that uses a list of sources saved manually from synaptix and dumped as txt, (url)[https://synaptix.sxp.merckgroup.com/datasets], stored in `submodules/TxGNN/synaptix/source_map.py`, as it is manual copy of table, each thirs row contains the source name that is extracted with `prep_data.py`. As the extraction doesn't work for all the source names, the final stape is done in data varification. 

As the mapping of source and uris didn't work perfectly a second step is added which used a manual config to update the names.

### Senario 1: Python Code Directly

`submodules/TxGNN/synaptix/prep_data.py` extract all the data and store it in `.images/neo4j/data_synaptix` with `kg_mapped.csv` and `kg_mapped_manual.csv`as further process is needed. Which includes:
    - source mapping, this data for id and sources has the uri of x and y nodes.
    - removing disconnected graphs.
    - renumbering, to sort the nodes based on the x node type and renumber.database.


```
cd ./submodules/TxGNN/
python synaptix/prep_data.py test_opt

```
### Scenario 2: Running Python Notebook
The code was explored in `notebooks/16_synaptix_post.ipynb` .

### Data Varification and output for training

### Senario 1: Python Code Directly

You can use the following command, you can skip test_opt if you want the complete outcome. The data_ver is set in `submodules/TxGNN/synaptix/constants.py`. 

```
python synaptix/kg_to_node_edges.py test_opt
```

### Scenario 2: Running Python Notebook
Here, we look at the stored data, renumber them, delete disconnected nodes and finally save it as ready for training in primekg format of, `kg.csv`, `nodes.csv` and `edge.csv`. The process is done in  `notebooks/16_synaptix_post.ipynb`. 


## TxGNN Training - Pretraining And Finetuning
### TxGNN Data Matching
Data extractiuon section tries to match the data as much as possible to primekg to avoid much of the update in the code level. However, there some mijor update still needed. The code requires lists to know which nodes and edges are considered ad drug or diseas or edges for them. 
### Training Log
### Model Outcome Comparison

## TxGNN Training - Explainability
### Drug Explore
### Database load

## Server Data Prepration

## edge source contribution

