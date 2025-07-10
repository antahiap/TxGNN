from pathlib import Path
import pandas as pd
from synaptix_map_edges import map_e
import kg_to_node_edges as kg_ne

class Synaptix:
    def __init__(self) -> None:  
        self.data_path = Path( '../../.images/neo4j/data_synaptix/')
        self.opt = {}
        self.attr_list= ['id', 'type', 'name', 'source', 'uri']


    def ver_(self):
        ver = ''

        file_in = self.data_path / Path('kg_mapped_manual.csv')
        out_dir = self.data_path / Path(ver)

        print('Make data in')
        print(out_dir)
        data = kg_ne.DataPost(
            out_dir, file_in, attr_list=self.attr_list, **self.opt)

        data.make_graph_output_synaptix(opt='01')


    def ver_03(self):
        ver = '03'


        file_in = self.data_path / Path('kg_mapped_manual.csv')
        out_dir = self.data_path / Path(ver)

        print('Make data in')
        print(out_dir)
        data = kg_ne.DataPost(
            out_dir, file_in, attr_list=self.attr_list, **self.opt)

        data.make_graph_output_synaptix(opt='02')


class PrimeKg:
    def __init__(self) -> None:  
        self.data_path = Path( '../../.images/neo4j/data_primekg/')
        self.opt = {}
        self.attr_list= ['id', 'type', 'name', 'source']
        
    def ver_02(self):
        ver = '02'

        file_in = self.data_path / Path('kg.csv')
        out_dir = self.data_path / Path(ver)

        print('Make data in')
        print(out_dir)
        data = kg_ne.DataPost(
            out_dir, file_in, attr_list=self.attr_list, **self.opt)
        data.make_g(data.kg_raw)
        data.out_put_G(data.G)

    def ver_03(self):
        # Only works for directed graph. 
        ver = '03'

        file_in = self.data_path / Path('kg.csv')
        out_dir = self.data_path / Path(ver)

        print('Make data in')
        print(out_dir)
        data = kg_ne.DataPost(
            out_dir, file_in, attr_list=self.attr_list, **self.opt)
        
        pkg = data.kg_raw

        df = pd.DataFrame.from_dict(map_e, orient='index').reset_index()
        df = df.rename(columns={'index': 'edge_type'})
        exclude_edge = df[~df['status'].str.startswith('ok')]['edge_type'].to_list()

        pkg_filtered = pkg[~pkg['relation'].isin(exclude_edge)]
        print(f'new number of edges:{len(pkg_filtered)}, from {len(pkg)}')
        
        data.kg = pkg_filtered

        attr_list= self.attr_list
        data.make_g(pkg_filtered)
        data.out_put_G(data.G)


if __name__ == '__main__':
    synptx = Synaptix()
    # ver 01 and 02 are outputed from 016 notebooks
    # synptx.ver_()


    pkg = PrimeKg()
    # pkg.ver_02()
    # pkg.ver_03()