import json,pathlib,shutil
import yaml
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
R=pathlib.Path(__file__).resolve().parent
def test_explicit_named_dim_uses_its_declared_extent(tmp_path):
 source=R/'test-output/provenance_head/test_parsed_body_wins_over_bui0/dsl'
 dest=tmp_path/'dsl';shutil.copytree(source,dest)
 p=translate_with_einsum(dest,tmp_path/'einsum');raw=yaml.safe_load(p.read_text())
 a=EinsumGraphAnalyzer().analyze_graph(str(p),str(tmp_path/'analysis'),precision='fp32',copy_graph=False)
 (tmp_path/'dimension-observation.json').write_text(json.dumps({'source':str(source),'raw':raw,'analysis':a},indent=2)+'\n')
 # The saved trace declares I=2 and H=4. Dim("I") * Dim("H") is 8 iterations.
 assert raw['layers']['gather#0']['loop_iterations']==8
