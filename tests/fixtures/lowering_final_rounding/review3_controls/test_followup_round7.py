import importlib.util,pathlib,pytest,numpy as np
R=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location("independent",R/"test_independent_round7.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.OUT=R/"runs/followup";p.OUT.mkdir(parents=True,exist_ok=True)
def helper(payload,module="numpy"):
 return f'def mutate():\n    from {module} import __builtins__ as space\n    space["exec"]({payload!r},space["globals"]())\n    return 0\n'
CASES={
"helper_default":helper("DT = 'fp32'")+'def marker(arg=mutate()):\n    pass\n',
"helper_keyword_default":helper("DT = 'fp32'")+'def marker(*, arg=mutate()):\n    pass\n',
"helper_module_annotation":helper("DT = 'fp32'")+'flag: mutate() = 0\n',
"helper_method_default":helper("DT = 'fp32'")+'class C:\n    def marker(self,arg=mutate()):\n        pass\n',
"helper_dataclasses_alias":helper("DT = 'fp32'","dataclasses")+'def marker(arg=mutate()):\n    pass\n',
"kernel_object_replaced":helper("kernel = lambda f: f")+'def marker(arg=mutate()):\n    pass\n',
}
@pytest.mark.parametrize("fp64",[True,False])
@pytest.mark.parametrize("name",list(CASES))
def test_alias_definition_time_side_effects_fail_closed(name,fp64):
 r=p.observe(name,CASES[name],fp64)
 assert "error" not in r,r
 assert not r["lowered"] and r["declines"] and r["declines"][0].startswith("kernel_output_dtype_unestablished:k"),r
 np.testing.assert_array_equal(r["public"],r["slow"])

